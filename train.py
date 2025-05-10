import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloader, get_ham10000_dataloaders
from diffusion_model import DiffusionModel
from roi_detector import ROIDetector
from metrics import calculate_metrics

def save_images(images, filename, nrow=4):
    """Save a grid of images to a file"""
    # Clamp images to [0, 1] range
    images = (images.clamp(-1, 1) * 0.5 + 0.5)
    # Save to file
    vutils.save_image(images, filename, nrow=nrow, padding=2, normalize=False)
    
def train(args):
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set a smaller batch size if training on CPU
    if device.type == 'cpu' and args.batch_size > 4:
        print(f"Reducing batch size from {args.batch_size} to 4 for CPU training")
        args.batch_size = 4
    
    # Load checkpoint if specified
    use_key_image = not args.use_gaussian_noise  # Default based on command line arg
    start_epoch = 0
    
    # Create checkpoint and samples directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize ROI detector based on dataset type
    if args.dataset_type == 'ham10000':
        roi_detector = None  # No ROI detection needed for HAM10000
        print("Using full image encryption for HAM10000 dataset")
    else:
        roi_detector = ROIDetector(confidence=args.yolo_confidence, force_cpu=True, 
                                  use_face_detection=True)
        print("Using face detection for ROI detection")
    
    # Initialize diffusion model
    diffusion_model = DiffusionModel(device=device)
    
    # Create optimizer
    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=args.learning_rate)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            diffusion_model.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            # Load encryption method choice if it exists in the checkpoint
            if 'use_key_image' in checkpoint:
                use_key_image = checkpoint['use_key_image']
                print(f"Loaded encryption method: {'Key Image' if use_key_image else 'Gaussian Noise'}")
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")
    
    # If command line argument wasn't provided and not resuming from checkpoint, ask user
    if not args.use_gaussian_noise and not args.resume:
        user_input = input("Use key image for encryption? (y/n, default: y): ").lower().strip()
        if user_input == 'n' or user_input == 'no':
            use_key_image = False
            print("Using Gaussian noise for encryption instead of key images")
        else:
            print("Using key images for encryption")
    elif args.use_gaussian_noise:
        use_key_image = False
        print("Using Gaussian noise for encryption (from command line argument)")
    else:
        print(f"Using {'Key Image' if use_key_image else 'Gaussian Noise'} for encryption")
    
    # Create dataloader based on dataset type
    if args.dataset_type == 'ham10000':
        print(f"Loading HAM10000 dataset from {args.data_dir}")
        train_dataloader, val_dataloader, _ = get_ham10000_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size,
            use_same_keys=not use_key_image,  # If not using key images, generate fixed noise keys
            transform=None  # Use default transform in dataset.py
        )
    else:
        print(f"Loading standard dataset from {args.data_dir}")
        train_dataloader = get_dataloader(
            args.data_dir, 
            batch_size=args.batch_size,
            transform=None  # Use default transform in dataset.py
        )
        val_dataloader = None
    
    # Training loop
    global_step = start_epoch * len(train_dataloader)
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        # Training phase
        diffusion_model.model.train()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                images = batch['image'].to(device)
                original_keys = batch['key'].to(device)
                
                # If using Gaussian noise instead of key images
                if not use_key_image:
                    # Generate random Gaussian noise with same shape as keys
                    keys = torch.randn_like(original_keys)
                else:
                    keys = original_keys
                
                # Detect ROIs for each image - use different approaches based on dataset
                batch_masks = []
                for i in range(images.shape[0]):
                    if args.dataset_type == 'ham10000':
                        # For HAM10000, use full image encryption
                        mask = torch.ones((images.shape[2], images.shape[3]), device=device)
                        mask = mask.unsqueeze(0)  # Add channel dimension
                    else:
                        # For general images, use bounding box detection
                        _, roi_coords = roi_detector.detect_rois(images[i].cpu())
                        
                        if not roi_coords:  # If no ROIs detected, use the entire image
                            mask = torch.ones((images.shape[2], images.shape[3]), device=device)
                        else:
                            mask = roi_detector.apply_rois_to_mask(images[i].shape, roi_coords)
                        
                        # Add channel dimension
                        mask = mask.unsqueeze(0).to(device)
                    
                    batch_masks.append(mask)
                
                # Stack masks along batch dimension
                masks = torch.stack(batch_masks, dim=0)
                
                # Train step
                loss = diffusion_model.train_step(images, keys, optimizer, masks)
                epoch_losses.append(loss)
                
                # Log to tensorboard
                writer.add_scalar('Loss/train', loss, global_step)
                
                # Visualize intermediate results every args.sample_interval steps
                if global_step % args.sample_interval == 0:
                    with torch.no_grad():
                        # Get a random noise level for encryption
                        t = torch.ones(images.shape[0], device=device).long() * min(999, args.encryption_timestep)
                        
                        # Encrypt images
                        encrypted_images, _ = diffusion_model.encrypt(images, keys, masks, t)
                        
                        # Decrypt images
                        decrypted_images, _ = diffusion_model.decrypt(encrypted_images, keys, t, masks)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(images, encrypted_images, decrypted_images)
                        
                        # Log metrics
                        for metric_name, metric_value in metrics.items():
                            writer.add_scalar(f'Metrics/{metric_name}', metric_value, global_step)
                        
                        # Save images
                        sample_images = torch.cat([
                            images[:4],
                            encrypted_images[:4],
                            decrypted_images[:4],
                        ], dim=0)
                        
                        save_images(
                            sample_images, 
                            os.path.join(args.samples_dir, f'sample_epoch{epoch+1}_step{global_step}.png'),
                            nrow=4
                        )
                        
                        # Save masks for visualization
                        if masks.shape[1] == 1:  # If mask has channel dimension
                            mask_display = masks[:4].repeat(1, 3, 1, 1)
                        else:
                            mask_display = masks[:4].unsqueeze(1).repeat(1, 3, 1, 1)
                        
                        # Add images to tensorboard
                        writer.add_image('Images/Original', vutils.make_grid((images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Encrypted', vutils.make_grid((encrypted_images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Decrypted', vutils.make_grid((decrypted_images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Masks', vutils.make_grid(mask_display, nrow=4), global_step)
                        
                        # Print metrics
                        print(f"\nStep {global_step} metrics:")
                        print(f"  Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
                        print(f"  Reconstruction PSNR: {metrics['reconstruction_psnr']:.2f}")
                        print(f"  Reconstruction SSIM: {metrics['reconstruction_ssim']:.4f}")
                        print(f"  Encryption MSE: {metrics['encryption_mse']:.6f}")
                        print(f"  Original Entropy: {metrics['original_entropy']:.4f}")
                        print(f"  Encrypted Entropy: {metrics['encrypted_entropy']:.4f}")
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': loss})
                global_step += 1
        
        # Validation phase
        if val_dataloader is not None:
            diffusion_model.model.eval()
            val_losses = []
            
            with torch.no_grad():
                with tqdm(total=len(val_dataloader), desc="Validation") as pbar:
                    for batch_idx, batch in enumerate(val_dataloader):
                        images = batch['image'].to(device)
                        original_keys = batch['key'].to(device)
                        
                        # If using Gaussian noise instead of key images
                        if not use_key_image:
                            # Generate random Gaussian noise with same shape as keys
                            keys = torch.randn_like(original_keys)
                        else:
                            keys = original_keys
                        
                        # Detect ROIs for each image
                        batch_masks = []
                        for i in range(images.shape[0]):
                            # For skin lesions, use detailed segmentation masks
                            mask = roi_detector.create_detailed_lesion_mask(images[i].cpu())
                            mask = mask.unsqueeze(0).to(device)  # Add channel dimension
                            batch_masks.append(mask)
                        
                        # Stack masks along batch dimension
                        masks = torch.stack(batch_masks, dim=0)
                        
                        # Compute batch_size random timesteps
                        t = torch.randint(0, diffusion_model.num_timesteps, (images.shape[0],), device=device)
                        
                        # Add noise to images
                        noise = torch.randn_like(images)
                        noisy_images = diffusion_model.q_sample(images, t, noise)
                        
                        # Apply mask if provided
                        if masks is not None:
                            # Only add noise to masked regions for validation
                            target_images = images * (1 - masks) + noisy_images * masks
                            # For validation, we want the model to predict what was added (full noise for masked regions, zero for others)
                            target_noise = noise * masks
                        else:
                            target_images = noisy_images
                            target_noise = noise
                        
                        # Forward pass
                        predicted_noise = diffusion_model.model(noisy_images, keys, t)
                        
                        # Calculate loss (mean squared error)
                        val_loss = torch.nn.functional.mse_loss(predicted_noise, target_noise).item()
                        val_losses.append(val_loss)
                        
                        pbar.update(1)
                        pbar.set_postfix({'val_loss': val_loss})
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss:.6f}")
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model': diffusion_model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'use_key_image': use_key_image,
                }, best_model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        
        # End of epoch, save checkpoint
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model': diffusion_model.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
            'use_key_image': use_key_image,  # Save the encryption method choice
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Close tensorboard writer
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image encryption-decryption model")
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--samples_dir', type=str, default='./samples', help='Directory to save sample images')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sample_interval', type=int, default=100, help='Interval for generating and saving samples')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training')
    
    # Model parameters
    parser.add_argument('--encryption_timestep', type=int, default=500, help='Timestep to use for encryption during sampling')
    parser.add_argument('--yolo_confidence', type=float, default=0.25, help='Confidence threshold for YOLO ROI detection')
    parser.add_argument('--use_gaussian_noise', action='store_true', help='Use Gaussian noise instead of key image for encryption')
    
    # Dataset parameters
    parser.add_argument('--dataset_type', type=str, choices=['standard', 'ham10000'], default='standard',
                       help='Type of dataset to use (standard or HAM10000)')
    
    args = parser.parse_args()
    
    train(args) 