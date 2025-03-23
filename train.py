import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloader
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint and samples directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize ROI detector
    roi_detector = ROIDetector(confidence=args.yolo_confidence, force_cpu=True, use_face_detection=True)
    
    # Initialize diffusion model
    diffusion_model = DiffusionModel(device=device)
    
    # Create optimizer
    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            diffusion_model.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")
    
    # Create dataloader
    train_dataloader = get_dataloader(
        args.data_dir, 
        batch_size=args.batch_size,
        transform=None  # Use default transform in dataset.py
    )
    
    # Training loop
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                images = batch['image'].to(device)
                keys = batch['key'].to(device)
                
                # Detect ROIs for each image
                batch_masks = []
                for i in range(images.shape[0]):
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
                
                print(f"Masks shape: {masks.shape}")
                
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
                        
                        # Add images to tensorboard
                        writer.add_image('Images/Original', vutils.make_grid((images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Encrypted', vutils.make_grid((encrypted_images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Decrypted', vutils.make_grid((decrypted_images[:4].clamp(-1, 1) * 0.5 + 0.5), nrow=4), global_step)
                        writer.add_image('Images/Masks', vutils.make_grid(masks[:4].unsqueeze(1).repeat(1, 3, 1, 1), nrow=4), global_step)
                        
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
    
    args = parser.parse_args()
    
    train(args) 