import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from diffusion_model import DiffusionModel
from roi_detector import ROIDetector
from metrics import calculate_metrics

def load_image(path, transform=None):
    """Load an image from path and apply transforms"""
    image = Image.open(path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def save_images_with_metrics(original, encrypted, decrypted, save_path, metrics=None):
    """Save a figure with original, encrypted, and decrypted images with metrics"""
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = (original.clamp(-1, 1) * 0.5 + 0.5).cpu()
        encrypted = (encrypted.clamp(-1, 1) * 0.5 + 0.5).cpu()
        decrypted = (decrypted.clamp(-1, 1) * 0.5 + 0.5).cpu()
        
        # Save individual images
        save_image(original, os.path.join(save_path, 'original.png'))
        save_image(encrypted, os.path.join(save_path, 'encrypted.png'))
        save_image(decrypted, os.path.join(save_path, 'decrypted.png'))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    if isinstance(original, torch.Tensor):
        axes[0].imshow(original.permute(1, 2, 0).numpy())
        axes[1].imshow(encrypted.permute(1, 2, 0).numpy())
        axes[2].imshow(decrypted.permute(1, 2, 0).numpy())
    else:
        axes[0].imshow(original)
        axes[1].imshow(encrypted)
        axes[2].imshow(decrypted)
    
    # Set titles
    axes[0].set_title('Original Image')
    axes[1].set_title('Encrypted Image')
    axes[2].set_title('Decrypted Image')
    
    # Add metrics as text if provided
    if metrics:
        metrics_text = "\n".join([
            f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}",
            f"Reconstruction PSNR: {metrics['reconstruction_psnr']:.2f}",
            f"Reconstruction SSIM: {metrics['reconstruction_ssim']:.4f}",
            f"Encryption MSE: {metrics['encryption_mse']:.6f}",
            f"Original Entropy: {metrics['original_entropy']:.4f}",
            f"Encrypted Entropy: {metrics['encrypted_entropy']:.4f}",
            f"Decrypted Entropy: {metrics['decrypted_entropy']:.4f}"
        ])
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Remove axes
    for ax in axes:
        ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison.png'), bbox_inches='tight')
    plt.close()

def run_inference(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    diffusion_model = DiffusionModel(device=device)
    if args.model_path:
        diffusion_model.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print("No model path provided, using randomly initialized model")
    
    # Initialize ROI detector based on the segmentation method chosen
    use_skin_lesion = args.use_skin_lesion_segmentation
    roi_detector = ROIDetector(
        confidence=args.yolo_confidence, 
        force_cpu=True, 
        use_face_detection=not use_skin_lesion,
        use_skin_lesion_segmentation=use_skin_lesion
    )
    
    print(f"Using {'skin lesion segmentation' if use_skin_lesion else 'YOLO face detection'} for ROI detection")
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load image and key
    image = load_image(args.image_path, transform).unsqueeze(0).to(device)
    key = load_image(args.key_path, transform).unsqueeze(0).to(device)
    
    print(f"Loaded image from {args.image_path}")
    print(f"Loaded key from {args.key_path}")
    
    # Detect ROIs based on the chosen segmentation method
    if use_skin_lesion:
        # For skin lesions, use detailed segmentation masks
        detailed_mask = roi_detector.create_detailed_lesion_mask(image[0].cpu())
        mask = detailed_mask.to(device)
        
        if mask.sum() < 10:  # If mask is too small/empty
            print("Lesion segmentation produced too small ROI, using entire image")
            mask = torch.ones((image.shape[2], image.shape[3]), device=device)
    else:
        # For general images, use bounding box detection
        _, roi_coords = roi_detector.detect_rois(image[0].cpu())
        
        if not roi_coords:  # If no ROIs detected, use the entire image
            print("No ROIs detected, using the entire image")
            mask = torch.ones((image.shape[2], image.shape[3]), device=device)
        else:
            print(f"Detected {len(roi_coords)} ROIs")
            mask = roi_detector.apply_rois_to_mask(image[0].shape, roi_coords)
    
    # Add batch and channel dimensions to mask
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)
    
    # Encryption
    print(f"Encrypting image with timestep {args.encryption_timestep}...")
    t = torch.tensor([args.encryption_timestep], device=device).long()
    encrypted_image, _ = diffusion_model.encrypt(image, key, mask, t)
    
    # Decryption
    print("Decrypting image...")
    decrypted_image, intermediates = diffusion_model.decrypt(encrypted_image, key, t, mask)
    
    # Calculate metrics
    metrics = calculate_metrics(image, encrypted_image, decrypted_image)
    print("\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    # Save results
    save_images_with_metrics(
        image[0], encrypted_image[0], decrypted_image[0], 
        args.output_dir, 
        metrics
    )
    
    # Save mask visualization
    mask_vis = mask[0].repeat(3, 1, 1)
    save_image(mask_vis, os.path.join(args.output_dir, 'mask.png'))
    
    # Save intermediate steps if requested
    if args.save_intermediates and intermediates:
        os.makedirs(os.path.join(args.output_dir, 'intermediates'), exist_ok=True)
        for i, img in enumerate(intermediates):
            save_image(
                (img[0].clamp(-1, 1) * 0.5 + 0.5), 
                os.path.join(args.output_dir, 'intermediates', f'step_{i}.png')
            )
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the image encryption-decryption model")
    
    # Input and output paths
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--key_path', type=str, required=True, help='Path to key image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output images')
    
    # Model parameters
    parser.add_argument('--encryption_timestep', type=int, default=500, help='Timestep to use for encryption')
    parser.add_argument('--yolo_confidence', type=float, default=0.25, help='Confidence threshold for YOLO ROI detection')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--save_intermediates', action='store_true', help='Save intermediate decryption steps')
    parser.add_argument('--use_skin_lesion_segmentation', action='store_true', 
                       help='Use skin lesion segmentation instead of YOLO detection')
    
    args = parser.parse_args()
    
    run_inference(args) 