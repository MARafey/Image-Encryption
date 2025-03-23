import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytorch_fid.fid_score as fid

def reconstruction_loss(original, reconstructed, reduction='mean'):
    """
    Calculate reconstruction loss (MSE, PSNR, SSIM)
    Args:
        original: Original image tensor (B, C, H, W)
        reconstructed: Reconstructed image tensor (B, C, H, W)
        reduction: How to reduce batch dimension ('mean', 'none')
    Returns:
        Dictionary with various reconstruction metrics
    """
    # Convert to numpy for some metrics
    if isinstance(original, torch.Tensor):
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()
    else:
        original_np = original
        reconstructed_np = reconstructed
    
    # MSE Loss
    mse = F.mse_loss(original, reconstructed, reduction=reduction)
    
    # Initialize results dictionary
    results = {'mse': mse}
    
    # Calculate batch-wise metrics
    batch_size = original.shape[0]
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # Convert from (C, H, W) to (H, W, C) and scale to [0, 1]
        orig_img = np.transpose(original_np[i], (1, 2, 0))
        recon_img = np.transpose(reconstructed_np[i], (1, 2, 0))
        
        # Ensure value range [0, 1]
        orig_img = (orig_img + 1) / 2  # Assuming [-1, 1] range
        recon_img = (recon_img + 1) / 2
        
        # Calculate PSNR (higher is better)
        psnr_val = psnr(orig_img, recon_img, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Calculate SSIM (higher is better)
        ssim_val = ssim(orig_img, recon_img, data_range=1.0, channel_axis=2, multichannel=True)
        ssim_values.append(ssim_val)
    
    # Add to results
    results['psnr'] = np.mean(psnr_values) if reduction == 'mean' else np.array(psnr_values)
    results['ssim'] = np.mean(ssim_values) if reduction == 'mean' else np.array(ssim_values)
    
    return results

def entropy_loss(images, reduction='mean'):
    """
    Calculate entropy loss - higher entropy means more randomness which is desirable for encrypted images
    Args:
        images: Image tensor (B, C, H, W) in range [-1, 1]
        reduction: How to reduce batch dimension ('mean', 'none')
    Returns:
        Entropy loss (lower is better for natural images, higher is better for encryption)
    """
    batch_size, channels, height, width = images.shape
    
    # Convert to [0, 255] for histogram calculation
    images_255 = ((images.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
    
    entropy_values = []
    
    # Calculate entropy for each image in batch
    for i in range(batch_size):
        img = images_255[i].detach().cpu().numpy()
        channel_entropy = []
        
        # Calculate entropy for each channel
        for c in range(channels):
            # Create histogram with 256 bins (one for each possible value)
            hist, _ = np.histogram(img[c], bins=256, range=(0, 255), density=True)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            channel_entropy.append(entropy)
        
        # Average entropy across channels
        entropy_values.append(np.mean(channel_entropy))
    
    entropy_tensor = torch.tensor(entropy_values, device=images.device)
    
    if reduction == 'mean':
        return entropy_tensor.mean()
    return entropy_tensor

def calculate_fid(real_paths, generated_paths):
    """
    Calculate FID score between real and generated images
    Args:
        real_paths: List of paths to real images or directory
        generated_paths: List of paths to generated images or directory
    Returns:
        FID score (lower is better)
    """
    # Use PyTorch FID implementation to calculate score
    score = fid.calculate_fid_given_paths([real_paths, generated_paths], 
                                         batch_size=50, 
                                         device='cuda' if torch.cuda.is_available() else 'cpu',
                                         dims=2048)
    return score

def calculate_metrics(original_images, encrypted_images, decrypted_images):
    """
    Calculate all metrics for the encryption-decryption pipeline
    Args:
        original_images: Original images (B, C, H, W)
        encrypted_images: Encrypted images (B, C, H, W)
        decrypted_images: Decrypted images (B, C, H, W)
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Original vs Decrypted (Reconstruction quality)
    recon_metrics = reconstruction_loss(original_images, decrypted_images)
    metrics['reconstruction_mse'] = recon_metrics['mse'].item() if isinstance(recon_metrics['mse'], torch.Tensor) else recon_metrics['mse']
    metrics['reconstruction_psnr'] = recon_metrics['psnr']
    metrics['reconstruction_ssim'] = recon_metrics['ssim']
    
    # Original vs Encrypted (Should be different)
    encrypt_metrics = reconstruction_loss(original_images, encrypted_images)
    metrics['encryption_mse'] = encrypt_metrics['mse'].item() if isinstance(encrypt_metrics['mse'], torch.Tensor) else encrypt_metrics['mse']
    metrics['encryption_psnr'] = encrypt_metrics['psnr']
    metrics['encryption_ssim'] = encrypt_metrics['ssim']
    
    # Entropy of encrypted images (higher is better for encryption)
    metrics['encrypted_entropy'] = entropy_loss(encrypted_images).item()
    metrics['original_entropy'] = entropy_loss(original_images).item()
    metrics['decrypted_entropy'] = entropy_loss(decrypted_images).item()
    
    return metrics 