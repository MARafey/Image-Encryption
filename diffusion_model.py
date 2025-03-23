import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels + out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = nn.Linear(1, dim)
        self.layer2 = nn.Linear(dim, dim)
        
    def forward(self, t):
        t = t.unsqueeze(-1).float()
        t = self.layer1(t)
        t = F.silu(t)
        return self.layer2(t)

class UNet(nn.Module):
    def __init__(self, in_channels=3, key_channels=3, base_channels=64, time_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Initial convolution to incorporate key
        self.key_embed = nn.Sequential(
            nn.Conv2d(key_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Input convolution
        self.inc = ConvBlock(in_channels, base_channels)
        
        # Downsampling path
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck1 = ConvBlock(base_channels * 8, base_channels * 16)
        self.bottleneck2 = ConvBlock(base_channels * 16, base_channels * 8)
        
        # Time embedding projection
        self.time_mlp1 = nn.Linear(time_dim, base_channels * 16)
        
        # Upsampling path
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
    def forward(self, x, key, t):
        # Embed time step
        t_emb = self.time_embedding(t)
        
        # Process key
        key_features = self.key_embed(key)
        
        # Initial convolution
        x = self.inc(x)
        x = x + key_features  # Add key features
        
        # Downsample
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # Bottleneck with time conditioning
        x = self.bottleneck1(x3)
        time_features = self.time_mlp1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + time_features
        x = self.bottleneck2(x)
        
        
        # Upsample with skip connections
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Output
        return self.outc(x)

class DiffusionModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = UNet().to(device)
        
        # Define diffusion hyperparameters
        self.num_timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Create beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t, key, t, t_index):
        """Sample from the reverse process"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.model(x_t, key, t)
        
        # No noise when t == 0
        noise = torch.randn_like(x_t) if t_index > 0 else torch.zeros_like(x_t)
        variance = torch.zeros_like(betas_t) if t_index == 0 else betas_t
        
        # Compute the mean
        mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        return mean + torch.sqrt(variance) * noise
    
    def encrypt(self, image, key, mask=None, t=None):
        """
        Encrypt an image using the key
        Args:
            image: Image tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            mask: Optional mask tensor (B, 1, H, W) to only encrypt specific regions
            t: Optional specific timestep to use for encryption
        """
        image = image.to(self.device)
        key = key.to(self.device)
        batch_size = image.shape[0]
        
        # Use a specific timestep or randomly chosen ones for each batch item
        if t is None:
            # Use random timesteps for each image in batch
            t = torch.randint(0, self.num_timesteps // 2, (batch_size,), device=self.device)
        elif isinstance(t, int):
            t = torch.ones(batch_size, device=self.device).long() * t
            
        # Generate noise
        noise = torch.randn_like(image)
        
        # Apply forward diffusion to create noisy image
        noisy_image = self.q_sample(image, t, noise)
        
        if mask is not None:
            mask = mask.to(self.device)
            # Only apply noise to masked regions
            encrypted_image = image * (1 - mask) + noisy_image * mask
        else:
            encrypted_image = noisy_image
            
        return encrypted_image, t
    
    def decrypt(self, encrypted_image, key, t, mask=None, steps=None):
        """
        Decrypt an image using the key
        Args:
            encrypted_image: Encrypted image tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            t: Timestep used for encryption (B,)
            mask: Optional mask tensor (B, 1, H, W) to only decrypt specific regions
            steps: Number of denoising steps (default: timestep value)
        """
        encrypted_image = encrypted_image.to(self.device)
        key = key.to(self.device)
        batch_size = encrypted_image.shape[0]
        
        # Default steps to t values
        if steps is None:
            steps = t.clone()
        
        # Initialize x_t with the encrypted image
        x_t = encrypted_image.clone()
        
        # For tracking intermediate steps
        intermediate_images = []
        
        # Get max steps to determine loop iterations
        max_steps = steps.max().item()
        
        # Run denoising for specified steps
        for i in tqdm(reversed(range(max_steps)), desc="Decrypting"):
            # Create a time tensor with the current timestep
            time_tensor = torch.ones(batch_size, device=self.device).long() * i
            
            # Only update pixels for images where i < steps
            mask_update = (i < steps).float().view(-1, 1, 1, 1)
            
            # Denoise step
            with torch.no_grad():
                denoised_image = self.p_sample(x_t, key, time_tensor, i)
            
            # Update only relevant images
            x_t = x_t * (1 - mask_update) + denoised_image * mask_update
            
            if i % (max_steps // 10) == 0 or i == max_steps - 1:
                intermediate_images.append(x_t.clone())
        
        # If we have a mask, only replace the encrypted regions
        if mask is not None:
            mask = mask.to(self.device)
            decrypted_image = encrypted_image * (1 - mask) + x_t * mask
        else:
            decrypted_image = x_t
            
        return decrypted_image, intermediate_images
    
    def train_step(self, image, key, optimizer, mask=None):
        """
        Train the diffusion model for one step
        Args:
            image: Image tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            optimizer: PyTorch optimizer
            mask: Optional mask tensor to only train on specific regions
        """
        image = image.to(self.device)
        key = key.to(self.device)
        batch_size = image.shape[0]
        
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # Generate random noise
        noise = torch.randn_like(image)
        
        # Forward diffusion to get noisy image
        noisy_image = self.q_sample(image, t, noise)
        
        # Predict noise
        noise_pred = self.model(noisy_image, key, t)
        
        # Calculate loss
        if mask is not None:
            mask = mask.to(self.device)
            # Only compute loss on masked regions
            loss = F.mse_loss(noise_pred * mask, noise * mask) / (mask.mean() + 1e-8)
        else:
            loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model from checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 