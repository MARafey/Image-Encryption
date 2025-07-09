import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ResNet-style Block for the hybrid architecture
class ResNetBlock(nn.Module):
    def __init__(self, channels, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Time embedding projection if provided
        self.time_mlp = nn.Linear(time_dim, channels) if time_dim is not None else None
        
        self.activation = nn.SiLU()
        
    def forward(self, x, t_emb=None):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Add time embedding if provided
        if t_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + time_emb
            
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Skip connection
        x = x + residual
        
        return self.activation(x)

# Attention Block for enhanced feature learning
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        # Size threshold for applying attention
        self.max_size_threshold = 64 * 64  # Max spatial dimensions for safe attention

    def forward(self, x):
        size = x.shape[-2:]
        batch_size = x.shape[0]
        num_pixels = size[0] * size[1]
        
        # Skip attention for large feature maps to prevent OOM errors
        if num_pixels > self.max_size_threshold:
            # Print warning only once
            if not hasattr(self, '_size_warned'):
                print(f"Warning: Skipping attention for large tensor of shape {x.shape} to prevent OOM")
                self._size_warned = True
            return x  # Identity function
            
        try:
            # Standard attention processing for smaller tensors
            x_flat = x.flatten(2).transpose(1, 2)
            x_ln = self.ln(x_flat)
            attention_value, _ = self.mha(x_ln, x_ln, x_ln)
            attention_value = attention_value + x_flat
            attention_value = self.ff_self(attention_value) + attention_value
            return attention_value.transpose(1, 2).view(batch_size, self.channels, *size)
        except RuntimeError as e:
            # Fallback when runtime error occurs (typically OOM)
            if not hasattr(self, '_error_warned'):
                print(f"Warning: Error in attention: {e}. Falling back to identity.")
                self._error_warned = True
            return x

# DownSample block that combines ResNet and UNet architectures
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=None, include_attention=False):
        super().__init__()
        self.res_block1 = ResNetBlock(in_channels, time_dim)
        self.res_block2 = ResNetBlock(in_channels, time_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.attention = SelfAttentionBlock(out_channels) if include_attention else nn.Identity()
        
    def forward(self, x, t_emb=None):
        x = self.res_block1(x, t_emb)
        skip = self.res_block2(x, t_emb)
        x = self.conv(skip)
        x = self.attention(x)
        return x, skip

# UpSample block that combines ResNet and UNet architectures
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=None, include_attention=False):
        super().__init__()
        self.res_block1 = ResNetBlock(in_channels + out_channels, time_dim)
        self.res_block2 = ResNetBlock(in_channels + out_channels, time_dim)
        self.upsample = nn.ConvTranspose2d(in_channels + out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.attention = SelfAttentionBlock(out_channels) if include_attention else nn.Identity()
        
    def forward(self, x, skip, t_emb=None):
        # Check if shapes match except in channel dimension
        if x.shape[0] != skip.shape[0] or x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            # Print warning but only once
            if not hasattr(self, '_warned'):
                self._warned = True
                print(f"WARNING: Shape mismatch in UpBlock. Input: {x.shape}, Skip: {skip.shape}. Resizing skip connection.")
            
            # Resize skip connection if spatial dimensions don't match
            skip = F.interpolate(skip, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.res_block1(x, t_emb)
        x = self.res_block2(x, t_emb)
        x = self.upsample(x)
        x = self.attention(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Improved time embedding with sinusoidal positional encoding
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)
        
        # MLP layers
        self.projection1 = nn.Linear(dim, dim * 2)
        self.projection2 = nn.Linear(dim * 2, dim)
        
    def forward(self, t):
        t = t.unsqueeze(-1).float()
        
        # Create sinusoidal embedding
        emb = t * self.emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Project to higher dimension and back
        emb = self.projection1(emb)
        emb = F.silu(emb)
        emb = self.projection2(emb)
        
        return emb

# Metadata Embedding for medical information conditioning
class MetadataEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        
        # Diagnosis embedding (7 main types)
        self.dx_embedding = nn.Embedding(10, dim // 4)  # Extra space for unknown types
        
        # Diagnosis type embedding (4 main types)
        self.dx_type_embedding = nn.Embedding(5, dim // 8)
        
        # Age embedding (normalized to 0-1 range)
        self.age_projection = nn.Linear(1, dim // 8)
        
        # Sex embedding (male=0, female=1, unknown=2)
        self.sex_embedding = nn.Embedding(3, dim // 8)
        
        # Body location embedding (12+ main locations)
        self.location_embedding = nn.Embedding(15, dim // 4)
        
        # Final projection to combine all embeddings
        self.final_projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # Diagnosis type mapping
        self.dx_mapping = {
            'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 'bkl': 4, 'df': 5, 'vasc': 6
        }
        
        # Diagnosis method mapping
        self.dx_type_mapping = {
            'histo': 0, 'follow_up': 1, 'consensus': 2, 'confocal': 3
        }
        
        # Sex mapping
        self.sex_mapping = {'male': 0, 'female': 1}
        
        # Body location mapping
        self.location_mapping = {
            'face': 0, 'scalp': 1, 'ear': 2, 'back': 3, 'trunk': 4, 'chest': 5,
            'upper extremity': 6, 'lower extremity': 7, 'abdomen': 8, 'neck': 9,
            'genital': 10, 'foot': 11, 'hand': 12, 'acral': 13, 'unknown': 14
        }
    
    def forward(self, metadata_dict):
        """
        metadata_dict should contain:
        - dx: list of diagnosis strings
        - dx_type: list of diagnosis type strings  
        - age: tensor of ages (normalized 0-1)
        - sex: list of sex strings
        - localization: list of body location strings
        """
        batch_size = len(metadata_dict['dx'])
        device = next(self.parameters()).device
        
        # Convert diagnosis to indices
        dx_indices = torch.tensor([
            self.dx_mapping.get(dx.lower(), 7) for dx in metadata_dict['dx']
        ], device=device)
        dx_emb = self.dx_embedding(dx_indices)
        
        # Convert diagnosis type to indices
        dx_type_indices = torch.tensor([
            self.dx_type_mapping.get(dx_type.lower(), 4) for dx_type in metadata_dict['dx_type']
        ], device=device)
        dx_type_emb = self.dx_type_embedding(dx_type_indices)
        
        # Age embedding (already normalized)
        age_emb = self.age_projection(metadata_dict['age'].unsqueeze(1).to(device))
        
        # Sex embedding
        sex_indices = torch.tensor([
            self.sex_mapping.get(sex.lower(), 2) for sex in metadata_dict['sex']
        ], device=device)
        sex_emb = self.sex_embedding(sex_indices)
        
        # Location embedding
        loc_indices = torch.tensor([
            self.location_mapping.get(loc.lower(), 14) for loc in metadata_dict['localization']
        ], device=device)
        loc_emb = self.location_embedding(loc_indices)
        
        # Concatenate all embeddings
        combined = torch.cat([dx_emb, dx_type_emb, age_emb, sex_emb, loc_emb], dim=1)
        
        # Final projection
        metadata_embedding = self.final_projection(combined)
        
        return metadata_embedding

# Improved UNet with ResNet blocks and attention mechanisms
class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, key_channels=3, base_channels=64, time_dim=256, use_metadata=True, dropout_rate=0.1):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.use_metadata = use_metadata
        self.dropout_rate = dropout_rate
        
        # Metadata embedding for conditioning
        if use_metadata:
            self.metadata_embedding = MetadataEmbedding(time_dim)
            # Fusion layer for time and metadata
            self.time_metadata_fusion = nn.Sequential(
                nn.Linear(time_dim * 2, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim)
            )
        
        # Key image processing path
        self.key_embed = nn.Sequential(
            nn.Conv2d(key_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ResNetBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Encrypted image processing path (for decryption)
        self.encrypted_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ResNetBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Initial convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ResNetBlock(base_channels)
        )
        
        # Downsampling path
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim, include_attention=True)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)
        
        # Bottleneck
        self.bottleneck1 = ResNetBlock(base_channels * 8, time_dim)
        self.bottleneck_attn = SelfAttentionBlock(base_channels * 8)
        self.bottleneck_dropout = nn.Dropout2d(dropout_rate)
        self.bottleneck2 = ResNetBlock(base_channels * 8, time_dim)
        
        # Upsampling path - Fix the channel dimensions to match skip connections
        # The first parameter should match the bottleneck output channels
        # The second parameter should match the corresponding skip connection channels
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, time_dim)
        # Only use attention for smaller feature maps to avoid OOM
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_dim, include_attention=False)
        self.up3 = UpBlock(base_channels * 2, base_channels, time_dim)
        
        # Final processing
        self.final_res = ResNetBlock(base_channels * 2, time_dim)
        self.final_dropout = nn.Dropout2d(dropout_rate)
        
        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
        )
        
    def forward(self, x, key, t, encrypted_img=None, metadata=None):
        # Embed time step
        t_emb = self.time_embedding(t)
        
        # Add metadata conditioning if available
        if self.use_metadata and metadata is not None:
            metadata_emb = self.metadata_embedding(metadata)
            # Fuse time and metadata embeddings
            combined_emb = torch.cat([t_emb, metadata_emb], dim=1)
            t_emb = self.time_metadata_fusion(combined_emb)
        
        # Process key
        key_features = self.key_embed(key)
        
        # Initial convolution
        x_features = self.inc(x)
        
        # Combine with key features
        x = x_features + key_features
        
        # If encrypted image is provided (during decryption), use it
        if encrypted_img is not None:
            encrypted_features = self.encrypted_embed(encrypted_img)
            x = x + encrypted_features
        
        # Downsample
        x1, skip1 = self.down1(x, t_emb)
        x2, skip2 = self.down2(x1, t_emb)
        x3, skip3 = self.down3(x2, t_emb)
        
        # Bottleneck with attention
        x = self.bottleneck1(x3, t_emb)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_dropout(x)
        x = self.bottleneck2(x, t_emb)
        
        # Upsample with skip connections
        x = self.up1(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up3(x, skip1, t_emb)
        
        # Concatenate with initial features for better gradient flow
        x = torch.cat([x, x_features], dim=1)
        
        # Final processing
        x = self.final_res(x, t_emb)
        x = self.final_dropout(x)
        
        # Output
        return self.outc(x)

class DiffusionModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", use_metadata=True, dropout_rate=0.1):
        self.device = device
        self.use_metadata = use_metadata
        self.model = ResNetUNet(use_metadata=use_metadata, dropout_rate=dropout_rate).to(device)
        
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
    
    def p_sample(self, x_t, key, t, t_index, orig_encrypted=None, metadata=None):
        """Sample from the reverse process"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # Predict noise using both current state and original encrypted image
        predicted_noise = self.model(x_t, key, t, orig_encrypted, metadata)
        
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
    
    def decrypt(self, encrypted_image, key, t, mask=None, steps=None, metadata=None):
        """
        Decrypt an image using the key and the original encrypted image
        Args:
            encrypted_image: Encrypted image tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            t: Timestep used for encryption (B,)
            mask: Optional mask tensor (B, 1, H, W) to only decrypt specific regions
            steps: Number of denoising steps (default: timestep value)
            metadata: Optional metadata dict with medical information for conditioning
        """
        encrypted_image = encrypted_image.to(self.device)
        key = key.to(self.device)
        batch_size = encrypted_image.shape[0]
        
        # Default steps to t values
        if steps is None:
            steps = t.clone()
        
        # Initialize x_t with the encrypted image
        x_t = encrypted_image.clone()
        
        # Store original encrypted image for reference during decryption
        orig_encrypted = encrypted_image.clone()
        
        # For tracking intermediate steps
        intermediate_images = []
        
        # Get max steps to determine loop iterations
        max_steps = steps.max().item()
        
        # Reverse diffusion process (denoising)
        for i in tqdm(reversed(range(0, max_steps + 1)), desc="Decrypting", total=max_steps + 1):
            # Get indices where current timestep is active
            active_indices = (steps >= i).nonzero().squeeze(-1)
            
            if active_indices.shape[0] == 0:
                continue
                
            # Get active batch items and their timesteps
            active_t = torch.ones(active_indices.shape[0], device=self.device).long() * i
            active_x_t = x_t[active_indices]
            active_key = key[active_indices]
            active_orig_encrypted = orig_encrypted[active_indices]
            
            # Sample from p(x_{t-1} | x_t, x_0)
            # Extract metadata for active indices if provided
            active_metadata = None
            if metadata is not None:
                active_metadata = {}
                for key_meta in metadata:
                    if isinstance(metadata[key_meta], list):
                        active_metadata[key_meta] = [metadata[key_meta][idx] for idx in active_indices]
                    else:  # tensor
                        active_metadata[key_meta] = metadata[key_meta][active_indices]
            
            pred_x_0 = self.p_sample(active_x_t, active_key, active_t, i, active_orig_encrypted, active_metadata)
            
            # Update only active indices
            x_t[active_indices] = pred_x_0
            
            # Apply mask if provided
            if mask is not None:
                active_mask = mask[active_indices]
                # Only apply denoising to masked regions, keep original in unmasked regions
                x_t[active_indices] = encrypted_image[active_indices] * (1 - active_mask) + pred_x_0 * active_mask
            
            # Save intermediate for visualization (every 100 steps or last step)
            if i % 100 == 0 or i == max_steps:
                intermediate_images.append(x_t.detach().clone())
                
        return x_t, intermediate_images

    def train_step(self, image, key, optimizer, mask=None, metadata=None):
        """
        Train the model for one step
        Args:
            image: Image tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            optimizer: Optimizer to use
            mask: Optional mask tensor (B, 1, H, W) for selective encryption
            metadata: Optional metadata dict with medical information for conditioning
        """
        image = image.to(self.device)
        key = key.to(self.device)
        batch_size = image.shape[0]
        
        # Memory optimization for CPU training
        if self.device == "cpu" and batch_size > 4:
            # Process in smaller batches to save memory
            max_sub_batch = 4  # Maximum sub-batch size for CPU
            loss_sum = 0
            
            for i in range(0, batch_size, max_sub_batch):
                end_idx = min(i + max_sub_batch, batch_size)
                sub_batch_size = end_idx - i
                
                # Get sub-batch
                image_sub = image[i:end_idx]
                key_sub = key[i:end_idx]
                mask_sub = mask[i:end_idx] if mask is not None else None
                
                # Choose random timesteps for this sub-batch
                t_sub = torch.randint(0, self.num_timesteps, (sub_batch_size,), device=self.device)
                
                # Add noise to images
                noise_sub = torch.randn_like(image_sub)
                noisy_images_sub = self.q_sample(image_sub, t_sub, noise_sub)
                
                # Apply mask if provided
                if mask_sub is not None:
                    # Only add noise to masked regions for training
                    target_images_sub = image_sub * (1 - mask_sub) + noisy_images_sub * mask_sub
                    # For training, we want the model to predict what was added (full noise for masked regions, zero for others)
                    target_noise_sub = noise_sub * mask_sub
                else:
                    target_images_sub = noisy_images_sub
                    target_noise_sub = noise_sub
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Extract metadata for sub-batch if provided
                sub_metadata = None
                if metadata is not None:
                    sub_metadata = {}
                    for key_meta in metadata:
                        if isinstance(metadata[key_meta], list):
                            sub_metadata[key_meta] = metadata[key_meta][i:end_idx]
                        else:  # tensor
                            sub_metadata[key_meta] = metadata[key_meta][i:end_idx]
                
                predicted_noise_sub = self.model(noisy_images_sub, key_sub, t_sub, None, sub_metadata)
                
                # Calculate loss (mean squared error)
                loss_sub = F.mse_loss(predicted_noise_sub, target_noise_sub)
                
                # Backward pass and optimizer step
                loss_sub.backward()
                optimizer.step()
                
                loss_sum += loss_sub.item() * sub_batch_size
            
            # Average loss across all sub-batches
            return loss_sum / batch_size
        else:
            # Standard processing for GPU or small batches
            # Choose random timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            
            # Add noise to images
            noise = torch.randn_like(image)
            noisy_images = self.q_sample(image, t, noise)
            
            # Apply mask if provided
            if mask is not None:
                mask = mask.to(self.device)
                # Only add noise to masked regions for training
                # The model should learn to denoise properly in those regions while leaving others untouched
                target_images = image * (1 - mask) + noisy_images * mask
                # For training, we want the model to predict what was added (full noise for masked regions, zero for others)
                target_noise = noise * mask
            else:
                target_images = noisy_images
                target_noise = noise
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_noise = self.model(noisy_images, key, t, None, metadata)
            
            # Calculate loss (mean squared error)
            loss = F.mse_loss(predicted_noise, target_noise)
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            
            return loss.item()
    
    def save_model(self, path):
        """Save model to path"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load model from path"""
        self.model.load_state_dict(torch.load(path, map_location=self.device)) 