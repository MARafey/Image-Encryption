# Image Encryption and Decryption using Diffusion Models

This project implements a novel approach to image encryption and decryption using diffusion models. The system can selectively encrypt regions of interest (ROI) in images and then decrypt them using a key image.

## Project Overview

The core idea is to use a diffusion model as both an encryptor and decryptor for images:

1. **Encryption**: An image is encrypted by adding noise to it according to a diffusion process guided by a key image.
2. **ROI Detection**: YOLO model is used to detect regions of interest in the image that should be encrypted.
3. **Decryption**: The same key image is used with the diffusion model to denoise/decrypt the encrypted image. The decryption process also leverages information from the encrypted image itself to improve reconstruction quality.

This approach ensures that:

- Only the correct key image can properly decrypt the image
- Only specific regions of interest are encrypted, preserving the rest of the image
- The encryption is robust and generates visually distorted encrypted regions
- The decryption process is enhanced by utilizing both the key and encrypted image information

## Advanced ResNet-UNet Hybrid Architecture

The project features a state-of-the-art hybrid architecture that combines the strengths of ResNet and U-Net designs:

### Key Components

1. **ResNet Blocks**: Instead of standard convolutional blocks, we use residual blocks that preserve fine-grained information through skip connections, allowing better gradient flow during training and higher-quality reconstructions.

2. **Self-Attention Mechanisms**: Strategic integration of attention modules at specific locations in the architecture allows the model to focus on relevant image regions, improving the encryption-decryption quality.

3. **Enhanced Time Embedding**: Improved sinusoidal positional encoding for diffusion timesteps, inspired by transformer architectures, enhances the model's ability to understand the noise level during the diffusion process.

4. **Bidirectional Feature Flow**: Features from both the upsampling and initial encoding paths are combined at the end, creating richer representations for final reconstruction.

5. **Specialized Key and Encrypted Image Processing**: Dedicated paths for processing key images and encrypted content with residual connections for more effective feature extraction.

### Advantages over Standard UNet

- Improved stability during training with residual connections
- Better preservation of high-frequency details in decrypted images
- Enhanced ability to selectively encrypt/decrypt specific regions
- More robust against adversarial attacks attempting to decode without the correct key
- Higher structural similarity between original and decrypted images

## Installation

```bash
pip install -r requirements.txt
```

The project requires PyTorch, torchvision, ultralytics (for YOLO), and other dependencies listed in the requirements.txt file.

### YOLO Model Download

On first run, the system will automatically download the YOLOv8n model if not already present.

## Dataset Structure

The dataset should be organized in the following structure:

```
dataset/
  ├── Images/
  │     ├── image1.jpg
  │     ├── image2.jpg
  │     └── ...
  └── Key/
        ├── keyimage1.jpg
        ├── keyimage2.jpg
        └── ...
```

The key image for each image should be named with the prefix "key" followed by the image name. For example, the key for "image1.jpg" should be "keyimage1.jpg".

## Usage

### Training

To train the diffusion model for image encryption-decryption:

```bash
python train.py --data_dir ./dataset --epochs 100 --batch_size 8
```

Additional options:

```
--checkpoint_dir: Directory to save model checkpoints (default: ./checkpoints)
--samples_dir: Directory to save sample images during training (default: ./samples)
--log_dir: Directory for tensorboard logs (default: ./logs)
--resume: Path to checkpoint to resume training from
--learning_rate: Learning rate (default: 1e-4)
--sample_interval: Interval for generating and saving samples (default: 100)
--encryption_timestep: Timestep to use for encryption during sampling (default: 500)
--yolo_confidence: Confidence threshold for YOLO ROI detection (default: 0.25)
--no_cuda: Disable CUDA training
--use_gaussian_noise: Use Gaussian noise instead of key image for encryption
```

### Inference

To encrypt and decrypt an image using a trained model:

```bash
python inference.py --image_path path/to/image.jpg --key_path path/to/key.jpg --model_path path/to/model.pt
```

Additional options:

```
--output_dir: Directory to save output images (default: ./output)
--encryption_timestep: Timestep to use for encryption (default: 500)
--yolo_confidence: Confidence threshold for YOLO ROI detection (default: 0.25)
--no_cuda: Disable CUDA
--save_intermediates: Save intermediate decryption steps
```

## Monitoring Training

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir=./logs
```

This will display training losses, metrics (MSE, PSNR, SSIM, Entropy), and sample images during training.

## Metrics

The following metrics are calculated and logged during training and inference:

- **Reconstruction MSE**: Mean Squared Error between the original and decrypted images
- **Reconstruction PSNR**: Peak Signal-to-Noise Ratio between the original and decrypted images
- **Reconstruction SSIM**: Structural Similarity Index between the original and decrypted images
- **Encryption MSE**: Mean Squared Error between the original and encrypted images
- **Entropy**: Shannon entropy of the original, encrypted, and decrypted images
- **FID Score**: Fréchet Inception Distance between original and decrypted images (calculated during evaluation)

## Implementation Details

### Core Components

1. **ResNet-UNet Hybrid Model**: A state-of-the-art architecture that combines:

   - **ResNet Blocks**: For improved gradient flow and feature preservation
   - **Attention Mechanisms**: Strategically placed to focus on important regions
   - **Enhanced Time Embedding**: Using sinusoidal positional encoding for better diffusion control
   - **Bidirectional Feature Flow**: Combining features from different stages for better reconstruction

2. **ROI Detector**: Uses YOLOv8 to detect regions of interest in images
3. **Metrics**: Calculates quality metrics to evaluate the encryption and decryption performance
4. **Dataset Loader**: Handles loading image and key pairs from the dataset

### Encryption Process

The encryption process follows a forward diffusion process where noise is gradually added to the image according to a diffusion schedule, but only in the regions of interest. The process is conditioned on the key image to ensure that the same key is needed for decryption.

### Decryption Process

The decryption process reverses the diffusion process, gradually removing noise from the encrypted regions to recover the original image. This process requires the same key image used during encryption.

The enhanced decryption algorithm also utilizes information from the encrypted image itself. By incorporating features extracted from the encrypted image during the denoising process, the model can better preserve important details and achieve higher quality reconstruction. This dual-input approach (key + encrypted image) makes the decryption more robust and effective, especially for complex images with fine details.

## License

[MIT License](LICENSE)
