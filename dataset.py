import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import random

class EncryptionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'Images')
        self.keys_dir = os.path.join(root_dir, 'Key')
        self.image_files = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        key_name = "key" + img_name
        key_path = os.path.join(self.keys_dir, key_name)
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        key = Image.open(key_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            key = self.transform(key)
        
        return {'image': image, 'key': key, 'image_path': img_path, 'key_path': key_path}

class HAM10000Dataset(Dataset):
    def __init__(self, data_dir, metadata_file=None, transform=None, use_same_keys=False):
        """
        HAM10000 skin cancer dataset loader
        Args:
            data_dir: Directory containing the HAM10000 dataset
            metadata_file: Path to metadata CSV file (default is looking for HAM10000_metadata.csv)
            transform: Image transform to apply
            use_same_keys: If True, use a fixed set of keys for encryption instead of using other images as keys
        """
        self.data_dir = data_dir
        self.use_same_keys = use_same_keys
        
        # Default image directory in HAM10000
        self.images_dir = os.path.join(data_dir, 'HAM10000_images')
        if not os.path.exists(self.images_dir):
            # Try alternative directory structure
            self.images_dir = os.path.join(data_dir, 'images')
        
        # Load metadata
        if metadata_file is None:
            metadata_file = os.path.join(data_dir, 'HAM10000_metadata.csv')
            if not os.path.exists(metadata_file):
                # Try alternative file name
                metadata_file = os.path.join(data_dir, 'metadata.csv')
        
        self.metadata = pd.read_csv(metadata_file)
        
        # Get image file paths
        self.image_paths = []
        for idx, row in self.metadata.iterrows():
            img_id = row['image_id']
            img_path = os.path.join(self.images_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                # Try alternative extension
                img_path = os.path.join(self.images_dir, img_id + '.png')
            
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
            else:
                print(f"Warning: Image {img_id} not found")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # If we're using fixed keys, prepare them
        if use_same_keys:
            # Create 10 random noise patterns as keys (one per class)
            self.fixed_keys = {}
            for dx_type in self.metadata['dx'].unique():
                # Create random noise
                random_key = torch.randn(3, 256, 256)
                # Normalize to [-1, 1] range
                random_key = (random_key - random_key.min()) / (random_key.max() - random_key.min()) * 2 - 1
                self.fixed_keys[dx_type] = random_key
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Get metadata for this image
        img_meta = self.metadata[self.metadata['image_id'] == img_id].iloc[0]
        dx_type = img_meta['dx']  # Diagnosis type
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Generate or select a key
        if self.use_same_keys:
            # Use the fixed key for this diagnosis type
            key = self.fixed_keys[dx_type]
        else:
            # Use another random image as key
            key_idx = random.randint(0, len(self.image_paths) - 1)
            if key_idx == idx:  # Avoid using the same image as its own key
                key_idx = (key_idx + 1) % len(self.image_paths)
            
            key_path = self.image_paths[key_idx]
            key = Image.open(key_path).convert('RGB')
            if self.transform:
                key = self.transform(key)
        
        return {
            'image': image, 
            'key': key, 
            'image_path': img_path, 
            'dx_type': dx_type,
            'lesion_id': img_meta['lesion_id'],
            'dx_id': img_meta.get('dx_type', -1)  # Some versions have different column names
        }

def get_dataloader(root_dir, batch_size=8, shuffle=True, transform=None):
    dataset = EncryptionDataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def get_ham10000_dataloaders(data_dir, batch_size=8, transform=None, use_same_keys=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train, validation, and test dataloaders for HAM10000 dataset
    Args:
        data_dir: Directory containing the HAM10000 dataset
        batch_size: Batch size for dataloaders
        transform: Image transform to apply
        use_same_keys: Whether to use fixed keys for encryption
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create the dataset
    dataset = HAM10000Dataset(data_dir, transform=transform, use_same_keys=use_same_keys)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
    
    return train_loader, val_loader, test_loader 