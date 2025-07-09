import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import random

def create_transforms(use_augmentation=False):
    """
    Create training transforms with optional data augmentation
    Args:
        use_augmentation: Whether to apply data augmentation
    Returns:
        torchvision.transforms.Compose object
    """
    if use_augmentation:
        # Data augmentation for training
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # Standard transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return transform

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
                print(f"Warning: Image {img_path} not found")
        
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
        
        # Process metadata for model conditioning
        age = img_meta.get('age', 0.0)
        if pd.isna(age):
            age = 0.0
        # Normalize age to 0-1 range (assuming max age around 100)
        normalized_age = min(age / 100.0, 1.0)
        
        sex = img_meta.get('sex', 'unknown')
        if pd.isna(sex):
            sex = 'unknown'
            
        localization = img_meta.get('localization', 'unknown')
        if pd.isna(localization):
            localization = 'unknown'
            
        dx_type_method = img_meta.get('dx_type', 'unknown')
        if pd.isna(dx_type_method):
            dx_type_method = 'unknown'
        
        return {
            'image': image, 
            'key': key, 
            'image_path': img_path, 
            'dx_type': dx_type,
            'lesion_id': img_meta['lesion_id'],
            'dx_id': img_meta.get('dx_type', -1),  # Some versions have different column names
            # Medical metadata for conditioning
            'metadata': {
                'dx': dx_type,
                'dx_type': dx_type_method,
                'age': normalized_age,
                'sex': sex,
                'localization': localization
            }
        }

def get_dataloader(root_dir, batch_size=8, shuffle=True, transform=None, use_augmentation=False):
    if transform is None:
        transform = create_transforms(use_augmentation)
    dataset = EncryptionDataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def collate_metadata(batch):
    """
    Collate function to properly handle metadata in batches
    """
    # Extract individual components
    images = torch.stack([item['image'] for item in batch])
    keys = torch.stack([item['key'] for item in batch])
    
    # Extract metadata into separate lists
    metadata = {
        'dx': [item['metadata']['dx'] for item in batch],
        'dx_type': [item['metadata']['dx_type'] for item in batch],
        'age': torch.tensor([item['metadata']['age'] for item in batch], dtype=torch.float32),
        'sex': [item['metadata']['sex'] for item in batch],
        'localization': [item['metadata']['localization'] for item in batch]
    }
    
    # Other fields
    image_paths = [item['image_path'] for item in batch]
    dx_types = [item['dx_type'] for item in batch]
    lesion_ids = [item['lesion_id'] for item in batch]
    
    return {
        'image': images,
        'key': keys,
        'image_path': image_paths,
        'dx_type': dx_types,
        'lesion_id': lesion_ids,
        'metadata': metadata
    }

def get_ham10000_dataloaders(data_dir, batch_size=8, transform=None, use_same_keys=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, use_metadata_collate=True, use_augmentation=False):
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
        use_metadata_collate: Whether to use custom collate function for metadata
        use_augmentation: Whether to apply data augmentation to training data
    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create transforms if not provided
    if transform is None:
        # Use augmentation for training, no augmentation for validation/test
        train_transform = create_transforms(use_augmentation)
        val_transform = create_transforms(use_augmentation=False)
    else:
        train_transform = transform
        val_transform = transform
    
    # Create separate datasets for each split with appropriate transforms
    train_dataset_full = HAM10000Dataset(data_dir, transform=train_transform, use_same_keys=use_same_keys)
    val_dataset_full = HAM10000Dataset(data_dir, transform=val_transform, use_same_keys=use_same_keys)
    test_dataset_full = HAM10000Dataset(data_dir, transform=val_transform, use_same_keys=use_same_keys)
    
    # Calculate split sizes
    total_size = len(train_dataset_full)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Get indices for splitting
    indices = list(range(total_size))
    random.Random(42).shuffle(indices)  # For reproducible splits
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    # Choose collate function
    collate_fn = collate_metadata if use_metadata_collate else None
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
    if use_augmentation:
        print("Data augmentation enabled for training data")
    
    return train_loader, val_loader, test_loader 