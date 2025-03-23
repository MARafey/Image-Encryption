import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

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

def get_dataloader(root_dir, batch_size=8, shuffle=True, transform=None):
    dataset = EncryptionDataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader 