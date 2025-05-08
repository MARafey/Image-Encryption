#!/usr/bin/env python
"""
Script to download and prepare the HAM10000 skin cancer dataset for use with our
image encryption model.
"""

import os
import argparse
import zipfile
import pandas as pd
import requests
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

def download_from_kaggle(output_dir):
    """
    Download the HAM10000 dataset from Kaggle using the Kaggle API.
    
    Before running this, make sure to set up your Kaggle API credentials:
    1. Go to kaggle.com -> Your Account -> Create New API Token
    2. Save the kaggle.json file to ~/.kaggle/kaggle.json
    3. Run `chmod 600 ~/.kaggle/kaggle.json` to secure the file
    """
    print("Downloading HAM10000 dataset from Kaggle...")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    api.dataset_download_files(
        dataset="kmader/skin-cancer-mnist-ham10000",
        path=output_dir,
        unzip=True,
        quiet=False
    )
    
    print(f"Dataset downloaded to {output_dir}")
    return True

def download_from_alternative(output_dir):
    """
    Alternative method to download the HAM10000 dataset from a direct URL.
    """
    # URLs for the dataset files
    urls = {
        "HAM10000_metadata.csv": "https://storage.googleapis.com/kaggle-data-sets/122/83162/bundle/archive/HAM10000_metadata.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230101%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230101T000000Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=abcdefg",
        "HAM10000_images.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("WARNING: Direct download URLs may not be stable. Kaggle API is recommended.")
    print("Attempting to download HAM10000 dataset from alternative sources...")
    
    # Download metadata
    try:
        print("Please download the HAM10000 dataset manually from Kaggle:")
        print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("\nAfter downloading, extract the files to the output directory.")
        print(f"Expected output directory: {output_dir}")
        
        manual_download = input("Press Enter once you've manually downloaded and extracted the dataset, or type 'skip' to skip: ")
        if manual_download.lower() == 'skip':
            return False
            
        # Check if files exist
        if not os.path.exists(os.path.join(output_dir, 'HAM10000_metadata.csv')):
            print("Error: Metadata file not found. Please make sure you've extracted the dataset correctly.")
            return False
            
        images_dir = os.path.join(output_dir, 'HAM10000_images')
        if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 1000:
            print("Error: Image files not found or incomplete. Please make sure you've extracted the dataset correctly.")
            return False
            
        print("Dataset files found!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

def verify_dataset(output_dir):
    """
    Verify that the HAM10000 dataset has been downloaded and extracted correctly.
    """
    # Check for metadata file
    metadata_path = os.path.join(output_dir, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        print("Error: Metadata file not found.")
        return False
    
    # Load metadata to check structure
    try:
        metadata = pd.read_csv(metadata_path)
        required_columns = ['image_id', 'lesion_id', 'dx']
        for col in required_columns:
            if col not in metadata.columns:
                print(f"Error: Required column '{col}' not found in metadata.")
                return False
        
        num_images = len(metadata)
        print(f"Metadata loaded successfully with {num_images} entries.")
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return False
    
    # Check for image directory
    images_dir = os.path.join(output_dir, 'HAM10000_images')
    if not os.path.exists(images_dir):
        # Try alternative paths
        images_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(images_dir):
            print("Error: Image directory not found.")
            return False
    
    # Count image files
    image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {image_count} image files in {images_dir}")
    
    if image_count < 1000:  # HAM10000 should have around 10,000 images
        print("Warning: Fewer images found than expected. Dataset may be incomplete.")
    
    return True

def prepare_dataset(output_dir):
    """
    Prepare the HAM10000 dataset for use with our model.
    """
    metadata_path = os.path.join(output_dir, 'HAM10000_metadata.csv')
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Print dataset statistics
    print("\nHAM10000 Dataset Statistics:")
    print(f"Total images: {len(metadata)}")
    
    # Print diagnosis distribution
    dx_counts = metadata['dx'].value_counts()
    print("\nDiagnosis Distribution:")
    for dx, count in dx_counts.items():
        print(f"  {dx}: {count} images ({count/len(metadata)*100:.1f}%)")
    
    # Check image paths
    images_dir = os.path.join(output_dir, 'HAM10000_images')
    if not os.path.exists(images_dir):
        images_dir = os.path.join(output_dir, 'images')
    
    print("\nDataset is ready for use with the image encryption model!")
    print(f"To train the model on this dataset, use:")
    print(f"  python train.py --data_dir {output_dir} --dataset_type ham10000")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Download and prepare the HAM10000 skin cancer dataset')
    parser.add_argument('--output_dir', type=str, default='./ham10000_dataset',
                       help='Directory to save the dataset')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip the download step (use if already downloaded)')
    
    args = parser.parse_args()
    
    if not args.skip_download:
        try:
            # Try using Kaggle API first
            success = download_from_kaggle(args.output_dir)
        except Exception as e:
            print(f"Error using Kaggle API: {str(e)}")
            print("Falling back to alternative download method...")
            success = download_from_alternative(args.output_dir)
        
        if not success:
            print("Dataset download failed. Please try downloading manually from Kaggle.")
            return
    
    # Verify the dataset
    print("\nVerifying dataset integrity...")
    if not verify_dataset(args.output_dir):
        print("Dataset verification failed. Please check the error messages above.")
        return
    
    # Prepare the dataset
    print("\nPreparing dataset for use with the model...")
    prepare_dataset(args.output_dir)

if __name__ == '__main__':
    main() 