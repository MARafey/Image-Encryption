#!/usr/bin/env python
"""
Script to download and prepare the HAM10000 skin cancer dataset for use with our
image encryption model.
"""

import os
import argparse
import zipfile
import pandas as pd
import shutil
import kagglehub

def download_from_kagglehub(output_dir):
    """
    Download the HAM10000 dataset from Kaggle using kagglehub.
    This is a simpler method than using the Kaggle API directly.
    """
    print("Downloading HAM10000 dataset from Kaggle using kagglehub...")
    
    try:
        # Download latest version of the dataset
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"Dataset downloaded to: {path}")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if the download was successful by looking for key files
        if os.path.exists(path):
            # Copy or move files to the user's specified output directory if different
            if path != output_dir and not os.path.samefile(path, output_dir):
                print(f"Copying files to user-specified directory: {output_dir}")
                
                # Copy files to output directory
                for item in os.listdir(path):
                    src_path = os.path.join(path, item)
                    dst_path = os.path.join(output_dir, item)
                    
                    if os.path.isdir(src_path):
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
            
            print(f"Dataset downloaded successfully to {output_dir}")
            return True
        else:
            print("Failed to download dataset")
            return False
    except Exception as e:
        print(f"Error downloading dataset with kagglehub: {str(e)}")
        print("Falling back to alternative download method...")
        return False

def download_from_alternative(output_dir):
    """
    Alternative method to download the HAM10000 dataset from a direct URL.
    """
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
        # Try using kagglehub first
        success = download_from_kagglehub(args.output_dir)
        
        if not success:
            print("Falling back to manual download method...")
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