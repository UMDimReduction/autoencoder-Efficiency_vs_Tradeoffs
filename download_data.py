#!/usr/bin/env python3
"""
================================================================================
Download Data Script
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Run Command:
    python download_data.py [--data_dir DATA_DIR] [--dataset {mnist,fashion_mnist,all}]

Examples:
    python download_data.py                          # Download all datasets to ./data
    python download_data.py --dataset mnist          # Download only MNIST
    python download_data.py --dataset fashion_mnist  # Download only Fashion-MNIST
    python download_data.py --data_dir /path/to/dir  # Custom data directory

Description:
    Downloads MNIST and Fashion-MNIST datasets used in the paper.
    Both datasets contain:
    - 60,000 training images (28x28 grayscale)
    - 10,000 test images (28x28 grayscale)
    - 10 classes each
================================================================================
"""

import os
import argparse
from torchvision import datasets, transforms


def download_mnist(data_dir: str = './data') -> None:
    """
    Download MNIST dataset.
    
    MNIST consists of handwritten digit images (0-9).
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Image size: 28x28 grayscale
    """
    print("=" * 60)
    print("Downloading MNIST dataset...")
    print("=" * 60)
    
    transform = transforms.ToTensor()
    
    # Download training set
    print("\n[1/2] Downloading training set...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    print(f"      Training samples: {len(train_dataset)}")
    
    # Download test set
    print("\n[2/2] Downloading test set...")
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    print(f"      Test samples: {len(test_dataset)}")
    
    print("\n✓ MNIST download complete!")
    print(f"  Location: {os.path.join(data_dir, 'MNIST')}")


def download_fashion_mnist(data_dir: str = './data') -> None:
    """
    Download Fashion-MNIST dataset.
    
    Fashion-MNIST consists of fashion product images:
    - 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
    - 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Image size: 28x28 grayscale
    """
    print("=" * 60)
    print("Downloading Fashion-MNIST dataset...")
    print("=" * 60)
    
    transform = transforms.ToTensor()
    
    # Download training set
    print("\n[1/2] Downloading training set...")
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    print(f"      Training samples: {len(train_dataset)}")
    
    # Download test set
    print("\n[2/2] Downloading test set...")
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    print(f"      Test samples: {len(test_dataset)}")
    
    print("\n✓ Fashion-MNIST download complete!")
    print(f"  Location: {os.path.join(data_dir, 'FashionMNIST')}")


def verify_datasets(data_dir: str = './data') -> None:
    """Verify that datasets are properly downloaded."""
    print("\n" + "=" * 60)
    print("Verifying datasets...")
    print("=" * 60)
    
    # Check MNIST
    mnist_path = os.path.join(data_dir, 'MNIST')
    if os.path.exists(mnist_path):
        print(f"\n✓ MNIST: Found at {mnist_path}")
    else:
        print(f"\n✗ MNIST: Not found at {mnist_path}")
    
    # Check Fashion-MNIST
    fmnist_path = os.path.join(data_dir, 'FashionMNIST')
    if os.path.exists(fmnist_path):
        print(f"✓ Fashion-MNIST: Found at {fmnist_path}")
    else:
        print(f"✗ Fashion-MNIST: Not found at {fmnist_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for Auto-Encoder Anomaly Detection Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py                          # Download all datasets
    python download_data.py --dataset mnist          # Download only MNIST
    python download_data.py --dataset fashion_mnist  # Download only Fashion-MNIST
    python download_data.py --data_dir ./datasets    # Custom directory
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory to store datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'fashion_mnist', 'all'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("AUTO-ENCODER ANOMALY DETECTION - DATA DOWNLOAD")
    print("=" * 60)
    print(f"\nData directory: {os.path.abspath(args.data_dir)}")
    print(f"Dataset(s): {args.dataset}")
    
    # Download requested datasets
    if args.dataset in ['mnist', 'all']:
        print("\n")
        download_mnist(args.data_dir)
    
    if args.dataset in ['fashion_mnist', 'all']:
        print("\n")
        download_fashion_mnist(args.data_dir)
    
    # Verify downloads
    verify_datasets(args.data_dir)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nYou can now run experiments with:")
    print("  python main.py --dataset mnist")
    print("  python main.py --dataset fashion_mnist")


if __name__ == "__main__":
    main()
