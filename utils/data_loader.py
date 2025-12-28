#!/usr/bin/env python3
"""
================================================================================
Data Loading Utilities
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Run Command (standalone test):
    python utils/data_loader.py

Description:
    Data loading utilities for MNIST and Fashion-MNIST datasets.
    Implements the anomaly detection setup from the paper where:
    - Training: Only samples from the "normal" class
    - Testing: All samples with binary labels (normal=0, anomaly=1)

Datasets:
    - MNIST: 60,000 train + 10,000 test, 28x28 grayscale, 10 digit classes
    - Fashion-MNIST: 60,000 train + 10,000 test, 28x28 grayscale, 10 fashion classes
================================================================================
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, Dict


class AnomalyDetectionDataset(Dataset):
    """
    Custom dataset for anomaly detection experiments.
    Following the paper's methodology:
    - Normal class: One digit class (e.g., digit 0)
    - Anomaly class: All other digit classes
    """
    def __init__(self, base_dataset, normal_class: int = 0, train: bool = True):
        self.base_dataset = base_dataset
        self.normal_class = normal_class
        self.train = train
        
        # Get all indices
        targets = np.array(base_dataset.targets)
        
        if train:
            # For training, only use normal class samples
            self.indices = np.where(targets == normal_class)[0]
            self.labels = np.zeros(len(self.indices))
        else:
            # For testing, use all samples
            # Label: 0 for normal, 1 for anomaly
            self.indices = np.arange(len(targets))
            self.labels = (targets != normal_class).astype(np.float32)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base_dataset[real_idx]
        label = self.labels[idx]
        return image, label


def get_transforms():
    """Get standard transforms for the datasets"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def load_mnist(data_dir: str = './data') -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST dataset"""
    transform = get_transforms()
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def load_fashion_mnist(data_dir: str = './data') -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Load Fashion-MNIST dataset"""
    transform = get_transforms()
    
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def get_anomaly_detection_loaders(
    dataset_name: str = 'mnist',
    normal_class: int = 0,
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders for anomaly detection experiments.
    
    Following the paper's methodology:
    - Training: Only normal class samples
    - Testing: All samples (normal + anomalous)
    
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        normal_class: Which class to treat as normal (0-9)
        batch_size: Batch size for training
        data_dir: Directory to store/load data
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader
    """
    if dataset_name == 'mnist':
        train_dataset, test_dataset = load_mnist(data_dir)
    elif dataset_name == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create anomaly detection datasets
    train_ad_dataset = AnomalyDetectionDataset(train_dataset, normal_class, train=True)
    test_ad_dataset = AnomalyDetectionDataset(test_dataset, normal_class, train=False)
    
    train_loader = DataLoader(
        train_ad_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ad_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_standard_loaders(
    dataset_name: str = 'mnist',
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get standard data loaders (all classes for training and testing).
    Used for general reconstruction and latent space visualization.
    """
    if dataset_name == 'mnist':
        train_dataset, test_dataset = load_mnist(data_dir)
    elif dataset_name == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def add_noise(images: torch.Tensor, noise_factor: float = 0.27) -> torch.Tensor:
    """
    Add Gaussian noise to images for DAE training.
    Paper mentions noise levels from 20% to 52%, with 27% used in experiments.
    """
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)


def get_conditional_data(
    dataset_name: str = 'mnist',
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders that include class labels for CVAE training.
    """
    if dataset_name == 'mnist':
        train_dataset, test_dataset = load_mnist(data_dir)
    elif dataset_name == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_all_data_loaders(
    dataset_name: str = 'mnist',
    normal_class: int = 0,
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4
) -> Dict:
    """
    Get all data loaders needed for experiments.
    
    Returns:
        Dictionary with 'train', 'test', 'train_standard', 'test_standard' loaders
    """
    train_ad, test_ad = get_anomaly_detection_loaders(
        dataset_name, normal_class, batch_size, data_dir, num_workers
    )
    train_std, test_std = get_standard_loaders(
        dataset_name, batch_size, data_dir, num_workers
    )
    
    return {
        'train': train_ad,
        'test': test_ad,
        'train_standard': train_std,
        'test_standard': test_std
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing MNIST loading...")
    train_loader, test_loader = get_anomaly_detection_loaders('mnist', normal_class=0)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Check a batch
    images, labels = next(iter(test_loader))
    print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    print("\nTesting Fashion-MNIST loading...")
    train_loader, test_loader = get_anomaly_detection_loaders('fashion_mnist', normal_class=0)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
