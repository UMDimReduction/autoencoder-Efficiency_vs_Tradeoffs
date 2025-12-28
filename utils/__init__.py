"""
================================================================================
Utilities Package
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

This package contains utility modules:
- data_loader: Data loading for MNIST and Fashion-MNIST
- trainer: Training utilities and anomaly detection evaluation
- visualization: Plotting utilities for figures

Usage:
    from utils import get_anomaly_detection_loaders, AnomalyTrainer
================================================================================
"""

from .data_loader import (
    get_anomaly_detection_loaders,
    get_standard_loaders,
    get_conditional_data,
    load_mnist,
    load_fashion_mnist,
    AnomalyDetectionDataset,
    add_noise
)

__all__ = [
    'get_anomaly_detection_loaders',
    'get_standard_loaders',
    'get_conditional_data',
    'load_mnist',
    'load_fashion_mnist',
    'AnomalyDetectionDataset',
    'add_noise',
]
