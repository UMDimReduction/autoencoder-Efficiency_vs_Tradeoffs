#!/usr/bin/env python3
"""
================================================================================
Configuration File
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon
Journal: Machine Learning with Applications

Description:
    Contains all hyperparameters, model configurations, and expected results
    from the paper.

    - DEVICE: GPU/CPU device selection
    - DATASET_CONFIG: MNIST and Fashion-MNIST specifications
    - TRAINING_CONFIG: Common training hyperparameters
    - MODEL_CONFIGS: Model-specific configurations
    - EXPECTED_RESULTS: Table 3 ROC-AUC values
    - TRAINING_TIMES: Table 2 training durations
================================================================================
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configurations
DATASET_CONFIG = {
    'mnist': {
        'input_dim': 784,  # 28 x 28
        'image_size': 28,
        'channels': 1,
        'num_classes': 10,
        'train_size': 60000,
        'test_size': 10000,
    },
    'fashion_mnist': {
        'input_dim': 784,
        'image_size': 28,
        'channels': 1,
        'num_classes': 10,
        'train_size': 60000,
        'test_size': 10000,
    }
}

# Common training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'latent_dim': 2,  # For 2D visualization as shown in paper
    'hidden_dims': [512, 256, 128],  # Encoder hidden layers
}

# Model-specific configurations based on paper descriptions
MODEL_CONFIGS = {
    'dae': {
        'name': 'Denoising Auto-Encoder',
        'noise_factor': 0.27,  # 27% noise as mentioned in paper (20%-52% range)
        'hidden_dims': [512, 256, 128],
        'latent_dim': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'sae': {
        'name': 'Sparse Auto-Encoder',
        'sparsity_weight': 1e-3,  # L1 regularization weight
        'sparsity_target': 0.45,  # 45% neurons activated as mentioned in paper
        'hidden_dims': [512, 256, 128],
        'latent_dim': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'cae': {
        'name': 'Contractive Auto-Encoder',
        'lambda_': 1e-4,  # Frobenius norm penalty weight
        'hidden_dims': [512, 256, 128],
        'latent_dim': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'vae': {
        'name': 'Variational Auto-Encoder',
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,  # 2D for visualization
        'learning_rate': 1e-3,
        'epochs': 50,
        'kl_weight': 1.0,
    },
    'beta_vae': {
        'name': 'Beta-VAE',
        'beta': 1.5,  # Beta values from 1.5 to 2 as shown in paper
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'advae': {
        'name': 'Self-Adversarial VAE',
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,
        'learning_rate': 1e-3,
        'epochs': 50,
        'kl_weight': 1.0,
        'discriminator_weight': 0.5,
    },
    'cvae': {
        'name': 'Conditional VAE',
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,
        'num_classes': 10,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'vqvae': {
        'name': 'Vector Quantized VAE',
        'num_embeddings': 512,  # Codebook size K
        'embedding_dim': 64,
        'commitment_cost': 0.25,  # Beta in VQ-VAE loss
        'hidden_dims': [128, 256],
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'iwae': {
        'name': 'Importance Weighted Auto-Encoder',
        'num_samples': 50,  # k=50 importance samples as mentioned in paper
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'pae': {
        'name': 'Probabilistic Auto-Encoder',
        'beta': 1.0,  # KL divergence weight
        'C': 0.5,  # Regularization constant
        'hidden_dims': [512, 256, 128],
        'latent_dim': 2,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
    'rda': {
        'name': 'Robust Deep Auto-Encoder',
        'lambda_': 1e-3,  # Sparsity regularization for S
        'hidden_dims': [512, 256, 128],
        'latent_dim': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
    },
}

# Expected ROC-AUC results from Table 3 in the paper
EXPECTED_RESULTS = {
    'mnist': {
        'cae': 0.22,
        'vae': 0.61,
        'vqvae': 0.82,
        'rda': 0.82,
        'cvae': 0.80,
        'sae': 0.83,
        'dae': 0.73,
        'beta_vae': 0.87,
        'pae': 0.89,
        'iwae': 0.87,
        'advae': 0.93,
    },
    'fashion_mnist': {
        'cae': 0.22,
        'vae': 0.56,
        'vqvae': 0.56,
        'rda': 0.60,
        'cvae': 0.66,
        'sae': 0.66,
        'dae': 0.56,
        'beta_vae': 0.59,
        'pae': 0.64,
        'iwae': 0.57,
        'advae': 0.87,
    }
}

# Training times from Table 2 (HH:MM:SS format)
TRAINING_TIMES = {
    'mnist': {
        'dae': '0:47:00',
        'sae': '1:07:21',
        'cae': '1:13:25',
        'vae': '0:35:50',
        'beta_vae': '0:23:33',
        'advae': '0:59:45',
        'cvae': '0:35:21',
        'iwae': '0:34:18',
        'pae': '0:20:17',
        'rda': '0:59:21',
        'vqvae': '0:30:33',
    },
    'fashion_mnist': {
        'dae': '0:56:45',
        'sae': '1:22:10',
        'cae': '1:45:27',
        'vae': '0:42:45',
        'beta_vae': '0:55:27',
        'advae': '1:01:45',
        'cvae': '0:59:40',
        'iwae': '0:37:16',
        'pae': '0:33:21',
        'rda': '1:02:10',
        'vqvae': '0:40:21',
    }
}
