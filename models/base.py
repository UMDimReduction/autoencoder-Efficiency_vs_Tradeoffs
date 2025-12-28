#!/usr/bin/env python3
"""
================================================================================
Base Auto-Encoder Class and Common Components
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Description:
    Base class and utility functions shared across all auto-encoder architectures.
    
    Classes:
    - BaseAutoEncoder: Abstract base class for all models
    - ConvEncoder: Convolutional encoder for image data
    - ConvDecoder: Convolutional decoder for image data
    
    Functions:
    - reparameterize: VAE reparameterization trick
    - kl_divergence: KL divergence computation
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class BaseAutoEncoder(nn.Module, ABC):
    """
    Abstract base class for all auto-encoder architectures.
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build the encoder network"""
        layers = []
        in_dim = self.input_dim
        
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build the decoder network"""
        layers = []
        in_dim = self.latent_dim
        
        for h_dim in reversed(self.hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.input_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] for image data
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        x = x.view(x.size(0), -1)  # Flatten
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return {'recon': x_recon, 'z': z}
    
    @abstractmethod
    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss - must be implemented by subclasses"""
        pass
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly scoring.
        Higher error indicates anomaly.
        """
        x_flat = x.view(x.size(0), -1)
        outputs = self.forward(x)
        recon = outputs['recon']
        
        # MSE per sample
        error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        return self.encode(x)


class ConvEncoder(nn.Module):
    """Convolutional encoder for image data"""
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 32
    ):
        super().__init__()
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        self.conv_layers = nn.Sequential(*modules)
        
        # Calculate flattened size
        # For 28x28 input with stride 2: 28 -> 14 -> 7 -> 4 (for 3 layers)
        self.flatten_size = hidden_dims[-1] * 4 * 4
        
        self.fc = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    """Convolutional decoder for image data"""
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 64, 32],
        out_channels: int = 1
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid()
            )
        )
        
        self.conv_layers = nn.Sequential(*modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.hidden_dims[0], 4, 4)
        x = self.conv_layers(x)
        # Crop to 28x28 if necessary
        return x[:, :, :28, :28]


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE.
    z = mu + std * epsilon, where epsilon ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(z|x) and p(z) = N(0, I)
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
