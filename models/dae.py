#!/usr/bin/env python3
"""
================================================================================
Denoising Auto-Encoder (DAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Vincent et al. (2008) - "Extracting and composing robust features 
           with denoising autoencoders"

Run Command (standalone test):
    python models/dae.py

Key Equation (Eq. 1 from paper):
    L_DAE(θ, φ) = (1/n) * Σ(x^(i) - f_θ(g_φ(x̃^(i))))²
    
    where:
    - x̃ is the corrupted input (noisy version of x)
    - g_φ is the encoder
    - f_θ is the decoder
    
Paper Notes:
    - Corruption process M_D adds Gaussian noise
    - Noise factor: 27% (experiments tested 20%-52% range)
    - Forces network to learn robust features
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import BaseAutoEncoder


class DAE(BaseAutoEncoder):
    """
    Denoising Auto-Encoder
    
    The DAE learns to reconstruct clean inputs from corrupted versions,
    enhancing robustness and generalization capabilities.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        noise_factor: float = 0.27,  # 27% noise as in paper experiments
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.noise_factor = noise_factor
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to input (corruption process M_D)
        x̃^(i) ~ M_D(x̃^(i) | x^(i))
        """
        noise = self.noise_factor * torch.randn_like(x)
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional noise injection.
        
        Args:
            x: Input images
            add_noise: Whether to add noise (True during training)
        """
        x_flat = x.view(x.size(0), -1)
        
        # Add noise during training
        if add_noise and self.training:
            x_noisy = self.add_noise(x_flat)
        else:
            x_noisy = x_flat
        
        # Encode and decode
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        
        return {
            'recon': x_recon,
            'z': z,
            'x_noisy': x_noisy,
            'x_clean': x_flat
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        DAE Loss Function (Eq. 1):
        L_DAE(θ, φ) = (1/n) * Σ(x^(i) - f_θ(g_φ(x̃^(i))))^2
        
        The loss measures discrepancy between original clean input x
        and reconstructed output f_θ(g_φ(x̃)).
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        
        # MSE reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss
        }
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly scoring.
        For DAE, we use clean input (no noise) during inference.
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            z = self.encoder(x_flat)
            recon = self.decoder(z)
            error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error


class ConvDAE(nn.Module):
    """
    Convolutional Denoising Auto-Encoder
    Better for capturing spatial patterns in images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 32,
        noise_factor: float = 0.27,
        **kwargs
    ):
        super().__init__()
        self.noise_factor = noise_factor
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(prev_dim, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Flatten size: 28 -> 14 -> 7 -> 4
        self.flatten_size = hidden_dims[-1] * 4 * 4
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_dims[i], reversed_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.ReLU()
            ])
        
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1], in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid()
        ])
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        self.hidden_dims = hidden_dims
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = self.noise_factor * torch.randn_like(x)
        return torch.clamp(x + noise, 0., 1.)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_dims[-1], 4, 4)
        h = self.decoder_conv(h)
        return h[:, :, :28, :28]
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> Dict[str, torch.Tensor]:
        if add_noise and self.training:
            x_input = self.add_noise(x)
        else:
            x_input = x
        
        z = self.encode(x_input)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'z': z,
            'x_noisy': x_input,
            'x_clean': x
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """DAE Loss - reconstruct clean from noisy"""
        recon = outputs['recon']
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss
        }
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score based on reconstruction error"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, add_noise=False)
            recon = outputs['recon']
            # Per-sample MSE
            error = F.mse_loss(recon, x, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
        return error


if __name__ == "__main__":
    # Test DAE
    model = DAE(input_dim=784, latent_dim=32)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Loss: {losses['loss'].item():.4f}")
