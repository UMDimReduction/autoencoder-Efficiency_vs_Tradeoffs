#!/usr/bin/env python3
"""
================================================================================
Variational Auto-Encoder (VAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Kingma & Welling (2013) - "Auto-encoding variational bayes"

Run Command (standalone test):
    python models/vae.py

Key Equations (Eq. 5-9 from paper):
    θ* = argmax_θ Σ log p_θ(x^(i))  (Eq. 5)
    p_θ(x^(i)) = ∫ p_θ(x^(i)|z) p_θ(z) dz  (Eq. 6)
    KL: D_KL(q_φ(z|x) || p_θ(z|x))  (Eq. 7)
    L_VAE = -E[log p_θ(x|z)] + D_KL(q_φ(z|x) || p_θ(z))  (Eq. 8-9)
    
Paper Notes:
    - Maps input to distribution (Gaussian) rather than fixed vector
    - Uses SGVB (Stochastic Gradient Variational Bayes)
    - Latent dim: 2 (for 2D visualization)
    - Enables both reconstruction and generation
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .base import BaseAutoEncoder, reparameterize, kl_divergence


class VAE(nn.Module):
    """
    Variational Auto-Encoder
    
    Models the latent space as a probability distribution (Gaussian),
    enabling both reconstruction and generation capabilities.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        kl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        decoder_layers.extend([
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        Returns: mu (mean) and logvar (log variance)
        """
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
        z = μ + σ * ε, where ε ~ N(0, I)
        
        This allows backpropagation through the sampling process.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        VAE Loss (Eq. 8-9):
        L_VAE(θ, φ) = -E_{z~q_φ(z|x)} [log p_θ(x|z)] + D_KL(q_φ(z|x) || p_θ(z))
        
        Which simplifies to:
        -L_VAE = log p_θ(x) - D_KL(q_φ(z|x) || p_θ(z|x)) ≤ log p_θ(x)
        
        This is the Evidence Lower Bound (ELBO).
        
        Components:
        1. Reconstruction loss: -E[log p(x|z)] ≈ ||x - x̂||^2 (for Gaussian decoder)
        2. KL divergence: D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (binary cross-entropy or MSE)
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence: D_KL(q(z|x) || p(z))
        # For Gaussian q and standard normal prior p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss (negative ELBO)
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by sampling from prior and decoding.
        z ~ p(z) = N(0, I)
        x ~ p(x|z)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly scoring.
        Uses the mean of the latent distribution (deterministic encoding).
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean of latent distribution"""
        mu, _ = self.encode(x)
        return mu


class ConvVAE(nn.Module):
    """
    Convolutional VAE for better spatial feature learning.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 2,
        kl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.hidden_dims = hidden_dims
        
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
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
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
        
        decoder_layers.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1], in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid()
        ])
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_dims[-1], 4, 4)
        h = self.decoder_conv(h)
        return h[:, :, :28, :28]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            error = F.mse_loss(recon, x, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


if __name__ == "__main__":
    # Test VAE
    model = VAE(input_dim=784, latent_dim=2)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Mu shape: {outputs['mu'].shape}")
    print(f"Logvar shape: {outputs['logvar'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"KL Loss: {losses['kl_loss'].item():.4f}")
    
    # Test sampling
    samples = model.sample(4, x.device)
    print(f"Sample shape: {samples.shape}")
