#!/usr/bin/env python3
"""
================================================================================
β-VAE (Beta Variational Auto-Encoder)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Higgins et al. (2017) - "β-VAE: Learning basic visual concepts with 
           a constrained variational framework"

Run Command (standalone test):
    python models/beta_vae.py

Key Equations (Eq. 12-13 from paper):
    Optimization (Eq. 12):
        max_{φ,θ} E_{x~D} [E_{z~q_φ(z|x)} [log p_θ(x|z)]]
        subject to D_KL(q_φ(z|x) || p_θ(z)) < δ
    
    Loss (Eq. 13):
        L_BETA(φ, β) = -E[log p_θ(x|z)] + β * D_KL(q_φ(z|x) || p_θ(z))
    
Paper Notes:
    - β = 1: Original VAE
    - β > 1: Stronger latent constraint, promotes disentanglement
    - Beta values: 1.5-2.0 (paper range)
    - Encourages more effective latent encoding
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class BetaVAE(nn.Module):
    """
    β-VAE: Variational Auto-Encoder with adjustable KL divergence weight.
    
    The β parameter controls the trade-off between reconstruction accuracy
    and disentanglement. Higher β encourages learning disentangled representations.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        beta: float = 1.5,  # β values from 1.5 to 2 as shown in paper
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        
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
        
        # Latent space parameters
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
        """Encode to latent distribution parameters"""
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
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
        β-VAE Loss (Eq. 13):
        L_BETA(φ, β) = -E_{z~q_φ(z|x)} [log p_θ(x|z)] + β D_KL(q_φ(z|x) || p_θ(z))
        
        The β parameter controls the strength of the KL divergence constraint.
        - β = 1: Standard VAE
        - β > 1: Stronger disentanglement, may sacrifice reconstruction quality
        - β < 1: Better reconstruction, less disentangled representations
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(self.beta)
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score based on reconstruction error"""
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
    
    def traverse_latent(
        self,
        x: torch.Tensor,
        dim: int,
        range_: Tuple[float, float] = (-3, 3),
        steps: int = 10
    ) -> torch.Tensor:
        """
        Traverse a single latent dimension to visualize what it represents.
        Useful for analyzing disentanglement.
        """
        mu, _ = self.encode(x[:1])  # Use first sample
        
        traversals = []
        for val in torch.linspace(range_[0], range_[1], steps):
            z = mu.clone()
            z[0, dim] = val
            recon = self.decode(z)
            traversals.append(recon)
        
        return torch.cat(traversals, dim=0)


class AnnealedBetaVAE(BetaVAE):
    """
    β-VAE with KL annealing schedule.
    
    Gradually increases β during training to help the model
    learn useful representations before applying strong regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        beta_start: float = 0.0,
        beta_end: float = 2.0,
        annealing_steps: int = 10000,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, beta_start, **kwargs)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.annealing_steps = annealing_steps
        self.current_step = 0
    
    def update_beta(self):
        """Update β based on current training step"""
        if self.current_step < self.annealing_steps:
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * \
                       (self.current_step / self.annealing_steps)
        else:
            self.beta = self.beta_end
        self.current_step += 1
        return self.beta


class CapacityBetaVAE(BetaVAE):
    """
    β-VAE with controlled capacity increase.
    
    Instead of fixed β, gradually increases the capacity of the latent channel
    as described in "Understanding disentangling in β-VAE" (Burgess et al., 2018).
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        gamma: float = 100.0,  # Weight of capacity constraint
        capacity_max: float = 25.0,
        capacity_increment: float = 0.01,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.gamma = gamma
        self.capacity_max = capacity_max
        self.capacity_increment = capacity_increment
        self.current_capacity = 0.0
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Loss with controlled capacity:
        L = Recon_loss + γ * |KL - C|
        
        Where C is the target capacity that gradually increases.
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Capacity constraint
        capacity_loss = self.gamma * torch.abs(kl_loss - self.current_capacity)
        
        # Update capacity
        if self.training:
            self.current_capacity = min(
                self.capacity_max,
                self.current_capacity + self.capacity_increment
            )
        
        total_loss = recon_loss + capacity_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'capacity': torch.tensor(self.current_capacity)
        }


if __name__ == "__main__":
    # Test β-VAE
    model = BetaVAE(input_dim=784, latent_dim=2, beta=1.5)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"KL Loss: {losses['kl_loss'].item():.4f}")
    print(f"Beta: {losses['beta'].item():.2f}")
    
    # Test traversal
    traversals = model.traverse_latent(x, dim=0, steps=5)
    print(f"Traversal shape: {traversals.shape}")
