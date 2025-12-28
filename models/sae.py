#!/usr/bin/env python3
"""
================================================================================
Sparse Auto-Encoder (SAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Makhzani & Frey (2013) - "K-sparse autoencoders"

Run Command (standalone test):
    python models/sae.py

Key Equation (Eq. 2 from paper):
    X̃ = H_{W,b}(X) ≈ X
    
    Loss = Reconstruction Loss + λ * Sparsity Penalty
    
Paper Notes:
    - SAE learns sparse representation with single hidden layer
    - Sparsity target: 45% neurons activated
    - L1 regularization weight: 1e-3
    - Encourages compact, efficient representations
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import BaseAutoEncoder


class SAE(BaseAutoEncoder):
    """
    Sparse Auto-Encoder
    
    Uses L1 regularization (sparsity penalty) on hidden activations
    to encourage sparse representations.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        sparsity_weight: float = 1e-3,  # λ for L1 penalty
        sparsity_target: float = 0.45,  # Target sparsity level (45% as in paper)
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        
        # Store intermediate activations for sparsity calculation
        self.hidden_activations = []
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder with activation capture"""
        layers = []
        in_dim = self.input_dim
        
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def encode_with_activations(self, x: torch.Tensor) -> tuple:
        """
        Encode and capture hidden layer activations for sparsity penalty.
        """
        x = x.view(x.size(0), -1)
        activations = []
        
        idx = 0
        for layer in self.encoder:
            x = layer(x)
            # Capture activations after ReLU
            if isinstance(layer, nn.ReLU):
                activations.append(x)
            idx += 1
        
        return x, activations
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass capturing hidden activations"""
        x_flat = x.view(x.size(0), -1)
        
        # Encode with activation capture
        z, activations = self.encode_with_activations(x)
        
        # Decode
        x_recon = self.decoder(z)
        
        return {
            'recon': x_recon,
            'z': z,
            'activations': activations,
            'x_flat': x_flat
        }
    
    def kl_divergence_sparsity(self, rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between target sparsity and actual sparsity.
        KL(ρ || ρ̂) = ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))
        """
        rho_hat = torch.clamp(rho_hat, 1e-8, 1 - 1e-8)
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        SAE Loss Function:
        L = L_reconstruction + λ * L_sparsity
        
        Where L_sparsity is L1 norm or KL divergence to encourage sparse activations.
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        activations = outputs['activations']
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # Sparsity loss (L1 regularization on activations)
        sparsity_loss = torch.tensor(0.0, device=x.device)
        for act in activations:
            # Average activation per unit
            avg_activation = act.abs().mean()
            sparsity_loss = sparsity_loss + avg_activation
        
        sparsity_loss = self.sparsity_weight * sparsity_loss / len(activations)
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def get_sparsity_level(self, activations: List[torch.Tensor], threshold: float = 0.01) -> float:
        """
        Calculate actual sparsity level (percentage of near-zero activations).
        """
        total_neurons = 0
        sparse_neurons = 0
        
        for act in activations:
            total_neurons += act.numel()
            sparse_neurons += (act.abs() < threshold).sum().item()
        
        return sparse_neurons / total_neurons if total_neurons > 0 else 0.0


class KSparseSAE(BaseAutoEncoder):
    """
    K-Sparse Auto-Encoder
    
    Implements hard sparsity by keeping only top-k activations.
    More aggressive sparsity than L1 regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        k: int = 50,  # Number of active neurons to keep
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.k = k
    
    def k_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Keep only top-k activations, set rest to zero.
        """
        if x.size(1) <= self.k:
            return x
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x.abs(), self.k, dim=1)
        
        # Create sparse output
        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(1, topk_indices, x.gather(1, topk_indices))
        
        return sparse_x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with k-sparse constraint"""
        x = x.view(x.size(0), -1)
        
        # Pass through encoder layers with k-sparse after each ReLU
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Apply k-sparse after ReLU (except last layer)
            if isinstance(layer, nn.ReLU) and i < len(self.encoder) - 1:
                if self.training:
                    x = self.k_sparse(x)
        
        return x
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """K-Sparse AE loss - just reconstruction (sparsity enforced structurally)"""
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss
        }


class WinnerTakeAllSAE(nn.Module):
    """
    Winner-Take-All Sparse Auto-Encoder
    
    Alternative sparsity mechanism where only the maximum activation
    in each spatial/feature region is kept.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 1000,
        latent_dim: int = 32,
        lifetime_sparsity: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lifetime_sparsity = lifetime_sparsity
        
        # Single hidden layer for simpler sparsity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Running average of activations for lifetime sparsity
        self.register_buffer('running_avg', torch.zeros(hidden_dim))
        self.register_buffer('count', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        
        return {
            'recon': recon,
            'z': z
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss
        }
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            outputs = self.forward(x)
            error = F.mse_loss(outputs['recon'], x_flat, reduction='none').mean(dim=1)
        return error


if __name__ == "__main__":
    # Test SAE
    model = SAE(input_dim=784, latent_dim=32, sparsity_weight=1e-3)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Number of activation layers: {len(outputs['activations'])}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"Sparsity Loss: {losses['sparsity_loss'].item():.4f}")
