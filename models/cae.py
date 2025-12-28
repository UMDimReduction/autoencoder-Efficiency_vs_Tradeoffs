#!/usr/bin/env python3
"""
================================================================================
Contractive Auto-Encoder (CAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Rifai et al. (2011) - "Contractive auto-encoders: Explicit invariance 
           during feature extraction"

Run Command (standalone test):
    python models/cae.py

Key Equations (Eq. 3-4 from paper):
    Frobenius norm: ||J_f(x)||²_F = Σ_{i,j} (∂h_j(x) / ∂x_i)²  (Eq. 3)
    
    Loss: J_CAE(θ) = Σ_{x∈D_n} (L(x, g(f(x))) + λ||J_f(x)||²_F)  (Eq. 4)
    
Paper Notes:
    - Penalty term based on Frobenius norm of Jacobian
    - Reduces sensitivity to minor input variations
    - Lambda (λ): 1e-4
    - Learns locally-invariant representations
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import BaseAutoEncoder


class CAE(BaseAutoEncoder):
    """
    Contractive Auto-Encoder
    
    Uses Jacobian Frobenius norm penalty to learn robust,
    locally-invariant representations.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        lambda_: float = 1e-4,  # Contractive penalty weight
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.lambda_ = lambda_
    
    def _build_encoder(self) -> nn.Module:
        """
        Build encoder where we need to track weights for Jacobian computation.
        We use a simpler architecture for efficient Jacobian computation.
        """
        # For CAE, we typically use a single hidden layer for tractable Jacobian
        # But we can approximate for multi-layer
        layers = []
        in_dim = self.input_dim
        
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Sigmoid())  # Sigmoid for analytical derivative
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.latent_dim))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def encode_with_jacobian(self, x: torch.Tensor) -> tuple:
        """
        Encode input and compute Jacobian of encoder w.r.t. input.
        
        For efficiency, we compute the Frobenius norm of Jacobian directly
        using the analytical form for sigmoid activations:
        ||J_f||^2_F = Σ_j h_j(1-h_j)^2 * Σ_i W_ij^2
        """
        x_flat = x.view(x.size(0), -1)
        x_flat.requires_grad_(True)
        
        # Forward pass through encoder
        h = x_flat
        for layer in self.encoder:
            h = layer(h)
        
        return h, x_flat
    
    def compute_jacobian_frobenius_approx(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Frobenius norm of Jacobian using backpropagation.
        
        ||J_f(x)||^2_F = Σ_j ||∂z_j/∂x||^2
        
        This is computed by summing squared gradients of each latent dimension.
        """
        batch_size = x.size(0)
        jacobian_norm = torch.zeros(batch_size, device=x.device)
        
        # For each latent dimension, compute gradient w.r.t input
        for j in range(z.size(1)):
            grad_outputs = torch.zeros_like(z)
            grad_outputs[:, j] = 1
            
            grad_j = torch.autograd.grad(
                outputs=z,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Sum of squared gradients
            jacobian_norm = jacobian_norm + (grad_j ** 2).sum(dim=1)
        
        return jacobian_norm
    
    def compute_jacobian_frobenius_analytical(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Analytical computation of Jacobian Frobenius norm for single-layer
        encoder with sigmoid activation.
        
        ||J_f||^2_F = Σ_j [h_j(1-h_j)]^2 * ||W_j||^2
        
        For multi-layer, this is an approximation using the final layer.
        """
        # Get the last linear layer weights before final activation
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                W = layer.weight
        
        # Sigmoid derivative: σ'(x) = σ(x)(1 - σ(x)) = h(1-h)
        sigmoid_derivative = h * (1 - h)  # [batch, latent_dim]
        
        # ||J||^2_F = Σ_j (h_j(1-h_j))^2 * Σ_i W_ij^2
        W_squared_sum = (W ** 2).sum(dim=1)  # Sum over input dim
        
        jacobian_norm = ((sigmoid_derivative ** 2) * W_squared_sum).sum(dim=1)
        
        return jacobian_norm
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with Jacobian computation support"""
        x_flat = x.view(x.size(0), -1)
        x_flat.requires_grad_(True)
        
        # Encode
        z = self.encoder(x_flat)
        
        # Decode
        x_recon = self.decoder(z)
        
        return {
            'recon': x_recon,
            'z': z,
            'x_flat': x_flat
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        CAE Loss Function (Eq. 4):
        J_CAE(θ) = Σ_{x∈D_n} (L(x, g(f(x))) + λ||J_f(x)||^2_F)
        
        Where:
        - L(x, g(f(x))) is the reconstruction loss
        - ||J_f(x)||^2_F is the Frobenius norm of the Jacobian
        - λ modulates the strength of the penalty
        """
        recon = outputs['recon']
        z = outputs['z']
        x_flat = outputs['x_flat']
        
        # Ensure we have original flattened input
        original_x_flat = x.view(x.size(0), -1)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, original_x_flat, reduction='mean')
        
        # Contractive penalty (Frobenius norm of Jacobian)
        # Use analytical form for efficiency
        contractive_loss = self.compute_jacobian_frobenius_analytical(x_flat, z).mean()
        
        # Total loss
        total_loss = recon_loss + self.lambda_ * contractive_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'contractive_loss': contractive_loss
        }


class TwoLayerCAE(nn.Module):
    """
    Two-layer deterministic CAE as described in the paper
    for anomaly data reconstruction experiments (Section 5.2.9).
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        lambda_: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        z = self.encode(x)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'z': z,
            'x_flat': x_flat
        }
    
    def compute_jacobian_penalty(
        self,
        x_flat: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contractive penalty using the first layer weights
        and sigmoid derivative.
        """
        # Get first encoder layer weights
        W1 = self.encoder[0].weight  # [hidden_dim, input_dim]
        
        # Activation after first layer
        h1 = torch.sigmoid(F.linear(x_flat, W1, self.encoder[0].bias))
        
        # Sigmoid derivative
        h1_deriv = h1 * (1 - h1)
        
        # Jacobian Frobenius norm
        # ||J||^2_F ≈ Σ_j (h1_j')^2 * Σ_i W1_ij^2
        W1_squared_sum = (W1 ** 2).sum(dim=1)  # [hidden_dim]
        
        jacobian_norm = ((h1_deriv ** 2) * W1_squared_sum).sum(dim=1)  # [batch]
        
        return jacobian_norm.mean()
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        z = outputs['z']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # Contractive penalty
        contractive_loss = self.compute_jacobian_penalty(x_flat, z)
        
        total_loss = recon_loss + self.lambda_ * contractive_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'contractive_loss': contractive_loss
        }
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score"""
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            outputs = self.forward(x)
            error = F.mse_loss(outputs['recon'], x_flat, reduction='none').mean(dim=1)
        return error


if __name__ == "__main__":
    # Test CAE
    model = CAE(input_dim=784, latent_dim=32, lambda_=1e-4)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"Contractive Loss: {losses['contractive_loss'].item():.4f}")
