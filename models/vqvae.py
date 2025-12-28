#!/usr/bin/env python3
"""
================================================================================
Vector Quantized Variational Auto-Encoder (VQ-VAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Marimont & Tarroni (2021) - "Anomaly detection through latent space 
           restoration using vector quantized variational autoencoders"

Run Command (standalone test):
    python models/vqvae.py

Key Equations (Eq. 20-21 from paper):
    Quantization (Eq. 20):
        z_q(x) = Quantize(E(x)) = e_k
        where k = argmin_i ||E(x) - e_i||₂
    
    Loss (Eq. 21):
        L = ||x - D(e_k)||² + ||sg[E(x)] - e_k||² + β||E(x) - sg[e_k]||²
        
        Components:
        1. Reconstruction: ||x - D(e_k)||²
        2. VQ Loss: ||sg[E(x)] - e_k||² (codebook)
        3. Commitment: β||E(x) - sg[e_k]||² (encoder)
        
Paper Notes:
    - Discrete latent variables via vector quantization
    - Codebook size K: 512 embeddings
    - Embedding dim: 64
    - Commitment cost β: 0.25
    - sg[] = stop-gradient operation
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.
    
    Maps continuous encoder outputs to discrete codebook entries
    using nearest neighbor lookup.
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,  # K - codebook size
        embedding_dim: int = 64,     # D - embedding dimension
        commitment_cost: float = 0.25  # β - commitment loss weight
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: K x D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize codebook uniformly
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize encoder output to nearest codebook entry.
        
        Args:
            z: Encoder output [B, D] or [B, H, W, D]
        
        Returns:
            quantized: Quantized output (same shape as z)
            vq_loss: Vector quantization loss
            indices: Codebook indices
        """
        # Flatten if spatial
        input_shape = z.shape
        flat_z = z.view(-1, self.embedding_dim)  # [B*H*W, D]
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True) +
            self.embedding.weight.pow(2).sum(dim=1) -
            2 * flat_z @ self.embedding.weight.t()
        )  # [B*H*W, K]
        
        # Find nearest codebook entry (Eq. 20)
        indices = distances.argmin(dim=1)  # [B*H*W]
        
        # Get quantized values
        quantized = self.embedding(indices)  # [B*H*W, D]
        quantized = quantized.view(input_shape)
        
        # Compute losses (Eq. 21 components)
        # VQ loss: ||sg[E(x)] - e_k||^2 (codebook learning)
        e_latent_loss = F.mse_loss(quantized, z.detach())
        
        # Commitment loss: ||E(x) - sg[e_k]||^2 (encoder commitment)
        q_latent_loss = F.mse_loss(z, quantized.detach())
        
        # Combined VQ loss
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss, indices.view(input_shape[:-1])
    
    def get_codebook_usage(self, indices: torch.Tensor) -> Dict[str, float]:
        """Compute codebook usage statistics"""
        unique_indices = indices.unique()
        usage_rate = len(unique_indices) / self.num_embeddings
        return {
            'usage_rate': usage_rate,
            'unique_codes': len(unique_indices)
        }


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Auto-Encoder
    
    Uses discrete latent representations through vector quantization,
    enabling sharp reconstructions and discrete latent codes.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [128, 256],
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        encoder_layers.append(nn.Linear(in_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Build decoder
        decoder_layers = []
        in_dim = embedding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        decoder_layers.extend([
            nn.Linear(in_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to continuous latent space (before quantization)"""
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)
    
    def quantize(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output"""
        return self.vq(z)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent"""
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VQ-VAE"""
        # Encode
        z = self.encode(x)
        
        # Quantize
        z_q, vq_loss, indices = self.quantize(z)
        
        # Decode
        recon = self.decode(z_q)
        
        return {
            'recon': recon,
            'z': z,
            'z_q': z_q,
            'vq_loss': vq_loss,
            'indices': indices
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        VQ-VAE Loss (Eq. 21):
        L = ||x - D(e_k)||^2 + ||sg[E(x)] - e_k||^2 + β||E(x) - sg[e_k]||^2
        
        The vq_loss from VectorQuantizer includes the last two terms.
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        vq_loss = outputs['vq_loss']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by sampling from codebook.
        """
        # Sample random codebook indices
        indices = torch.randint(
            0, self.vq.num_embeddings,
            (num_samples,), device=device
        )
        
        # Get corresponding embeddings
        z_q = self.vq.embedding(indices)
        
        # Decode
        samples = self.decode(z_q)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score based on reconstruction error"""
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            outputs = self.forward(x)
            recon = outputs['recon']
            error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get quantized latent representation"""
        z = self.encode(x)
        z_q, _, _ = self.quantize(z)
        return z_q
    
    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get codebook indices for input"""
        z = self.encode(x)
        _, _, indices = self.quantize(z)
        return indices


class ConvVQVAE(nn.Module):
    """
    Convolutional VQ-VAE for better spatial feature learning.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64],
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        **kwargs
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(prev_dim, h_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            prev_dim = h_dim
        
        # Project to embedding dimension
        encoder_layers.append(
            nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        )
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        decoder_layers = [
            nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1)
        ]
        
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_dims[i], reversed_dims[i + 1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU()
            ])
        
        decoder_layers.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1], in_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        z = self.encoder(x)  # [B, D, H, W]
        
        # Reshape for quantization: [B, H, W, D]
        z_perm = z.permute(0, 2, 3, 1)
        
        # Quantize
        z_q_perm, vq_loss, indices = self.vq(z_perm)
        
        # Reshape back: [B, D, H, W]
        z_q = z_q_perm.permute(0, 3, 1, 2)
        
        # Decode
        recon = self.decoder(z_q)
        recon = recon[:, :, :28, :28]  # Ensure correct output size
        
        return {
            'recon': recon,
            'z': z,
            'z_q': z_q,
            'vq_loss': vq_loss,
            'indices': indices
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        recon = outputs['recon']
        vq_loss = outputs['vq_loss']
        
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        total_loss = recon_loss + vq_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss
        }
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            recon = outputs['recon']
            error = F.mse_loss(recon, x, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
        return error


if __name__ == "__main__":
    # Test VQ-VAE
    model = VQVAE(input_dim=784, num_embeddings=512, embedding_dim=64)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Continuous latent shape: {outputs['z'].shape}")
    print(f"Quantized latent shape: {outputs['z_q'].shape}")
    print(f"Indices shape: {outputs['indices'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"VQ Loss: {losses['vq_loss'].item():.4f}")
    
    # Check codebook usage
    usage = model.vq.get_codebook_usage(outputs['indices'])
    print(f"Codebook usage: {usage['usage_rate']*100:.1f}%")
