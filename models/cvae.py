#!/usr/bin/env python3
"""
================================================================================
Conditional Variational Auto-Encoder (CVAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Pol et al. (2019) - "Anomaly detection with conditional variational 
           autoencoders"

Run Command (standalone test):
    python models/cvae.py

Key Equations (Eq. 10-11 from paper):
    L_VAE(θ, φ) = -E[log p_θ(x|z)] + D_KL(q_φ(z|x) || p_θ(z))  (Eq. 10)
    
    L_CVAE(θ, φ) = -E[log p_θ(x|z,c)] + D_KL(q_φ(z|x,c) || p_θ(z|c))  (Eq. 11)
    
Paper Notes:
    - Extends VAE with conditioning variable c (class label)
    - Decoder: p_θ(x|z,c), Recognition: q_φ(z|x,c)
    - Enables controlled generation based on labels
    - 10 classes for MNIST/Fashion-MNIST
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CVAE(nn.Module):
    """
    Conditional Variational Auto-Encoder
    
    Extends VAE by conditioning on class labels, allowing for
    controlled generation and better representation learning.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        num_classes: int = 10,
        kl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Encoder: takes input + label
        encoder_layers = []
        in_dim = input_dim + num_classes  # Concatenate input with label
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: takes latent + label
        decoder_layers = []
        in_dim = latent_dim + num_classes
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
    
    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input conditioned on label.
        q_φ(z|x,c)
        """
        x_flat = x.view(x.size(0), -1)
        
        # One-hot encode labels
        c_onehot = F.one_hot(c, self.num_classes).float()
        
        # Concatenate input with label
        x_cond = torch.cat([x_flat, c_onehot], dim=1)
        
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode latent conditioned on label.
        p_θ(x|z,c)
        """
        # One-hot encode labels
        c_onehot = F.one_hot(c, self.num_classes).float()
        
        # Concatenate latent with label
        z_cond = torch.cat([z, c_onehot], dim=1)
        
        return self.decoder(z_cond)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CVAE.
        
        Args:
            x: Input images
            c: Class labels (optional, random if not provided)
        """
        if c is None:
            # Use random labels if not provided
            c = torch.randint(0, self.num_classes, (x.size(0),), device=x.device)
        
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        
        return {
            'recon': recon,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'c': c
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        CVAE Loss (Eq. 11):
        L_CVAE(θ, φ) = -E_{z~q_φ(z|c,x)} [log p_θ(x|z,c)] + D_KL(q_φ(z|x,c) || p_θ(z|c))
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(
        self,
        num_samples: int,
        c: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate samples conditioned on class labels.
        """
        if c is None:
            c = torch.randint(0, self.num_classes, (num_samples,), device=device)
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, c)
        return samples.view(num_samples, 1, 28, 28)
    
    def generate_by_class(
        self,
        class_idx: int,
        num_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate samples for a specific class"""
        c = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        return self.sample(num_samples, c, device)
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly scoring.
        If labels not provided, compute error across all possible labels
        and return minimum.
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            
            if c is not None:
                # Use provided labels
                mu, _ = self.encode(x, c)
                recon = self.decode(mu, c)
                error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
            else:
                # Compute minimum error across all classes
                errors = []
                for class_idx in range(self.num_classes):
                    c_temp = torch.full(
                        (x.size(0),), class_idx,
                        dtype=torch.long, device=x.device
                    )
                    mu, _ = self.encode(x, c_temp)
                    recon = self.decode(mu, c_temp)
                    error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
                    errors.append(error)
                
                # Use minimum error (best matching class)
                error = torch.stack(errors, dim=1).min(dim=1)[0]
        
        return error
    
    def get_latent(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Get mean of latent distribution"""
        mu, _ = self.encode(x, c)
        return mu


class ConditionalDecoder(nn.Module):
    """
    Decoder that can be conditioned on multiple factors.
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        super().__init__()
        
        layers = []
        in_dim = latent_dim + condition_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        layers.extend([
            nn.Linear(in_dim, output_dim),
            nn.Sigmoid()
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([z, condition], dim=1)
        return self.net(x)


if __name__ == "__main__":
    # Test CVAE
    model = CVAE(input_dim=784, latent_dim=2, num_classes=10)
    x = torch.randn(16, 1, 28, 28)
    c = torch.randint(0, 10, (16,))
    
    outputs = model(x, c)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"KL Loss: {losses['kl_loss'].item():.4f}")
    
    # Test generation by class
    samples = model.generate_by_class(5, 4, x.device)
    print(f"Class-conditioned samples shape: {samples.shape}")
