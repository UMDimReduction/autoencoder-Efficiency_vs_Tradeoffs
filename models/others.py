#!/usr/bin/env python3
"""
================================================================================
IWAE, PAE, and RDA Auto-Encoders
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Run Command (standalone test):
    python models/others.py

================================================================================
IWAE - Importance Weighted Auto-Encoder (Section 3.9)
================================================================================
Reference: Burda et al. (2015) - "Importance weighted autoencoders"

Key Equation (Eq. 16 from paper):
    L_k(x) = E_{h_1,...,h_k~q(h|x)} [log (1/k) Σ p(x,h_i)/q(h_i|x)]

Paper Notes:
    - Uses k=50 importance samples
    - Tighter ELBO bound than VAE
    - Log-sum-exp for numerical stability

================================================================================
PAE - Probabilistic Auto-Encoder (Section 3.10)
================================================================================
Reference: Böhm & Seljak (2020) - "Probabilistic auto-encoder"

Key Equation (Eq. 17 from paper):
    L = E[|p(x|z)|] - β|D_KL(q(z|x)||p(z)) - C|

Paper Notes:
    - Two-stage generative model
    - Absolute value around KL creates "dead zone"
    - Beta: 1.0, C: 0.5

================================================================================
RDA - Robust Deep Auto-Encoder (Section 3.11)
================================================================================
Reference: Zhou & Paffenroth (2017) - "Anomaly detection with robust deep 
           autoencoders"

Key Equations (Eq. 18-19 from paper):
    Decomposition: X = L_D + S  (Eq. 18)
    Loss: min||L_D - D_θ(E_θ(L_D))||₂ + λ||S^T||_{2,1}  (Eq. 19)

Paper Notes:
    - Inspired by Robust PCA
    - L_D: low-rank component
    - S: sparse outlier component
    - Lambda: 1e-3
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================================
# IWAE - Importance Weighted Auto-Encoder
# ============================================================================

class IWAE(nn.Module):
    """
    Importance Weighted Auto-Encoder
    
    Uses importance sampling to compute a tighter bound on the log-likelihood.
    Instead of single sample, uses k samples and importance-weighted average.
    
    From paper (Section 3.9, Eq. 16):
    L_k(x) = E_{h_1,...,h_k~q(h|x)} [log (1/k) Σ_{i=1}^k p(x,h_i)/q(h_i|x)]
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        num_samples: int = 50,  # k=50 as mentioned in paper
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        
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
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Reparameterization with multiple samples.
        Returns shape: [num_samples, batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        
        # Expand for multiple samples
        mu = mu.unsqueeze(0).expand(num_samples, -1, -1)
        std = std.unsqueeze(0).expand(num_samples, -1, -1)
        
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with importance sampling"""
        mu, logvar = self.encode(x)
        
        # Sample k latent vectors for each input
        z = self.reparameterize(mu, logvar, self.num_samples)  # [k, B, D]
        
        # Decode each sample
        batch_size = x.size(0)
        z_flat = z.view(-1, self.latent_dim)  # [k*B, D]
        recon_flat = self.decode(z_flat)  # [k*B, input_dim]
        recon = recon_flat.view(self.num_samples, batch_size, -1)  # [k, B, input_dim]
        
        return {
            'recon': recon,  # [k, B, input_dim]
            'z': z,  # [k, B, latent_dim]
            'mu': mu,
            'logvar': logvar
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        IWAE Loss (Eq. 16):
        L_k(x) = E[log (1/k) Σ_{i=1}^k p(x,h_i)/q(h_i|x)]
        
        This is computed using log-sum-exp for numerical stability:
        L_k = log (1/k) + logsumexp(log w_i) where w_i = p(x,h_i)/q(h_i|x)
        """
        x_flat = x.view(x.size(0), -1)  # [B, input_dim]
        recon = outputs['recon']  # [k, B, input_dim]
        z = outputs['z']  # [k, B, latent_dim]
        mu = outputs['mu']  # [B, latent_dim]
        logvar = outputs['logvar']  # [B, latent_dim]
        
        k = self.num_samples
        batch_size = x.size(0)
        
        # Expand x for k samples
        x_expanded = x_flat.unsqueeze(0).expand(k, -1, -1)  # [k, B, input_dim]
        
        # Log p(x|z) - reconstruction log probability (Gaussian)
        recon_loss = -0.5 * ((x_expanded - recon) ** 2).sum(dim=2)  # [k, B]
        
        # Log p(z) - prior log probability (standard normal)
        log_pz = -0.5 * (z ** 2).sum(dim=2)  # [k, B]
        
        # Log q(z|x) - posterior log probability
        std = torch.exp(0.5 * logvar)
        mu_expanded = mu.unsqueeze(0).expand(k, -1, -1)
        std_expanded = std.unsqueeze(0).expand(k, -1, -1)
        
        log_qz = -0.5 * (((z - mu_expanded) / std_expanded) ** 2).sum(dim=2)
        log_qz = log_qz - 0.5 * self.latent_dim * torch.log(torch.tensor(2 * 3.14159))
        log_qz = log_qz - logvar.sum(dim=1).unsqueeze(0) * 0.5
        
        # Log importance weights: log p(x,z) - log q(z|x)
        log_weights = recon_loss + log_pz - log_qz  # [k, B]
        
        # IWAE bound using log-sum-exp
        iwae_bound = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(float(k)))
        
        # Negative IWAE bound as loss
        loss = -iwae_bound.mean()
        
        # For monitoring, compute standard VAE terms
        recon_loss_mean = F.mse_loss(recon[0], x_flat, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return {
            'loss': loss,
            'recon_loss': recon_loss_mean,
            'kl_loss': kl_loss,
            'iwae_bound': iwae_bound.mean()
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score"""
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


# ============================================================================
# PAE - Probabilistic Auto-Encoder
# ============================================================================

class PAE(nn.Module):
    """
    Probabilistic Auto-Encoder
    
    Two-stage generative model that interprets AE probabilistically
    after training with Normalizing Flow.
    
    From paper (Section 3.10, Eq. 17):
    L_{β-VAE} = E_{q_φ(z|x)} [|p_θ(x|z)|] - β |D_{KL}(q_φ(z|x)||p(z)) - C|
    
    Parameters β and C control trade-off between reconstruction and regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        beta: float = 1.0,
        C: float = 0.5,  # Regularization constant
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.C = C
        
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
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
        """
        PAE Loss (Eq. 17):
        L = E[|p(x|z)|] - β|D_KL(q(z|x)||p(z)) - C|
        
        Uses absolute value around KL divergence to create a "dead zone"
        around the capacity C.
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Likelihood term (reconstruction)
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # PAE regularization: β|KL - C|
        pae_reg = self.beta * torch.abs(kl_loss - self.C)
        
        # Total loss
        total_loss = recon_loss + pae_reg
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'pae_reg': pae_reg
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


# ============================================================================
# RDA - Robust Deep Auto-Encoder
# ============================================================================

class RDA(nn.Module):
    """
    Robust Deep Auto-Encoder
    
    Inspired by Robust PCA (RPCA), decomposes data into low-rank (L_D)
    and sparse (S) components.
    
    From paper (Section 3.11, Eq. 18-19):
    min_{θ,S} ||L_D - D_θ(E_θ(L_D))||_2 + λ||S^T||_{2,1}, where X - L_D - S = 0
    
    Encoder and decoder functions (Eq. 19):
    E_θ(x) = E_{W,b}(x) = logit(Wx + b_E)
    D_θ(x) = D_{W,b}(x) = logit(W^T E_{W,b}(x) + b_D)
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        lambda_: float = 1e-3,  # Sparsity weight
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        
        # Build encoder with tied weights option
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Sigmoid()  # logit activation as in Eq. 19
            ])
            in_dim = h_dim
        
        encoder_layers.extend([
            nn.Linear(in_dim, latent_dim),
            nn.Sigmoid()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Sigmoid()
            ])
            in_dim = h_dim
        
        decoder_layers.extend([
            nn.Linear(in_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Sparse component (learnable)
        self.register_buffer('S', torch.zeros(1, input_dim))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass decomposing X into L_D (low-rank) and S (sparse).
        """
        x_flat = x.view(x.size(0), -1)
        
        # L_D = X - S (low-rank component)
        L_D = x_flat - self.S
        
        # Encode low-rank component
        z = self.encoder(L_D)
        
        # Reconstruct
        recon = self.decoder(z)
        
        return {
            'recon': recon,
            'z': z,
            'L_D': L_D,
            'S': self.S.expand(x.size(0), -1),
            'x_flat': x_flat
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        RDA Loss (Eq. 18):
        L = ||L_D - D(E(L_D))||_2 + λ||S^T||_{2,1}
        
        Where ||S^T||_{2,1} is the L2,1 norm (sum of L2 norms of columns).
        """
        L_D = outputs['L_D']
        recon = outputs['recon']
        S = outputs['S']
        
        # Reconstruction loss (low-rank component)
        recon_loss = F.mse_loss(recon, L_D, reduction='mean')
        
        # Sparsity regularization: L2,1 norm of S
        # ||S^T||_{2,1} = sum of L2 norms of rows of S
        sparsity_loss = torch.norm(S, p=2, dim=1).mean()
        
        # Total loss
        total_loss = recon_loss + self.lambda_ * sparsity_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def update_sparse_component(
        self,
        x: torch.Tensor,
        learning_rate: float = 0.01
    ):
        """
        Update sparse component S using proximal gradient.
        """
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            
            # Compute residual
            L_D = x_flat - self.S
            z = self.encoder(L_D)
            recon = self.decoder(z)
            residual = x_flat - recon
            
            # Update S towards residual with soft thresholding
            self.S = self.S + learning_rate * (residual.mean(dim=0, keepdim=True) - self.S)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score based on reconstruction + sparse magnitude"""
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            outputs = self.forward(x)
            
            # Reconstruction error of low-rank component
            recon_error = F.mse_loss(
                outputs['recon'], outputs['L_D'],
                reduction='none'
            ).mean(dim=1)
            
            # Magnitude of sparse component for each sample
            sparse_magnitude = torch.norm(outputs['S'], p=2, dim=1)
            
            # Combined score
            error = recon_error + 0.1 * sparse_magnitude
        
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return outputs['z']


if __name__ == "__main__":
    x = torch.randn(16, 1, 28, 28)
    
    # Test IWAE
    print("Testing IWAE...")
    iwae = IWAE(input_dim=784, latent_dim=2, num_samples=5)
    outputs = iwae(x)
    losses = iwae.loss_function(x, outputs)
    print(f"IWAE Loss: {losses['loss'].item():.4f}")
    
    # Test PAE
    print("\nTesting PAE...")
    pae = PAE(input_dim=784, latent_dim=2)
    outputs = pae(x)
    losses = pae.loss_function(x, outputs)
    print(f"PAE Loss: {losses['loss'].item():.4f}")
    
    # Test RDA
    print("\nTesting RDA...")
    rda = RDA(input_dim=784, latent_dim=32)
    outputs = rda(x)
    losses = rda.loss_function(x, outputs)
    print(f"RDA Loss: {losses['loss'].item():.4f}")
