#!/usr/bin/env python3
"""
================================================================================
Self-Adversarial Variational Auto-Encoder (adVAE)
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Reference: Wang et al. (2020) - "adVAE: A self-adversarial variational autoencoder 
           with Gaussian anomaly prior knowledge for anomaly detection"

Run Command (standalone test):
    python models/advae.py

Key Equations (Eq. 14-15 from paper):
    max_{φ,θ} E_{x~D} [E_{z~q_φ(z|x)} [log p_θ(x|z)]]  (Eq. 14)
        where D_KL(q_φ(z|x)||p_θ(z)) < δ
    
    L_ADVAE(φ, β) = -E[log p_θ(x|z)] + β * D_KL(q_φ(z|x) || p_θ(z))  (Eq. 15)
    
Paper Notes:
    - Three networks: Encoder E, Decoder D, Transformer T
    - T (Gaussian transformer) regularizes decoder
    - Adversarial training distinguishes normal/anomalous
    - Two-step training process
    - Best performing model on MNIST (ROC-AUC: 0.93)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GaussianTransformer(nn.Module):
    """
    Gaussian Transformer network T.
    Transforms the latent representation of normal observations
    to predict anomalous latent codes.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminator network E (turned into discriminator).
    Distinguishes between normal and anomalous latent codes.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdVAE(nn.Module):
    """
    Self-Adversarial Variational Auto-Encoder
    
    Combines VAE with adversarial training to improve anomaly detection.
    Uses a Gaussian transformer and discriminator for two-step training.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        kl_weight: float = 1.0,
        discriminator_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.discriminator_weight = discriminator_weight
        
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
        
        # Gaussian transformer
        self.transformer = GaussianTransformer(latent_dim)
        
        # Discriminator
        self.discriminator = Discriminator(latent_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution"""
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
        """Forward pass through adVAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Generate anomalous latent code using transformer
        z_anomaly = self.transformer(z)
        
        # Discriminator predictions
        disc_normal = self.discriminator(z)
        disc_anomaly = self.discriminator(z_anomaly)
        
        return {
            'recon': recon,
            'z': z,
            'z_anomaly': z_anomaly,
            'mu': mu,
            'logvar': logvar,
            'disc_normal': disc_normal,
            'disc_anomaly': disc_anomaly
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        adVAE Loss (Eq. 15):
        L_ADVAE(φ, β) = -E_{z~q_φ(z|x)} [log p_θ(x|z)] + β D_KL(q_φ(z|x) || p_θ(z))
        
        Plus adversarial losses for discriminator training.
        
        Components:
        1. Reconstruction loss (VAE)
        2. KL divergence (VAE)
        3. Discriminator loss (adversarial)
        """
        x_flat = x.view(x.size(0), -1)
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        disc_normal = outputs['disc_normal']
        disc_anomaly = outputs['disc_anomaly']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Discriminator loss (BCE)
        # Normal samples should be classified as 1, anomalous as 0
        disc_loss_normal = F.binary_cross_entropy(
            disc_normal,
            torch.ones_like(disc_normal)
        )
        disc_loss_anomaly = F.binary_cross_entropy(
            disc_anomaly,
            torch.zeros_like(disc_anomaly)
        )
        disc_loss = (disc_loss_normal + disc_loss_anomaly) / 2
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss + \
                     self.discriminator_weight * disc_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'disc_loss': disc_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.view(num_samples, 1, 28, 28)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score combining reconstruction error and discriminator score.
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            
            # Reconstruction error
            recon_error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
            
            # Discriminator score (lower = more anomalous)
            disc_score = self.discriminator(mu).squeeze()
            
            # Combined score: high recon error + low disc score = anomaly
            anomaly_score = recon_error * (1 - disc_score)
        
        return anomaly_score
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean of latent distribution"""
        mu, _ = self.encode(x)
        return mu


class AdVAETrainer:
    """
    Two-step training procedure for adVAE as described in the paper.
    
    Step 1: Train decoder D and transformer T
    - D learns to generate distinct normal and anomalous samples
    - T minimizes KL divergence between normal and anomalous codes
    
    Step 2: Train encoder E as discriminator
    - E is updated with combined VAE and discrimination loss
    """
    
    def __init__(
        self,
        model: AdVAE,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model
        self.device = device
        
        # Separate optimizers for two-step training
        self.optimizer_encoder = torch.optim.Adam(
            list(model.encoder.parameters()) +
            list(model.fc_mu.parameters()) +
            list(model.fc_logvar.parameters()),
            lr=learning_rate
        )
        
        self.optimizer_decoder = torch.optim.Adam(
            model.decoder.parameters(),
            lr=learning_rate
        )
        
        self.optimizer_transformer = torch.optim.Adam(
            model.transformer.parameters(),
            lr=learning_rate
        )
        
        self.optimizer_discriminator = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=learning_rate
        )
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Execute one training step"""
        self.model.train()
        x = x.to(self.device)
        
        # Step 1: Update decoder and transformer
        self.optimizer_decoder.zero_grad()
        self.optimizer_transformer.zero_grad()
        
        outputs = self.model(x)
        x_flat = x.view(x.size(0), -1)
        
        # Decoder reconstruction loss
        recon_loss = F.mse_loss(outputs['recon'], x_flat, reduction='mean')
        
        # Transformer should minimize KL between normal and anomalous
        z = outputs['z'].detach()
        z_anomaly = self.model.transformer(z)
        transformer_loss = -0.5 * torch.mean(
            1 + torch.zeros_like(z_anomaly) - 
            z_anomaly.pow(2) - torch.ones_like(z_anomaly)
        )
        
        step1_loss = recon_loss + 0.1 * transformer_loss
        step1_loss.backward()
        self.optimizer_decoder.step()
        self.optimizer_transformer.step()
        
        # Step 2: Update encoder and discriminator
        self.optimizer_encoder.zero_grad()
        self.optimizer_discriminator.zero_grad()
        
        outputs = self.model(x)
        losses = self.model.loss_function(x, outputs)
        
        losses['loss'].backward()
        self.optimizer_encoder.step()
        self.optimizer_discriminator.step()
        
        return {
            'loss': losses['loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'kl_loss': losses['kl_loss'].item(),
            'disc_loss': losses['disc_loss'].item()
        }


if __name__ == "__main__":
    # Test adVAE
    model = AdVAE(input_dim=784, latent_dim=2)
    x = torch.randn(16, 1, 28, 28)
    
    outputs = model(x)
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Anomalous latent shape: {outputs['z_anomaly'].shape}")
    
    losses = model.loss_function(x, outputs)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"KL Loss: {losses['kl_loss'].item():.4f}")
    print(f"Disc Loss: {losses['disc_loss'].item():.4f}")
