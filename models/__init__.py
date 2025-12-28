"""
================================================================================
Auto-Encoder Models Package
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

This package contains implementations of 11 auto-encoder architectures:

1. DAE  - Denoising Auto-Encoder (Vincent et al., 2008)
2. SAE  - Sparse Auto-Encoder (Ng, 2011)
3. CAE  - Contractive Auto-Encoder (Rifai et al., 2011)
4. VAE  - Variational Auto-Encoder (Kingma & Welling, 2013)
5. β-VAE - Beta-VAE (Higgins et al., 2017)
6. adVAE - Self-Adversarial VAE (Khalifa & Gao, 2020)
7. CVAE - Conditional VAE (Sohn et al., 2015)
8. VQ-VAE - Vector Quantized VAE (van den Oord et al., 2017)
9. IWAE - Importance Weighted Auto-Encoder (Burda et al., 2015)
10. PAE - Probabilistic Auto-Encoder (Ghosh et al., 2020)
11. RDA - Robust Deep Auto-Encoder (Zhou & Paffenroth, 2017)

Usage:
    from models import DAE, VAE, VQVAE, ...
    
    model = VAE(input_dim=784, latent_dim=2)
================================================================================
"""

from .base import BaseAutoEncoder, ConvEncoder, ConvDecoder, reparameterize, kl_divergence
from .dae import DAE, ConvDAE
from .sae import SAE
from .cae import CAE
from .vae import VAE, ConvVAE
from .beta_vae import BetaVAE
from .advae import AdVAE
from .cvae import CVAE
from .vqvae import VQVAE
from .others import IWAE, PAE, RDA

__all__ = [
    # Base
    'BaseAutoEncoder',
    'ConvEncoder',
    'ConvDecoder',
    'reparameterize',
    'kl_divergence',
    # Models
    'DAE',
    'ConvDAE',
    'SAE',
    'CAE',
    'VAE',
    'ConvVAE',
    'BetaVAE',
    'AdVAE',
    'CVAE',
    'VQVAE',
    'IWAE',
    'PAE',
    'RDA',
]
