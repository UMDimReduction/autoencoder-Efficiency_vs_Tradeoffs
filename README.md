# Auto-Encoder Anomaly Detection: Efficiency vs Trade-offs

![Author](https://img.shields.io/badge/author-aaneloy-blue) 
![Author](https://img.shields.io/badge/turgeonmaxime-red)
[![MIT](https://img.shields.io/badge/license-MIT-5eba00.svg)](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs/blob/main/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)](https://pytorch.org/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs)

## Overview

This repository reproduces the paper **"A Comprehensive Study of Auto-Encoders for Anomaly Detection: Efficiency and Trade-Offs"** by **[Asif Ahmed Neloy](https://aaneloy.github.io/)** and **[Dr. Max Turgeon](https://www.maxturgeon.ca/)**, published in *Machine Learning with Applications*.

### Implemented Auto-Encoders (11 architectures)

| Model | Full Name | Reference |
|-------|-----------|-----------|
| **DAE** | Denoising Auto-Encoder | Vincent et al. (2008) |
| **SAE** | Sparse Auto-Encoder | Makhzani & Frey (2013) |
| **CAE** | Contractive Auto-Encoder | Rifai et al. (2011) |
| **VAE** | Variational Auto-Encoder | Kingma & Welling (2013) |
| **β-VAE** | Beta-VAE | Higgins et al. (2017) |
| **adVAE** | Self-Adversarial VAE | Wang et al. (2020) |
| **CVAE** | Conditional VAE | Pol et al. (2019) |
| **VQ-VAE** | Vector Quantized VAE | Marimont & Tarroni (2021) |
| **IWAE** | Importance Weighted AE | Burda et al. (2015) |
| **PAE** | Probabilistic AE | Böhm & Seljak (2020) |
| **RDA** | Robust Deep AE | Zhou & Paffenroth (2017) |


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs.git
cd autoencoder-Efficiency_vs_Tradeoffs

# Create virtual environment (recommended)
conda create -n autoencoder python=3.7 -y
conda activate autoencoder

# Install requirements
pip install -r requirements.txt
```

### Download Data

```bash
# Download all datasets (MNIST and Fashion-MNIST)
python download_data.py

# Download specific dataset
python download_data.py --dataset mnist
python download_data.py --dataset fashion_mnist
```

### Reproduce All Results

```bash
# Train and evaluate all models on MNIST
python main.py --dataset mnist --all

# Train and evaluate all models on Fashion-MNIST
python main.py --dataset fashion_mnist --all

# Quick test (5 epochs)
python main.py --dataset mnist --all --quick
```

### Train Individual Models

```bash
# Train a specific model
python main.py --dataset mnist --model vae --epochs 50

# Train multiple models
python main.py --dataset mnist --models vae beta_vae advae

# With custom parameters
python main.py --dataset mnist --model vae --epochs 100 --batch_size 64
```

### Evaluate Trained Models

```bash
# Evaluate a single model
python evaluate.py --model vae --checkpoint checkpoints/vae_mnist.pt --dataset mnist

# Evaluate all models and compare with paper
python evaluate.py --all --dataset mnist --compare_paper

# Generate visualizations
python evaluate.py --model vae --checkpoint checkpoints/vae_mnist.pt --visualize
```


## Expected Results (Table 3)

### MNIST ROC-AUC Scores

| Model | Paper |
|-------|-------|
| CAE | 0.22 |
| VAE | 0.61 |
| VQ-VAE | 0.82 |
| RDA | 0.82 |
| CVAE | 0.80 |
| SAE | 0.83 |
| DAE | 0.73 |
| β-VAE | 0.87 |
| PAE | 0.89 |
| IWAE | 0.87 |
| **adVAE** | **0.93** |

### Fashion-MNIST ROC-AUC Scores

| Model | Paper |
|-------|-------|
| CAE | 0.22 |
| VAE | 0.56 |
| VQ-VAE | 0.56 |
| RDA | 0.60 |
| CVAE | 0.66 |
| SAE | 0.66 |
| DAE | 0.56 |
| β-VAE | 0.59 |
| PAE | 0.64 |
| IWAE | 0.57 |
| **adVAE** | **0.87** |


## Project Structure

```
autoencoder-Efficiency_vs_Tradeoffs/
├── main.py                 # Main script to reproduce all results
├── evaluate.py             # Evaluation and metrics computation
├── download_data.py        # Dataset download script
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── README.md              # This file
│
├── configs/
│   ├── __init__.py
│   └── config.py          # Hyperparameters and expected results
│
├── models/
│   ├── __init__.py
│   ├── base.py            # Base auto-encoder class
│   ├── dae.py             # Denoising Auto-Encoder
│   ├── sae.py             # Sparse Auto-Encoder
│   ├── cae.py             # Contractive Auto-Encoder
│   ├── vae.py             # Variational Auto-Encoder
│   ├── beta_vae.py        # Beta-VAE
│   ├── advae.py           # Self-Adversarial VAE
│   ├── cvae.py            # Conditional VAE
│   ├── vqvae.py           # Vector Quantized VAE
│   └── others.py          # IWAE, PAE, RDA
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   ├── trainer.py         # Training utilities
│   └── visualization.py   # Plotting functions
│
├── checkpoints/           # Saved model weights (generated)
├── results/               # Evaluation results (generated)
└── data/                  # Datasets (downloaded, not tracked)
```


## Key Equations from Paper

### DAE (Eq. 1)

$$L_{DAE}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^{n} \left( x^{(i)} - f_\theta(g_\phi(\tilde{x}^{(i)})) \right)^2$$

### SAE (Eq. 2)

$$\tilde{X} = H_{W,b}(X) \approx X + \lambda \cdot \text{sparsity\_penalty}$$

### CAE (Eq. 3-4)

$$\|J_f(x)\|_F^2 = \sum_{i,j} \left( \frac{\partial h_j}{\partial x_i} \right)^2$$

$$L_{CAE} = L_{recon} + \lambda \|J_f(x)\|_F^2$$

### VAE (Eq. 5-9)

$$L_{VAE} = -\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) \| p_\theta(z))$$

### β-VAE (Eq. 12-13)

$$L_{\beta} = -\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot D_{KL}(q_\phi(z|x) \| p_\theta(z))$$

### CVAE (Eq. 10-11)

$$L_{CVAE} = -\mathbb{E}_{z \sim q_\phi(z|x,c)}[\log p_\theta(x|z,c)] + D_{KL}(q_\phi(z|x,c) \| p_\theta(z|c))$$

### adVAE (Eq. 14-15)

$$L_{adVAE} = -\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot D_{KL} + L_{adversarial}$$

### IWAE (Eq. 16)

$$L_k = \mathbb{E}_{h_1,...,h_k \sim q(h|x)} \left[ \log \frac{1}{k} \sum_{i=1}^{k} \frac{p(x,h_i)}{q(h_i|x)} \right]$$

### PAE (Eq. 17)

$$L = \mathbb{E}[|p(x|z)|] - \beta |D_{KL}(q(z|x) \| p(z)) - C|$$

### RDA (Eq. 18-19)

$$X = L_D + S$$

$$L = \|L_D - D_\theta(E_\theta(L_D))\|_2 + \lambda \|S^T\|_{2,1}$$

### VQ-VAE (Eq. 20-21)

$$z_q = e_k, \quad \text{where} \quad k = \arg\min_i \|E(x) - e_i\|_2$$

$$L = \|x - D(e_k)\|^2 + \|sg[E(x)] - e_k\|^2 + \beta \|E(x) - sg[e_k]\|^2$$


## Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Hardware
- GPU recommended for full training
- CPU-only mode supported (slower)

### System
- Windows 10/11, Linux, or macOS


## Citation

If you use this code, please cite the original paper:

```bibtex
@article{neloy2024autoencoder,
  title={A comprehensive study of auto-encoders for anomaly detection: Efficiency and trade-offs},
  author={Neloy, Asif Ahmed and Turgeon, Maxime},
  journal={Machine Learning with Applications},
  year={2024},
  publisher={Elsevier}
}
```


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

- **Asif Ahmed Neloy** - [Website](https://aaneloy.github.io/)
- **Dr. Max Turgeon** - [Website](https://www.maxturgeon.ca/)
