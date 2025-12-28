#!/usr/bin/env python3
"""
================================================================================
Main Script - Auto-Encoder Anomaly Detection Study
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon
Journal: Machine Learning with Applications

Run Commands:
    # Train and evaluate all models on MNIST
    python main.py --dataset mnist --all

    # Train and evaluate all models on Fashion-MNIST
    python main.py --dataset fashion_mnist --all

    # Train a specific model
    python main.py --dataset mnist --model vae --epochs 50

    # Train multiple specific models
    python main.py --dataset mnist --models vae beta_vae advae

    # Quick test run (reduced epochs)
    python main.py --dataset mnist --model vae --epochs 5 --quick

    # Generate visualizations
    python main.py --dataset mnist --all --visualize

Description:
    Trains and evaluates 11 auto-encoder architectures for anomaly detection:
    - DAE  (Denoising Auto-Encoder)
    - SAE  (Sparse Auto-Encoder)
    - CAE  (Contractive Auto-Encoder)
    - VAE  (Variational Auto-Encoder)
    - β-VAE (Beta-VAE)
    - adVAE (Self-Adversarial VAE)
    - CVAE (Conditional VAE)
    - VQ-VAE (Vector Quantized VAE)
    - IWAE (Importance Weighted Auto-Encoder)
    - PAE  (Probabilistic Auto-Encoder)
    - RDA  (Robust Deep Auto-Encoder)

Expected Results (Table 3 from paper):
    MNIST ROC-AUC:
        CAE: 0.22, VAE: 0.61, VQ-VAE: 0.82, RDA: 0.82, CVAE: 0.80
        SAE: 0.83, DAE: 0.73, β-VAE: 0.87, PAE: 0.89, IWAE: 0.87, adVAE: 0.93
    
    Fashion-MNIST ROC-AUC:
        CAE: 0.22, VAE: 0.56, VQ-VAE: 0.56, RDA: 0.60, CVAE: 0.66
        SAE: 0.66, DAE: 0.56, β-VAE: 0.59, PAE: 0.64, IWAE: 0.57, adVAE: 0.87
================================================================================
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    DEVICE, TRAINING_CONFIG, MODEL_CONFIGS,
    EXPECTED_RESULTS, DATASET_CONFIG
)
from utils.data_loader import (
    get_anomaly_detection_loaders,
    get_standard_loaders
)

# Import all models
from models.dae import DAE
from models.sae import SAE
from models.cae import CAE
from models.vae import VAE
from models.beta_vae import BetaVAE
from models.advae import AdVAE
from models.cvae import CVAE
from models.vqvae import VQVAE
from models.others import IWAE, PAE, RDA


# Model registry
MODEL_CLASSES = {
    'dae': DAE,
    'sae': SAE,
    'cae': CAE,
    'vae': VAE,
    'beta_vae': BetaVAE,
    'advae': AdVAE,
    'cvae': CVAE,
    'vqvae': VQVAE,
    'iwae': IWAE,
    'pae': PAE,
    'rda': RDA,
}

ALL_MODELS = list(MODEL_CLASSES.keys())


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create an auto-encoder model.
    
    Args:
        model_name: Name of the model
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Instantiated model
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")
    
    config = MODEL_CONFIGS.get(model_name, {}).copy()
    config.update(kwargs)
    
    model_class = MODEL_CLASSES[model_name]
    
    # Model-specific instantiation
    if model_name == 'dae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 32),
            noise_factor=config.get('noise_factor', 0.27)
        )
    elif model_name == 'sae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 32),
            sparsity_weight=config.get('sparsity_weight', 1e-3),
            sparsity_target=config.get('sparsity_target', 0.45)
        )
    elif model_name == 'cae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 32),
            lambda_=config.get('lambda_', 1e-4)
        )
    elif model_name == 'vae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2),
            kl_weight=config.get('kl_weight', 1.0)
        )
    elif model_name == 'beta_vae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2),
            beta=config.get('beta', 1.5)
        )
    elif model_name == 'advae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2)
        )
    elif model_name == 'cvae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2),
            num_classes=config.get('num_classes', 10)
        )
    elif model_name == 'vqvae':
        model = model_class(
            in_channels=1,
            hidden_dims=config.get('hidden_dims', [128, 256]),
            num_embeddings=config.get('num_embeddings', 512),
            embedding_dim=config.get('embedding_dim', 64),
            commitment_cost=config.get('commitment_cost', 0.25)
        )
    elif model_name == 'iwae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2),
            num_samples=config.get('num_samples', 50)
        )
    elif model_name == 'pae':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 2),
            beta=config.get('beta', 1.0),
            C=config.get('C', 0.5)
        )
    elif model_name == 'rda':
        model = model_class(
            input_dim=784,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            latent_dim=config.get('latent_dim', 32),
            lambda_=config.get('lambda_', 1e-3)
        )
    else:
        model = model_class(**config)
    
    return model


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = True
) -> Dict:
    """
    Train an auto-encoder model.
    
    Args:
        model: Model to train
        model_name: Name of the model
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
        verbose: Print progress
        
    Returns:
        Training history
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'loss': [], 'epoch_times': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', disable=not verbose)
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass based on model type
            if model_name == 'cvae':
                outputs = model(data, labels.long())
            else:
                outputs = model(data)
            
            # Compute loss
            if isinstance(outputs, dict):
                losses = model.loss_function(data, outputs)
                loss = losses['loss']
            elif isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = F.mse_loss(outputs, data.view(data.size(0), -1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose:
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        epoch_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        history['loss'].append(epoch_loss)
        history['epoch_times'].append(epoch_time)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Time: {epoch_time:.1f}s")
    
    history['total_time'] = time.time() - start_time
    
    return history


def evaluate_model(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    normal_class: int = 0,
    device: torch.device = DEVICE
) -> Dict:
    """
    Evaluate model for anomaly detection.
    
    Args:
        model: Trained model
        model_name: Name of the model
        test_loader: Test data loader
        normal_class: Normal class index
        device: Device for evaluation
        
    Returns:
        Evaluation results
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    model = model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Get reconstruction
            if model_name == 'cvae':
                outputs = model(data, labels.long().to(device))
            else:
                outputs = model(data)
            
            if isinstance(outputs, dict):
                recon = outputs.get('recon', outputs.get('reconstruction'))
            elif isinstance(outputs, tuple):
                recon = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                recon = outputs
            
            # Compute reconstruction error
            data_flat = data.view(batch_size, -1)
            recon_flat = recon.view(batch_size, -1)
            scores = torch.mean((data_flat - recon_flat) ** 2, dim=1)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    
    # Compute metrics
    roc_auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    # Find optimal threshold
    best_f1 = 0
    for threshold in np.percentile(scores, np.linspace(0, 100, 100)):
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    
    return {
        'roc_auc': roc_auc,
        'ap': ap,
        'f1': best_f1,
        'scores': scores,
        'labels': labels
    }


def run_experiment(
    model_name: str,
    dataset: str,
    epochs: int = 50,
    batch_size: int = 128,
    normal_class: int = 0,
    save_dir: str = './checkpoints',
    device: torch.device = DEVICE,
    verbose: bool = True
) -> Dict:
    """
    Run a complete experiment for one model.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name ('mnist' or 'fashion_mnist')
        epochs: Number of training epochs
        batch_size: Batch size
        normal_class: Normal class for anomaly detection
        save_dir: Directory to save checkpoints
        device: Device for training
        verbose: Print progress
        
    Returns:
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset.upper()}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model(model_name)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    train_loader, test_loader = get_anomaly_detection_loaders(
        dataset_name=dataset,
        normal_class=normal_class,
        batch_size=batch_size
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train
    lr = MODEL_CONFIGS.get(model_name, {}).get('learning_rate', 1e-3)
    history = train_model(
        model, model_name, train_loader,
        epochs=epochs, lr=lr, device=device, verbose=verbose
    )
    
    print(f"\nTraining completed in {history['total_time']:.1f} seconds")
    
    # Evaluate
    results = evaluate_model(model, model_name, test_loader, normal_class, device)
    
    print(f"\nResults:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AP: {results['ap']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    # Compare with paper
    expected = EXPECTED_RESULTS.get(dataset, {}).get(model_name)
    if expected:
        diff = results['roc_auc'] - expected
        print(f"  Paper ROC-AUC: {expected:.2f} (diff: {diff:+.4f})")
    
    # Save checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{model_name}_{dataset}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'dataset': dataset,
        'history': history,
        'results': {k: v for k, v in results.items() if k not in ['scores', 'labels']}
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    
    return {
        'model_name': model_name,
        'dataset': dataset,
        'history': history,
        'results': results
    }


def run_all_experiments(
    dataset: str,
    models: Optional[List[str]] = None,
    epochs: int = 50,
    batch_size: int = 128,
    normal_class: int = 0,
    save_dir: str = './checkpoints',
    results_dir: str = './results',
    device: torch.device = DEVICE,
    verbose: bool = True
) -> Dict:
    """
    Run experiments for multiple models.
    
    Args:
        dataset: Dataset name
        models: List of models to train (default: all)
        epochs: Number of training epochs
        batch_size: Batch size
        normal_class: Normal class for anomaly detection
        save_dir: Directory to save checkpoints
        results_dir: Directory to save results
        device: Device for training
        verbose: Print progress
        
    Returns:
        All experiment results
    """
    if models is None:
        models = ALL_MODELS
    
    print("\n" + "=" * 70)
    print("AUTO-ENCODER ANOMALY DETECTION - FULL EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 70)
    
    all_results = {}
    
    for model_name in models:
        try:
            result = run_experiment(
                model_name, dataset, epochs, batch_size,
                normal_class, save_dir, device, verbose
            )
            all_results[model_name] = result
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'ROC-AUC':<12} {'AP':<12} {'F1':<12} {'Time (s)':<12}")
    print("-" * 70)
    
    for model_name, result in all_results.items():
        r = result['results']
        t = result['history']['total_time']
        print(f"{model_name:<15} {r['roc_auc']:<12.4f} {r['ap']:<12.4f} {r['f1']:<12.4f} {t:<12.1f}")
    
    print("=" * 70)
    
    # Compare with paper
    print("\nCOMPARISON WITH PAPER (Table 3)")
    print("-" * 70)
    print(f"{'Model':<15} {'Our AUC':<12} {'Paper AUC':<12} {'Difference':<12}")
    print("-" * 70)
    
    expected = EXPECTED_RESULTS.get(dataset, {})
    for model_name, result in all_results.items():
        our_auc = result['results']['roc_auc']
        paper_auc = expected.get(model_name, 'N/A')
        if isinstance(paper_auc, (int, float)):
            diff = our_auc - paper_auc
            print(f"{model_name:<15} {our_auc:<12.4f} {paper_auc:<12.2f} {diff:<+12.4f}")
        else:
            print(f"{model_name:<15} {our_auc:<12.4f} {'N/A':<12} {'N/A':<12}")
    
    print("=" * 70)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON
    json_results = {}
    for model_name, result in all_results.items():
        json_results[model_name] = {
            'roc_auc': float(result['results']['roc_auc']),
            'ap': float(result['results']['ap']),
            'f1': float(result['results']['f1']),
            'training_time': float(result['history']['total_time'])
        }
    
    json_path = os.path.join(results_dir, f'results_{dataset}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Auto-Encoder Anomaly Detection Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --dataset mnist --all
    python main.py --dataset fashion_mnist --all
    python main.py --dataset mnist --model vae --epochs 50
    python main.py --dataset mnist --models vae beta_vae advae
    python main.py --dataset mnist --all --quick
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'fashion_mnist'],
        default='mnist',
        help='Dataset to use (default: mnist)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=ALL_MODELS,
        help='Single model to train'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=ALL_MODELS,
        help='Multiple models to train'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all models'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )
    
    parser.add_argument(
        '--normal_class',
        type=int,
        default=0,
        help='Normal class for anomaly detection (default: 0)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with reduced epochs (5)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = DEVICE
    if args.no_cuda:
        device = torch.device('cpu')
    
    # Set epochs
    epochs = args.epochs
    if args.quick:
        epochs = 5
        print("Quick mode: using 5 epochs")
    
    # Determine which models to train
    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        print("\nError: Please specify --model, --models, or --all")
        return
    
    # Run experiments
    run_all_experiments(
        dataset=args.dataset,
        models=models,
        epochs=epochs,
        batch_size=args.batch_size,
        normal_class=args.normal_class,
        save_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=device,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
