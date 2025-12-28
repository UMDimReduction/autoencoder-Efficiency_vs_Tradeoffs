#!/usr/bin/env python3
"""
================================================================================
Evaluation Script for Auto-Encoder Anomaly Detection
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Run Command:
    python evaluate.py --model MODEL_NAME --checkpoint PATH [options]

Examples:
    # Evaluate a single model
    python evaluate.py --model vae --checkpoint checkpoints/vae_mnist.pt --dataset mnist

    # Evaluate all models
    python evaluate.py --all --dataset mnist

    # Generate visualizations
    python evaluate.py --model vae --checkpoint checkpoints/vae_mnist.pt --visualize

    # Compare with paper results
    python evaluate.py --all --compare_paper

Description:
    Evaluates trained auto-encoder models for anomaly detection.
    
    Metrics computed:
    - ROC-AUC (Area Under ROC Curve)
    - Average Precision (AP)
    - Precision, Recall, F1-Score
    - Reconstruction Error Statistics
================================================================================
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    confusion_matrix
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DEVICE, MODEL_CONFIGS, EXPECTED_RESULTS
from utils.data_loader import get_anomaly_detection_loaders


def compute_reconstruction_error(
    model,
    data_loader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction error for all samples.
    
    Args:
        model: Trained auto-encoder model
        data_loader: DataLoader with test data
        device: Device for computation
        
    Returns:
        scores: Anomaly scores (reconstruction errors)
        labels: Ground truth labels (0=normal, 1=anomaly)
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Computing scores"):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Get reconstruction
            outputs = model(data)
            
            if isinstance(outputs, dict):
                recon = outputs.get('recon', outputs.get('reconstruction'))
            elif isinstance(outputs, tuple):
                recon = outputs[1]
            else:
                recon = outputs
            
            # Compute per-sample MSE
            data_flat = data.view(batch_size, -1)
            recon_flat = recon.view(batch_size, -1)
            mse = torch.mean((data_flat - recon_flat) ** 2, dim=1)
            
            all_scores.extend(mse.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_scores), np.array(all_labels)


def evaluate_anomaly_detection(
    scores: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    Evaluate anomaly detection performance.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0=normal, 1=anomaly)
        
    Returns:
        Dictionary with all metrics
    """
    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    
    # Average Precision
    ap = average_precision_score(labels, scores)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    
    # Find optimal threshold using F1 score
    best_f1 = 0
    best_threshold = 0
    for threshold in np.percentile(scores, np.linspace(0, 100, 100)):
        pred_labels = (scores >= threshold).astype(int)
        f1 = f1_score(labels, pred_labels, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Get metrics at optimal threshold
    pred_labels = (scores >= best_threshold).astype(int)
    precision_at_threshold = precision_score(labels, pred_labels, zero_division=0)
    recall_at_threshold = recall_score(labels, pred_labels, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    
    return {
        'roc_auc': roc_auc,
        'ap': ap,
        'f1': best_f1,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'optimal_threshold': best_threshold,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision_curve': precision.tolist(),
        'recall_curve': recall.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'normal_scores_mean': float(np.mean(scores[labels == 0])),
        'normal_scores_std': float(np.std(scores[labels == 0])),
        'anomaly_scores_mean': float(np.mean(scores[labels == 1])),
        'anomaly_scores_std': float(np.std(scores[labels == 1])),
    }


def plot_roc_curve(
    results: Dict,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Plot ROC curve for a model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(results['fpr'], results['tpr'], 'b-', linewidth=2,
            label=f"AUC = {results['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of anomaly scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    
    ax.set_xlabel('Reconstruction Error (Anomaly Score)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Anomaly Score Distribution - {model_name}', fontsize=14)
    ax.legend(fontsize=11)
    
    # Add statistics
    stats_text = (
        f"Normal: μ={np.mean(normal_scores):.4f}, σ={np.std(normal_scores):.4f}\n"
        f"Anomaly: μ={np.mean(anomaly_scores):.4f}, σ={np.std(anomaly_scores):.4f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_with_paper(
    results_dict: Dict[str, Dict],
    dataset: str
) -> None:
    """Compare results with paper's Table 3."""
    print("\n" + "=" * 70)
    print(f"COMPARISON WITH PAPER RESULTS - {dataset.upper()}")
    print("=" * 70)
    print(f"{'Model':<15} {'Our ROC-AUC':<15} {'Paper ROC-AUC':<15} {'Difference':<15}")
    print("-" * 70)
    
    expected = EXPECTED_RESULTS.get(dataset, {})
    
    for model_name, results in results_dict.items():
        our_auc = results['roc_auc']
        paper_auc = expected.get(model_name, 'N/A')
        
        if isinstance(paper_auc, (int, float)):
            diff = our_auc - paper_auc
            diff_str = f"{diff:+.3f}"
        else:
            diff_str = "N/A"
        
        paper_str = f"{paper_auc:.2f}" if isinstance(paper_auc, (int, float)) else paper_auc
        print(f"{model_name:<15} {our_auc:<15.3f} {paper_str:<15} {diff_str:<15}")
    
    print("=" * 70)


def generate_results_table(
    results_dict: Dict[str, Dict],
    dataset: str,
    save_path: Optional[str] = None
) -> str:
    """Generate a formatted results table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"ANOMALY DETECTION RESULTS - {dataset.upper()}")
    lines.append("=" * 80)
    lines.append(f"{'Model':<12} {'ROC-AUC':<10} {'AP':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    lines.append("-" * 80)
    
    for model_name, results in results_dict.items():
        lines.append(
            f"{model_name:<12} {results['roc_auc']:<10.3f} {results['ap']:<10.3f} "
            f"{results['f1']:<10.3f} {results['precision']:<12.3f} {results['recall']:<10.3f}"
        )
    
    lines.append("=" * 80)
    
    table = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
    
    return table


def load_model(model_name: str, checkpoint_path: str, device: torch.device = DEVICE):
    """Load a trained model from checkpoint."""
    from models.dae import DAE
    from models.sae import SAE
    from models.cae import CAE
    from models.vae import VAE
    from models.beta_vae import BetaVAE
    from models.advae import AdversarialVAE
    from models.cvae import CVAE
    from models.vqvae import VQVAE
    from models.others import IWAE, PAE, RobustDeepAutoEncoder
    
    model_classes = {
        'dae': DAE,
        'sae': SAE,
        'cae': CAE,
        'vae': VAE,
        'beta_vae': BetaVAE,
        'advae': AdversarialVAE,
        'cvae': CVAE,
        'vqvae': VQVAE,
        'iwae': IWAE,
        'pae': PAE,
        'rda': RobustDeepAutoEncoder,
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = MODEL_CONFIGS.get(model_name, {})
    
    # Create model instance
    model_class = model_classes[model_name]
    model = model_class(**config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Auto-Encoder models for Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['dae', 'sae', 'cae', 'vae', 'beta_vae', 'advae', 'cvae', 'vqvae', 'iwae', 'pae', 'rda'],
        help='Model to evaluate'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'fashion_mnist'],
        default='mnist',
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--normal_class',
        type=int,
        default=0,
        help='Normal class for anomaly detection (0-9)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all available models in checkpoints directory'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory containing model checkpoints'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--compare_paper',
        action='store_true',
        help='Compare results with paper'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("AUTO-ENCODER ANOMALY DETECTION - EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Normal class: {args.normal_class}")
    print(f"Device: {DEVICE}")
    
    # Load test data
    _, test_loader = get_anomaly_detection_loaders(
        dataset_name=args.dataset,
        normal_class=args.normal_class,
        batch_size=args.batch_size
    )
    
    results_dict = {}
    
    if args.all:
        # Evaluate all models
        model_names = ['dae', 'sae', 'cae', 'vae', 'beta_vae', 'advae', 'cvae', 'vqvae', 'iwae', 'pae', 'rda']
        
        for model_name in model_names:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}_{args.dataset}.pt')
            
            if not os.path.exists(checkpoint_path):
                print(f"\nSkipping {model_name}: checkpoint not found at {checkpoint_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name.upper()}")
            print('='*60)
            
            model = load_model(model_name, checkpoint_path)
            scores, labels = compute_reconstruction_error(model, test_loader)
            results = evaluate_anomaly_detection(scores, labels)
            results_dict[model_name] = results
            
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
            print(f"AP: {results['ap']:.4f}")
            print(f"F1: {results['f1']:.4f}")
            
            if args.visualize:
                plot_roc_curve(
                    results, model_name,
                    save_path=os.path.join(args.output_dir, f'roc_{model_name}_{args.dataset}.png')
                )
                plot_score_distribution(
                    scores, labels, model_name,
                    save_path=os.path.join(args.output_dir, f'scores_{model_name}_{args.dataset}.png')
                )
    
    elif args.model and args.checkpoint:
        # Evaluate single model
        print(f"\nEvaluating {args.model.upper()}")
        
        model = load_model(args.model, args.checkpoint)
        scores, labels = compute_reconstruction_error(model, test_loader)
        results = evaluate_anomaly_detection(scores, labels)
        results_dict[args.model] = results
        
        print(f"\nResults:")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  AP: {results['ap']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        
        if args.visualize:
            plot_roc_curve(
                results, args.model,
                save_path=os.path.join(args.output_dir, f'roc_{args.model}_{args.dataset}.png')
            )
            plot_score_distribution(
                scores, labels, args.model,
                save_path=os.path.join(args.output_dir, f'scores_{args.model}_{args.dataset}.png')
            )
    
    else:
        parser.print_help()
        return
    
    # Compare with paper
    if args.compare_paper and results_dict:
        compare_with_paper(results_dict, args.dataset)
    
    # Generate results table
    if results_dict:
        table = generate_results_table(
            results_dict, args.dataset,
            save_path=os.path.join(args.output_dir, f'results_{args.dataset}.txt')
        )
        print("\n" + table)
        
        # Save JSON results
        json_path = os.path.join(args.output_dir, f'results_{args.dataset}.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
