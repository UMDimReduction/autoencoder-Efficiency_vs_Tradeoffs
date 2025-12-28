"""
Visualization utilities for Auto-Encoder Anomaly Detection Study
Generates figures matching the paper: ROC curves (Figs 17-18), latent space (Figs 13-14),
2D manifolds (Figs 15-16), and reconstruction examples (Figs 2-12).
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.manifold import TSNE
import os


def setup_plotting_style():
    """Set up matplotlib style for paper-quality figures"""
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150


def plot_roc_curve(fpr, tpr, auc_score, model_name, save_path=None):
    """
    Plot ROC curve for a single model (matching Figs 17-18 style).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        model_name: Name of the model
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.2f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(model_name)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_all_roc_curves(results_dict, dataset_name, save_path=None):
    """
    Plot ROC curves for all models in a grid (matching Figs 17-18 layout).
    
    Args:
        results_dict: Dictionary mapping model_name -> evaluation results
        dataset_name: 'MNIST' or 'FMNIST'
        save_path: Optional path to save figure
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    
    # Calculate grid dimensions
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_models > 2 else [axes] if n_models == 1 else axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        results = results_dict[model_name]
        
        fpr = results['fpr']
        tpr = results['tpr']
        auc_score = results['roc_auc']
        
        ax.plot(fpr, tpr, 'b-', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(model_name)
        
        # Add AUC annotation
        ax.text(0.95, 0.05, f'AUC = {auc_score:.2f}', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'ROC-AUC ({dataset_name})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_latent_space(latents, labels, model_name, n_classes=10, save_path=None):
    """
    Plot 2D latent space representation (matching Figs 13-14 style).
    
    Args:
        latents: (N, 2) array of latent representations
        labels: (N,) array of class labels
        model_name: Name of the model
        n_classes: Number of classes
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color map for digits
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        mask = labels == i
        ax.scatter(latents[mask, 0], latents[mask, 1], 
                   c=[colors[i]], label=str(i), alpha=0.6, s=10)
    
    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    ax.set_title(model_name)
    
    # Normalize axis ranges to [0, 1] as in paper
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # Legend on the side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              title='label', markerscale=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_all_latent_spaces(latent_dict, labels_dict, dataset_name, save_path=None):
    """
    Plot latent spaces for all models in a grid (matching Figs 13-14 layout).
    
    Args:
        latent_dict: Dictionary mapping model_name -> latent representations
        labels_dict: Dictionary mapping model_name -> labels
        dataset_name: 'MNIST' or 'FMNIST'
        save_path: Optional path to save figure
    """
    model_names = list(latent_dict.keys())
    n_models = len(model_names)
    
    # Calculate grid dimensions (2 columns as in paper)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_models == 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        latents = latent_dict[model_name]
        labels = labels_dict[model_name]
        
        # Normalize to [0, 1] range for visualization
        latents_norm = (latents - latents.min(axis=0)) / (latents.max(axis=0) - latents.min(axis=0) + 1e-8)
        
        for i in range(10):
            mask = labels == i
            ax.scatter(latents_norm[mask, 0], latents_norm[mask, 1],
                       c=[colors[i]], label=str(i), alpha=0.6, s=5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(model_name)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  title='label', markerscale=2, fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Latent Space Representation ({dataset_name})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_2d_manifold(decoder, latent_dim=2, n_points=20, digit_size=28, 
                     model_name='', device='cpu', save_path=None):
    """
    Plot 2D manifold of generated samples (matching Figs 15-16 style).
    
    Args:
        decoder: Decoder function that takes z and returns image
        latent_dim: Dimension of latent space
        n_points: Number of points per axis
        digit_size: Size of generated images
        model_name: Name of the model
        device: Device for computation
        save_path: Optional path to save figure
    """
    if latent_dim != 2:
        print(f"Warning: Manifold visualization requires 2D latent space, got {latent_dim}D")
        return None
    
    # Create grid of latent points
    z1 = np.linspace(0.0, 1.0, n_points)
    z2 = np.linspace(1.0, 0.0, n_points)  # Reversed for proper orientation
    
    # Generate images for each point
    figure = np.zeros((digit_size * n_points, digit_size * n_points))
    
    with torch.no_grad():
        for i, z2_val in enumerate(z2):
            for j, z1_val in enumerate(z1):
                z = torch.tensor([[z1_val, z2_val]], dtype=torch.float32).to(device)
                
                # Decode to image
                x_decoded = decoder(z)
                digit = x_decoded.cpu().numpy().reshape(digit_size, digit_size)
                
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.imshow(figure, cmap='gray')
    ax.set_title(model_name)
    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    
    # Set tick labels to show latent values
    ax.set_xticks(np.linspace(0, digit_size * n_points, 6))
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.linspace(0, digit_size * n_points, 6))
    ax.set_yticklabels(['1.0', '0.8', '0.6', '0.4', '0.2', '0.0'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_reconstruction_grid(originals, reconstructions, n_rows=8, n_cols=8,
                             title='Reconstruction', save_path=None):
    """
    Plot original and reconstructed images side by side (matching Figs 2-5 style).
    
    Args:
        originals: Original images tensor
        reconstructions: Reconstructed images tensor
        n_rows: Number of rows
        n_cols: Number of columns (will show originals and recons)
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 2, n_rows))
    
    n_samples = min(n_rows * n_cols, len(originals))
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= n_samples:
                break
            
            # Original
            ax_orig = axes[i, j * 2]
            img_orig = originals[idx].squeeze().numpy()
            ax_orig.imshow(img_orig, cmap='gray')
            ax_orig.axis('off')
            
            # Reconstruction
            ax_recon = axes[i, j * 2 + 1]
            img_recon = reconstructions[idx].squeeze().numpy()
            ax_recon.imshow(img_recon, cmap='gray')
            ax_recon.axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_noisy_reconstruction(noisy_images, reconstructions, noise_factor=0.27,
                              n_rows=8, n_cols=7, title='DAE Reconstruction',
                              save_path=None):
    """
    Plot noisy inputs and reconstructions for DAE (matching Fig 2 style).
    
    Args:
        noisy_images: Noisy input images
        reconstructions: Reconstructed images
        noise_factor: Noise factor used
        n_rows: Number of rows
        n_cols: Number of columns
        title: Plot title
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(n_rows, n_cols * 2 + 1, width_ratios=[1]*n_cols + [0.5] + [1]*n_cols)
    
    n_samples = min(n_rows * n_cols, len(noisy_images))
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= n_samples:
                break
            
            # Noisy input (left)
            ax_noisy = fig.add_subplot(gs[i, j])
            img_noisy = noisy_images[idx].squeeze().numpy()
            ax_noisy.imshow(img_noisy, cmap='gray')
            ax_noisy.axis('off')
            
            # Reconstruction (right)
            ax_recon = fig.add_subplot(gs[i, n_cols + 1 + j])
            img_recon = reconstructions[idx].squeeze().numpy()
            ax_recon.imshow(img_recon, cmap='gray')
            ax_recon.axis('off')
    
    # Add labels
    fig.text(0.25, 0.02, f'(a) {noise_factor*100:.0f}% Noisy Input', ha='center', fontsize=12)
    fig.text(0.75, 0.02, '(b) Reconstruction', ha='center', fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_generated_samples(samples, n_rows=10, n_cols=10, title='Generated Samples',
                           save_path=None):
    """
    Plot generated samples in a grid (matching Figs 6-8 style).
    
    Args:
        samples: Generated samples tensor
        n_rows: Number of rows
        n_cols: Number of columns
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    
    n_samples = min(n_rows * n_cols, len(samples))
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            
            if idx < n_samples:
                img = samples[idx].squeeze().numpy()
                ax.imshow(img, cmap='gray')
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_sparse_activation(activations, sparsity_level=0.45, save_path=None):
    """
    Visualize sparse activations for SAE (matching Fig 9 style).
    
    Args:
        activations: Activation matrix
        sparsity_level: Target sparsity level
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(activations, cmap='viridis', aspect='auto')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Sample Index')
    ax.set_title(f'Sparse Activations (sparsity={sparsity_level*100:.0f}%)')
    
    plt.colorbar(im, ax=ax, label='Activation Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_training_curves(history, model_name, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and optionally 'val_loss'
        model_name: Name of the model
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_anomaly_score_distribution(scores, labels, model_name, save_path=None):
    """
    Plot distribution of anomaly scores for normal vs anomalous samples.
    
    Args:
        scores: Anomaly scores
        labels: Binary labels (0=normal, 1=anomaly)
        model_name: Name of the model
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', color='red', density=True)
    
    ax.set_xlabel('Anomaly Score (Reconstruction Error)')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name} - Anomaly Score Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_results_comparison_table(results_dict, dataset_name, expected_results=None,
                                  save_path=None):
    """
    Create a comparison table of ROC-AUC results (matching Table 3 style).
    
    Args:
        results_dict: Dictionary mapping model_name -> evaluation results
        dataset_name: 'MNIST' or 'FMNIST'
        expected_results: Dictionary of expected results from paper
        save_path: Optional path to save figure
    """
    model_names = list(results_dict.keys())
    roc_aucs = [results_dict[m]['roc_auc'] for m in model_names]
    
    fig, ax = plt.subplots(figsize=(12, len(model_names) * 0.5 + 2))
    ax.axis('off')
    
    # Create table data
    if expected_results:
        col_labels = ['Model', 'ROC-AUC (Ours)', 'ROC-AUC (Paper)', 'Difference']
        table_data = []
        for name in model_names:
            ours = results_dict[name]['roc_auc']
            paper = expected_results.get(name, 'N/A')
            if isinstance(paper, (int, float)):
                diff = ours - paper
                table_data.append([name, f'{ours:.2f}', f'{paper:.2f}', f'{diff:+.2f}'])
            else:
                table_data.append([name, f'{ours:.2f}', paper, 'N/A'])
    else:
        col_labels = ['Model', 'ROC-AUC']
        table_data = [[name, f'{results_dict[name]["roc_auc"]:.2f}'] for name in model_names]
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title(f'Results Comparison - {dataset_name}', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def create_paper_figure_grid(trainers_dict, test_loader, dataset_name, 
                             output_dir, normal_class=0):
    """
    Generate all paper-style figures for a complete experiment.
    
    Args:
        trainers_dict: Dictionary mapping model_name -> AnomalyTrainer
        test_loader: Test data loader
        dataset_name: 'MNIST' or 'Fashion-MNIST'
        output_dir: Directory to save figures
        normal_class: Normal class for anomaly detection
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results
    results_dict = {}
    latent_dict = {}
    labels_dict = {}
    
    for model_name, trainer in trainers_dict.items():
        print(f"Processing {model_name}...")
        
        # Get evaluation results
        results = trainer.evaluate_anomaly_detection(test_loader, normal_class)
        results_dict[model_name] = results
        
        # Get latent representations (if 2D)
        try:
            latents, labels = trainer.get_latent_representations(test_loader)
            if latents.shape[1] == 2:
                latent_dict[model_name] = latents
                labels_dict[model_name] = labels
        except Exception as e:
            print(f"  Could not get latent representations for {model_name}: {e}")
        
        # Plot individual ROC curve
        plot_roc_curve(
            results['fpr'], results['tpr'], results['roc_auc'],
            model_name,
            save_path=os.path.join(output_dir, f'roc_{model_name}.png')
        )
        
        # Plot anomaly score distribution
        plot_anomaly_score_distribution(
            results['scores'], results['labels'], model_name,
            save_path=os.path.join(output_dir, f'scores_{model_name}.png')
        )
    
    # Plot combined ROC curves
    plot_all_roc_curves(
        results_dict, dataset_name,
        save_path=os.path.join(output_dir, f'roc_all_{dataset_name}.png')
    )
    
    # Plot combined latent spaces (if any)
    if latent_dict:
        plot_all_latent_spaces(
            latent_dict, labels_dict, dataset_name,
            save_path=os.path.join(output_dir, f'latent_all_{dataset_name}.png')
        )
    
    print(f"Figures saved to {output_dir}")
    return results_dict
