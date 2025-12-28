"""
Training utilities for Auto-Encoder models
Implements training loops, evaluation metrics (ROC-AUC, AP), and anomaly scoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm
import time
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DEVICE, MODEL_CONFIGS


class AnomalyTrainer:
    """
    Trainer class for auto-encoder based anomaly detection.
    Handles training, evaluation, and metric computation.
    """
    
    def __init__(self, model, model_name, device=DEVICE):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.config = MODEL_CONFIGS.get(model_name, {})
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'roc_auc': [],
            'ap': [],
        }
        
    def get_optimizer(self, lr=None):
        """Get optimizer based on model type"""
        lr = lr or self.config.get('learning_rate', 1e-3)
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
    
    def compute_reconstruction_error(self, x, x_recon):
        """
        Compute reconstruction error for anomaly scoring.
        Higher error indicates anomaly.
        """
        # Per-sample MSE
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_recon_flat = x_recon.view(batch_size, -1)
        mse = torch.mean((x_flat - x_recon_flat) ** 2, dim=1)
        return mse
    
    def train_epoch_standard(self, train_loader, optimizer):
        """Standard training epoch for DAE, SAE, CAE, RDA"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            if self.model_name == 'dae':
                loss, recon, _ = self.model(data)
            elif self.model_name == 'sae':
                loss, recon, _ = self.model(data)
            elif self.model_name == 'cae':
                loss, recon, _ = self.model(data)
            elif self.model_name == 'rda':
                loss, recon, _, _ = self.model(data)
            else:
                recon = self.model(data)
                loss = nn.MSELoss()(recon, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def train_epoch_vae(self, train_loader, optimizer):
        """Training epoch for VAE-based models (VAE, Beta-VAE, CVAE, IWAE, PAE)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass - handle different VAE types
            if self.model_name == 'cvae':
                outputs = self.model(data, labels)
            else:
                outputs = self.model(data)
            
            # Handle dict or tuple outputs
            if isinstance(outputs, dict):
                losses = self.model.loss_function(data, outputs)
                loss = losses['loss']
            else:
                # Tuple format (loss, recon, ...)
                loss = outputs[0]
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def train_epoch_advae(self, train_loader, optimizer_enc, optimizer_dec, optimizer_trans):
        """
        Training epoch for adVAE with adversarial training.
        Two-step training: (1) Train D with fixed E,T (2) Train E with fixed D,T
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Step 1: Train decoder and transformer
            optimizer_dec.zero_grad()
            optimizer_trans.zero_grad()
            
            loss_d, _, _, _ = self.model(data, train_mode='decoder')
            loss_d.backward()
            
            optimizer_dec.step()
            optimizer_trans.step()
            
            # Step 2: Train encoder
            optimizer_enc.zero_grad()
            
            loss_e, _, _, _ = self.model(data, train_mode='encoder')
            loss_e.backward()
            
            optimizer_enc.step()
            
            total_loss += (loss_d.item() + loss_e.item()) / 2
            num_batches += 1
            
        return total_loss / num_batches
    
    def train_epoch_vqvae(self, train_loader, optimizer):
        """Training epoch for VQ-VAE"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            
            loss, recon, vq_loss, perplexity = self.model(data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader=None, epochs=50, lr=1e-3, 
              verbose=True, save_path=None):
        """
        Main training loop.
        """
        optimizer = self.get_optimizer(lr)
        
        # Special handling for adVAE
        if self.model_name == 'advae':
            optimizer_enc = optim.Adam(self.model.encoder.parameters(), lr=lr)
            optimizer_dec = optim.Adam(self.model.decoder.parameters(), lr=lr)
            optimizer_trans = optim.Adam(self.model.transformer.parameters(), lr=lr)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            if self.model_name in ['dae', 'sae', 'cae', 'rda']:
                train_loss = self.train_epoch_standard(train_loader, optimizer)
            elif self.model_name in ['vae', 'beta_vae', 'cvae', 'iwae', 'pae']:
                train_loss = self.train_epoch_vae(train_loader, optimizer)
            elif self.model_name == 'advae':
                train_loss = self.train_epoch_advae(
                    train_loader, optimizer_enc, optimizer_dec, optimizer_trans
                )
            elif self.model_name == 'vqvae':
                train_loss = self.train_epoch_vqvae(train_loader, optimizer)
            else:
                train_loss = self.train_epoch_standard(train_loader, optimizer)
            
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate_loss(val_loader)
                self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - "
                      f"Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        if save_path:
            self.save_model(save_path)
            
        return total_time
    
    def evaluate_loss(self, data_loader):
        """Evaluate reconstruction loss on a dataset"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Get reconstruction based on model type
                if self.model_name == 'cvae':
                    _, recon, _, _ = self.model(data, labels)
                elif self.model_name in ['vae', 'beta_vae', 'iwae', 'pae']:
                    _, recon, _, _ = self.model(data)
                elif self.model_name == 'advae':
                    _, recon, _, _ = self.model(data)
                elif self.model_name == 'vqvae':
                    _, recon, _, _ = self.model(data)
                elif self.model_name in ['dae', 'sae', 'cae']:
                    _, recon, _ = self.model(data)
                elif self.model_name == 'rda':
                    _, recon, _, _ = self.model(data)
                else:
                    recon = self.model(data)
                
                loss = nn.MSELoss()(recon, data)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def compute_anomaly_scores(self, data_loader):
        """
        Compute anomaly scores for all samples in the dataset.
        Returns scores and labels.
        """
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Get reconstruction based on model type
                if self.model_name == 'cvae':
                    outputs = self.model(data, labels.to(self.device))
                else:
                    outputs = self.model(data)
                
                # Handle dict or tuple outputs
                if isinstance(outputs, dict):
                    recon = outputs.get('recon', outputs.get('reconstruction'))
                elif isinstance(outputs, tuple):
                    recon = outputs[1]  # (loss, recon, ...)
                else:
                    recon = outputs
                
                # Compute reconstruction error as anomaly score
                scores = self.compute_reconstruction_error(data, recon)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_scores), np.array(all_labels)
    
    def evaluate_anomaly_detection(self, test_loader, normal_class=0):
        """
        Evaluate anomaly detection performance.
        
        Args:
            test_loader: DataLoader with test data
            normal_class: The class label considered as normal (0-9)
            
        Returns:
            Dictionary with ROC-AUC, AP, and other metrics
        """
        scores, labels = self.compute_anomaly_scores(test_loader)
        
        # Convert to binary: normal_class -> 0 (normal), others -> 1 (anomaly)
        binary_labels = (labels != normal_class).astype(int)
        
        # ROC-AUC
        roc_auc = roc_auc_score(binary_labels, scores)
        
        # Average Precision
        ap = average_precision_score(binary_labels, scores)
        
        # ROC curve for plotting
        fpr, tpr, thresholds = roc_curve(binary_labels, scores)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(binary_labels, scores)
        
        # Find optimal threshold using F1 score
        f1_scores = []
        for threshold in np.percentile(scores, np.linspace(0, 100, 100)):
            pred_labels = (scores >= threshold).astype(int)
            f1 = f1_score(binary_labels, pred_labels, zero_division=0)
            f1_scores.append((threshold, f1))
        
        optimal_threshold = max(f1_scores, key=lambda x: x[1])[0]
        pred_labels = (scores >= optimal_threshold).astype(int)
        optimal_f1 = f1_score(binary_labels, pred_labels, zero_division=0)
        
        results = {
            'roc_auc': roc_auc,
            'ap': ap,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'precision': precision,
            'recall': recall,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            'scores': scores,
            'labels': binary_labels,
        }
        
        return results
    
    def get_latent_representations(self, data_loader):
        """
        Extract latent space representations for visualization.
        """
        self.model.eval()
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                
                # Get latent representation based on model type
                if self.model_name in ['vae', 'beta_vae', 'iwae', 'pae']:
                    mu, _ = self.model.encode(data)
                    latent = mu
                elif self.model_name == 'cvae':
                    mu, _ = self.model.encode(data, labels.to(self.device))
                    latent = mu
                elif self.model_name == 'advae':
                    mu, _ = self.model.encode(data)
                    latent = mu
                elif self.model_name == 'vqvae':
                    latent = self.model.encode(data)
                    # For VQ-VAE, flatten if needed
                    if latent.dim() > 2:
                        latent = latent.view(latent.size(0), -1)
                elif hasattr(self.model, 'encode'):
                    latent = self.model.encode(data)
                else:
                    # Fallback: use encoder directly
                    x_flat = data.view(data.size(0), -1)
                    latent = self.model.encoder(x_flat)
                
                all_latents.append(latent.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.vstack(all_latents), np.array(all_labels)
    
    def generate_samples(self, num_samples=100, labels=None):
        """
        Generate new samples from the model (for generative models).
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.model_name in ['vae', 'beta_vae', 'iwae', 'pae', 'advae']:
                # Sample from prior N(0,1)
                latent_dim = self.config.get('latent_dim', 2)
                z = torch.randn(num_samples, latent_dim).to(self.device)
                samples = self.model.decode(z)
            elif self.model_name == 'cvae':
                # Need labels for conditional generation
                if labels is None:
                    labels = torch.randint(0, 10, (num_samples,))
                labels = labels.to(self.device)
                latent_dim = self.config.get('latent_dim', 2)
                z = torch.randn(num_samples, latent_dim).to(self.device)
                samples = self.model.decode(z, labels)
            elif self.model_name == 'vqvae':
                # For VQ-VAE, sample from codebook
                samples = self.model.sample(num_samples)
            else:
                # For non-generative models, return None
                return None
                
        return samples.cpu()
    
    def reconstruct_samples(self, data_loader, num_samples=64):
        """
        Get original and reconstructed samples for visualization.
        """
        self.model.eval()
        originals = []
        reconstructions = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                
                # Get reconstruction
                if self.model_name == 'cvae':
                    _, recon, _, _ = self.model(data, labels.to(self.device))
                elif self.model_name in ['vae', 'beta_vae', 'iwae', 'pae']:
                    _, recon, _, _ = self.model(data)
                elif self.model_name == 'advae':
                    _, recon, _, _ = self.model(data)
                elif self.model_name == 'vqvae':
                    _, recon, _, _ = self.model(data)
                elif self.model_name in ['dae', 'sae', 'cae']:
                    _, recon, _ = self.model(data)
                elif self.model_name == 'rda':
                    _, recon, _, _ = self.model(data)
                else:
                    recon = self.model(data)
                
                originals.append(data.cpu())
                reconstructions.append(recon.cpu())
                labels_list.extend(labels.numpy())
                
                if len(labels_list) >= num_samples:
                    break
        
        originals = torch.cat(originals, dim=0)[:num_samples]
        reconstructions = torch.cat(reconstructions, dim=0)[:num_samples]
        labels = np.array(labels_list)[:num_samples]
        
        return originals, reconstructions, labels
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'config': self.config,
            'history': self.history,
        }, path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


def run_anomaly_detection_experiment(model, model_name, train_loader, test_loader,
                                     normal_class=0, epochs=50, lr=1e-3, verbose=True):
    """
    Run a complete anomaly detection experiment.
    
    Args:
        model: Auto-encoder model
        model_name: Name of the model (dae, vae, etc.)
        train_loader: Training data (normal class only)
        test_loader: Test data (all classes with labels)
        normal_class: Class considered as normal
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        Dictionary with training time and evaluation metrics
    """
    trainer = AnomalyTrainer(model, model_name)
    
    # Train the model
    training_time = trainer.train(
        train_loader, 
        epochs=epochs, 
        lr=lr, 
        verbose=verbose
    )
    
    # Evaluate anomaly detection
    results = trainer.evaluate_anomaly_detection(test_loader, normal_class)
    results['training_time'] = training_time
    results['trainer'] = trainer
    
    return results
