# src/VAE_GMM.py
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR
from lightning.pytorch.tuner import Tuner

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import argparse
import numpy as np

from src.dataset import DataModule
from config import (
    ModelConfig,
    TrainingConfig,
    TrainingSetup,
    HardwareConfig,
    DataConfig,
)
from typing import List, Tuple, Dict, Any, Optional



def lr_lambda(epoch, warmup_epochs = 35, linear_epochs=65):
    """"
    Calculates a linear learning rate annealing factor.
    Args:
        epoch (int): Current epoch number.
        warmup_epochs (int): Number of epochs for warmup phase.
        linear_epochs (int): Number of epochs for linear annealing phase.
    Returns:
        float: Annealing factor for the learning rate.
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < (warmup_epochs+linear_epochs):
        return float(epoch - warmup_epochs) / linear_epochs
    else:
        return 1.0



class Encoder(nn.Module):
    """
    Encoder-Network for the VAE.
    This network consists of several linear layers with Batch Normalization, Dropout, and ReLU activations.
    Layer settings are defined in the ModelConfig (Number of Neuron, Layers and Dropouts).
    The last layer outputs the mean and log variance for the latent space.

    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        input_size = np.prod(self.model_config.input_shape)
        layers = []
        if not isinstance(self.model_config.dropout_prob, tuple):
            dropout_probs = tuple([self.model_config.dropout_prob] * (len(self.model_config.layer_sizes) - 1))
        else:
            dropout_probs = self.model_config.dropout_prob

        for idx, size in enumerate(self.model_config.layer_sizes[:-1]):
            layers.extend([
                nn.Linear(input_size, size),
                nn.BatchNorm1d(size),
                nn.Dropout(p=dropout_probs[idx]),
                nn.ReLU()
            ])
            input_size = size

        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(self.model_config.layer_sizes[-2], self.model_config.layer_sizes[-1])
        self.log_var = nn.Linear(self.model_config.layer_sizes[-2], self.model_config.layer_sizes[-1])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class Decoder(nn.Module):
    """
    Decoder-Network for the VAE.
    This network consists of several linear layers with Batch Normalization, Dropout, and ReLU activations.
    Layer settings are defined in the ModelConfig (Number of Neuron, Layers and Dropouts).
    The last layer outputs the reconstructed input, which is reshaped to the original input shape.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config
        layers = []

        if not isinstance(self.model_config.dropout_prob, tuple):
            dropout_probs = tuple([self.model_config.dropout_prob] * (len(self.model_config.layer_sizes) - 1))
        else:
            dropout_probs = self.model_config.dropout_prob
        input_size = model_config.layer_sizes[-1]


        for idx, size in enumerate(reversed(self.model_config.layer_sizes[:-1])):
            layers.extend([
                nn.Linear(input_size, size),
                nn.BatchNorm1d(size),
                nn.Dropout(p=dropout_probs[idx]),
                nn.ReLU()
            ])
            input_size = size

        self.decoder = nn.Sequential(*layers)
        self.final_linear = nn.Linear(input_size, np.prod(self.model_config.input_shape))
        self.final_activation = nn.Tanh() ## remove final activation function! (currently optimized for this activation function)

    def forward(self, x):
        x = self.decoder(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x.view(x.size(0), *self.model_config.input_shape)

        

class VAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) with Gaussian Mixture Model (GMM) clustering.
    This module implements a VAE with an encoder and decoder network.
    The encoder outputs the mean and log variance for the latent space, which are used for reparameterization.
    The decoder reconstructs the input from the latent space.
    The model also includes GMM clustering with trainable cluster parameters (pi, mu_c, log_var_c).
    The loss function combines the reconstruction loss, global KLD, cluster KLD, categorical KLD, and variance regularization.
    The model supports annealing for the KLD weight and other hyperparameters.
    """
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, training_setup: TrainingSetup):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.training_setup = training_setup

        self.save_hyperparameters()
        # Modell-Parameter
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

        # Set dummy values for GMM training parameters - will be initialized later
        self.pi = nn.Parameter(torch.ones(self.model_config.num_clusters) / self.model_config.num_clusters, requires_grad=False)
        self.mu_c = nn.Parameter(torch.zeros(self.model_config.num_clusters, self.model_config.layer_sizes[-1]), requires_grad=False)
        self.log_var_c = nn.Parameter(torch.zeros(self.model_config.num_clusters, self.model_config.layer_sizes[-1]), requires_grad=False)

        self.clustering_params = [self.pi, self.mu_c, self.log_var_c]
        self.vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
       
        # Loss Function
        self.loss_func = getattr(nn, model_config.loss_func_name)()

        self.val_loss_hist = []
        self.automatic_optimization = False

    def initialize_cluster_parameters(self):
        """
        Initializes the clustering parameters (pi, mu_c, log_var_c) using K-Means++.
        This method collects samples from the encoder, applies K-Means++ to the latent space,
        and initializes the cluster parameters.
        """
        # Kodierte Daten aus dem Encoder holen
        samples = self.collect_samples(
            return_mu=True,
            return_x=False,
            return_timestamp=False, 
            dataloader= self.trainer.datamodule.all_data_dataloader())
        
        mu = samples["mu"]


        # Führe K-Means++ mit fester Clusteranzahl durch
        kmeans = KMeans(n_clusters=self.model_config.num_clusters, n_init=100, random_state=42, init='k-means++')
        kmeans.fit(mu)  # Trainiere K-Means auf den Latent-Space-Daten
        mu_c_init = kmeans.cluster_centers_  # K-Means Clusterzentren
        cluster_labels = kmeans.labels_       # Clusterzuweisungen

        # Speichere die z_sample und Clusterlabels im Log-Verzeichnis
        # Versuche das Log-Verzeichnis aus dem Logger zu beziehen, falls vorhanden.
        if hasattr(self, "logger") and hasattr(self.logger, "log_dir"):
            save_dir = self.logger.log_dir
        else:
            save_dir = "./logs"  # Fallback-Verzeichnis
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "cluster_labels.npy"), cluster_labels)
        print(f"K-Means-Initialisierung abgeschlossen! Cluster-Zentren erfolgreich gesetzt.")
        print(f"Clusterlabels wurden im Ordner {save_dir} gespeichert.")

        # Konvertiere K-Means-Zentren zu Torch-Parameter
        mu_c = torch.tensor(mu_c_init, device=self.device, dtype=torch.float32)

        # Leichte zufällige Störung für bessere Robustheit
        mu_c += torch.randn_like(mu_c) * 0.05  

        # Setze eine konstante Varianz für alle Cluster
        log_var_c = torch.ones_like(mu_c) * np.log(0.5)

        # Setze gleichmäßige Cluster-Prioren
        pi = torch.ones(self.model_config.num_clusters, device=self.device) / self.model_config.num_clusters

        # Setze die Parameter im Modell (nun trainierbar)
        self.mu_c = nn.Parameter(mu_c, requires_grad=True)
        self.log_var_c = nn.Parameter(log_var_c, requires_grad=True)
        self.pi = nn.Parameter(pi, requires_grad=True)

    

    def get_annealing_factor(self, current_epoch, start_epoch, duration, end_value):
        """
        Calculates the annealing factor with defined start and end boundaries.

        Args:
            current_epoch (int): Current epoch.
            start_epoch (int): Start point of the annealing.
            duration (int): Duration of the annealing.
            end_value (float): Target value at the end of the annealing.
            annealing_type (str): Type of the annealing, either "sigmoid" or "linear".

        Returns:
            float: The calculated annealing factor.
        """
        if not self.training_config.use_annealing:
            return end_value

        min_value = 1e-3 * end_value  # Start value of the annealing

        if current_epoch < start_epoch:
            return min_value  # Before the start, the value remains minimal

        # Calculate progress (between 0 and 1)
        progress = (current_epoch - start_epoch) / duration
        progress = np.clip(progress, 0.0, 1.0)  # Limit to [0, 1]

        if self.training_setup.annealing_type == "sigmoid":
            # Sigmoid annealing (smooth transition from min_value to end_value)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            return min_value + (sigmoid_progress * (end_value - min_value))

        elif self.training_setup.annealing_type == "linear":
            # Linear annealing (uniform increase from min_value to end_value)
            return min_value + (progress * (end_value - min_value))
        
        else:
            raise ValueError("annealing_type must be 'sigmoid' or 'linear'. ")

        

    def get_kld_weight(self):
        # Load the validation loss
        val_loss = self.trainer.callback_metrics.get('val/loss/recon')

        # Check if val_loss is valid (not None)
        if val_loss is not None:
            # Convert the tensor to a float (if necessary)
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()

            # Remove all None values from the history to safely use min()
            valid_losses = [loss for loss in self.val_loss_hist if loss is not None]

            # If there are valid values in the history, check if val_loss is smaller
            if valid_losses and min(valid_losses) >= val_loss:
                self.training_config.kld_weight *= 1.1

            self.training_config.kld_weight = min(self.training_config.kld_weight, self.training_config.vae_end_value)
            # Store the current loss in the history
            self.val_loss_hist.append(val_loss)


    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.
        Args:
            mu (Tensor): Mean of the latent space.
            log_var (Tensor): Log variance of the latent space.
        Returns:
            Tensor: Sampled latent variable z.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    def gaussian_mixture_log_prob(self, z):
        """        
        Computes the log probability of the latent variable z under a Gaussian Mixture Model (GMM).
        Args:
            z (Tensor): Latent variable (batch_size, latent_dim).
        Returns:
            Tuple[Tensor, Tensor]: Log probability of z under the GMM and soft assignments (gamma).
        """
        z = z.unsqueeze(1)  # [B, 1, D]
        mu_c = self.mu_c.unsqueeze(0)  # [1, K, D]
        log_var_c = self.log_var_c.unsqueeze(0)  # [1, K, D]
        pi = self.pi.unsqueeze(0)  # [1, K]

        log_p_c = torch.log(pi + 1e-10)
        log_p_z_c = -0.5 * (log_var_c + (z - mu_c).pow(2) / torch.exp(log_var_c)).sum(-1) \
                    - 0.5 * self.model_config.layer_sizes[-1] * torch.log(torch.tensor(2 * np.pi)) # layer_sizes[-1] is the latent dimension

        log_p_z = torch.logsumexp(log_p_c + log_p_z_c, dim=1)
        gamma = torch.exp(log_p_c + log_p_z_c - log_p_z.unsqueeze(1))

        return log_p_z, gamma.detach()

    def variance_regularization(self, log_var_c):
        """
        Regularizes the variance of the cluster parameters to ensure stability.
        Args:
            log_var_c (Tensor): Log variance of the cluster parameters (num_clusters, latent_dim)
        Returns:
            Tensor: Regularization term (scalar)
        """
        std_c = torch.exp(0.5 * log_var_c)
        max_std = std_c.max(dim=0)[0]
        min_std = std_c.min(dim=0)[0]
        return (max_std / (min_std + 1e-6)).mean()
    



    def cluster_balance_metric(self, gamma, threshold_factor=0.5):
        """
        Evaluates the cluster distribution and penalizes shrinking clusters.

        Args:
            gamma (Tensor): Soft Assignments (batch_size, num_clusters)
            threshold_factor (float): Threshold below which clusters are considered shrunk
            epsilon (float): Numerical stability

        Returns:
            Tensor: Score (higher is better)
        """
        # 1. Compute cluster sizes
        cluster_sizes = gamma.sum(0)  # Number of assignments per cluster
        avg_cluster_size = cluster_sizes.mean()  # Average cluster size

        # 2. Shrinkage Penalty: Penalizes clusters < threshold_factor * average
        shrinkage_penalty = torch.sum(torch.relu((threshold_factor * avg_cluster_size) - cluster_sizes))

        # 3. Balance Penalty: Penalizes imbalanced clusters (standard deviation)
        balance_penalty = torch.std(cluster_sizes)

        # 4. Combined Score (the smaller, the worse)
        score = - (balance_penalty + 5.0 * shrinkage_penalty)

        return score  # Higher score = better clustering

    
    def compute_silhouette(self, z, gamma):
        """
        Computes the silhouette score for the given latent variables and soft assignments.
        Args:
            z (Tensor): Latent variables (batch_size, latent_dim).
            gamma (Tensor): Soft assignments (batch_size, num_clusters). 
        Returns:
            float: Silhouette score for the clusters.           
        """
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()

        # Check number of unique labels
        unique_labels = len(np.unique(cluster_labels))

        # If less than 2 clusters, return a default value
        if unique_labels < 2:
            return torch.tensor(-1.0).to(self.device)  # or another meaningful default value

        return silhouette_score(z.cpu().numpy(), cluster_labels)


    def compute_loss_components(self, x, x_recon, mu, log_var, z):
        """
        Computes the individual loss components and returns them as a dictionary.
        Args:
            x (Tensor): Original input data.
            x_recon (Tensor): Reconstructed input data from the decoder.
            mu (Tensor): Mean of the latent space from the encoder.
            log_var (Tensor): Log variance of the latent space from the encoder.
            z (Tensor): Sampled latent variable from the reparameterization trick.
        Returns:
            Dict[str, Tensor]: Dictionary containing the individual loss components.
        """
        components = {}

        # Reconstruction loss (already weighted with the configured weight)
        components['recon'] = self.loss_func(x_recon, x) * self.training_config.recon_weight

        # Global KLD loss
        components['global_kld'] = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # If clustering part is active:
        if self.current_epoch >= self.training_setup.warmup_epochs:
            log_p_z, gamma = self.gaussian_mixture_log_prob(z)
            kl_loss_clusters = torch.zeros(1, device=self.device)
            for i in range(self.model_config.num_clusters):
                cluster_kl = 1 + log_var - (z - self.mu_c[i]).pow(2) - log_var.exp()
                cluster_kl = torch.sum(cluster_kl, dim=-1) * gamma[:, i]
                kl_loss_clusters -= 0.5 * cluster_kl.mean()
            components['cluster_kld'] = kl_loss_clusters

            # Categorical loss (e.g. for the cluster priors)
            cluster_props = gamma.mean(0)
            cat_kl = torch.sum(cluster_props * torch.log(cluster_props / (self.pi + 1e-10)))
            if self.training_config.log_scaled:
                cat_kl = torch.log(1 + cat_kl)
            components['cat_kld'] = cat_kl

            # Variance regularization of the cluster parameters
            components['var_reg'] = self.variance_regularization(self.log_var_c)

        return components

    def loss_function(self, x, x_recon, mu, log_var, z, prefix='train'):
        """
        Computes the overall loss as the sum of the individual components, weighted by
        dynamically computed factors.
        """
        # Compute individual loss components
        components = self.compute_loss_components(x, x_recon, mu, log_var, z)

        # Base epoch for the clustering part
        gmm_epoch = self.training_setup.vae_epochs + self.training_setup.adapt_epochs + self.training_setup.warmup_epochs

        # Compute the VAE factor (global KLD)
        if self.current_epoch < self.training_setup.vae_epochs + self.training_setup.warmup_epochs:
            vae_factor = self.training_config.kld_weight
        else:
            gmm_progress = min(
                1.0,
                (self.current_epoch - (self.training_setup.vae_epochs +
                                        self.training_setup.adapt_epochs +
                                        self.training_setup.warmup_epochs))
                / self.training_setup.gmm_epochs
            )
            reduction_factor = 0.7  # Example value: 0.7 for 30% reduction
            vae_factor = self.training_config.vae_end_value * (reduction_factor + (1 - reduction_factor) * (1 - gmm_progress))

        # Compute the other factors
        self.factors = {
            "vae_factor": vae_factor,
            "gmm_factor": self.get_annealing_factor(current_epoch=self.current_epoch,
                                                    start_epoch=gmm_epoch,
                                                    duration=self.training_setup.gmm_epochs,
                                                    end_value=self.training_config.gmm_end_value),
            "reg_factor": self.get_annealing_factor(current_epoch=self.current_epoch,
                                                    start_epoch=gmm_epoch,
                                                    duration=self.training_setup.reg_epochs,
                                                    end_value=self.training_config.reg_end_value),
            "cat_factor": self.get_annealing_factor(current_epoch=self.current_epoch,
                                                    start_epoch=gmm_epoch,
                                                    duration=self.training_setup.cat_epochs,
                                                    end_value=self.training_config.cat_end_value),
        }

        # Use the dynamic multiplier from the config (e.g. from training_config)
        dynamic_multiplier = self.training_config.dynamic_multiplier

        # Add the weighted loss components to a dictionary:
        losses = {}
        losses[f'{prefix}/loss/recon'] = components['recon']
        losses[f'{prefix}/loss/global_kld'] = components['global_kld'] * self.factors["vae_factor"]
        losses[f'{prefix}/loss/cluster_kld'] = components.get('cluster_kld', 0) * self.factors["gmm_factor"] * dynamic_multiplier
        losses[f'{prefix}/loss/cat_kld'] = components.get('cat_kld', 0) * self.factors["cat_factor"] * dynamic_multiplier
        losses[f'{prefix}/loss/var_reg'] = components.get('var_reg', 0) * self.factors["reg_factor"] * dynamic_multiplier

        # Combine the loss components into the total loss.
        # If the clustering part is active, sum all weighted components;
        # otherwise only the reconstruction and the global KLD.
        if self.current_epoch >= self.training_setup.warmup_epochs:
            total_loss = sum(value for key, value in losses.items() if not key.endswith('total'))
        else:
            total_loss = components['recon'] + self.factors["vae_factor"] * components["global_kld"]

        losses[f'{prefix}/loss/total'] = total_loss
        return losses


    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        Initializes the KLD weight and clustering parameters if necessary.
        """

        self.get_kld_weight()

        # if self.current_epoch == self.training_setup.warmup_epochs:
        #     print("Initializing clustering parameters with KMeans++...")
        #     self.initialize_cluster_parameters()

        if self.current_epoch == self.training_setup.kmeans_init_epoch:
            print("Initializing clustering parameters with KMeans++...")
            self.initialize_cluster_parameters()


    def training_step(self, batch, batch_idx):
        """
        Training step for the VAE-GMM model.
        Args:
            batch (Tuple[Tensor, Tensor]): A batch of data (input, target).
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: The total loss for the current batch.
        """
        vae_opt, clustering_opt = self.optimizers()
        x, _ = batch
        
        # Gradienten vor jedem Schritt zurücksetzen
        vae_opt.zero_grad()
        clustering_opt.zero_grad()
        
        x_recon, mu, log_var, z = self(x)
        losses = self.loss_function(x, x_recon, mu, log_var, z, prefix='train')
        
        if self.current_epoch < self.training_setup.warmup_epochs:
            # Nur VAE Training
            vae_loss = losses['train/loss/recon'] + losses['train/loss/global_kld']
            self.manual_backward(vae_loss)
            vae_opt.step()
        else:
            # Volles Training mit allen Losses
            self.manual_backward(losses['train/loss/total'])
            vae_opt.step()
            clustering_opt.step()
        
        self.log_dict(losses, batch_size=self.trainer.datamodule.batch_size,
                    on_step=False, on_epoch=True, sync_dist=True)
        
        return losses['train/loss/total']
    

    def compute_cluster_metrics(self, z, gamma):
        """
        Computes various metrics to evaluate cluster stability and quality:
        - Global Silhouette Score
        - Per-Cluster Silhouette Scores (as Dictionary)
        - Davies-Bouldin Index
        - Calinski-Harabasz Index
        - Cluster Balance
        - Local Density
        - Latent Smoothness, Density Variation, and Gaussian Similarity
        - Cluster Frequencies and Distribution Entropy
        - Variance of Individual Latent Dimensions
        """
        # Convert z to NumPy and determine hard cluster assignments
        z_np = z.cpu().numpy()
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()
        unique_labels = np.unique(cluster_labels)

        # Global Silhouette Score
        if len(unique_labels) < 2:
            global_sil = -1.0
        else:
            global_sil = silhouette_score(z_np, cluster_labels)
        
        # Per-Cluster Silhouette Scores
        if len(unique_labels) < 2:
            per_cluster_sil = {f"cluster_{i}": np.nan for i in range(self.model_config.num_clusters)}
        else:
            sample_sil = silhouette_samples(z_np, cluster_labels)
            per_cluster_sil = {}
            for i in range(self.model_config.num_clusters):
                idx = np.where(cluster_labels == i)[0]
                if len(idx) < 2:
                    per_cluster_sil[f"cluster_{i}"] = np.nan
                else:
                    per_cluster_sil[f"cluster_{i}"] = float(np.mean(sample_sil[idx]))

        # Davies-Bouldin und Calinski-Harabasz Index (only if at least 2 clusters exist)
        if len(unique_labels) < 2:
            db_index = np.nan
            ch_index = np.nan
        else:
            db_index = davies_bouldin_score(z_np, cluster_labels)
            ch_index = calinski_harabasz_score(z_np, cluster_labels)

        # Cluster Balance (as previously defined)
        balance = self.cluster_balance_metric(gamma)
        balance_val = balance.item() if isinstance(balance, torch.Tensor) else balance

        # Local Density in Latent Space
        local_density = self.compute_local_density(z)
        local_density_val = local_density.item() if isinstance(local_density, torch.Tensor) else local_density

        # Latent Smoothness and associated metrics
        smoothness, density_variation, gaussian_similarity = self.compute_latent_smoothness(z)
        smoothness_val = smoothness.item() if isinstance(smoothness, torch.Tensor) else smoothness
        density_variation_val = density_variation.item() if isinstance(density_variation, torch.Tensor) else density_variation
        gaussian_similarity_val = gaussian_similarity.item() if isinstance(gaussian_similarity, torch.Tensor) else gaussian_similarity

        # Cluster Frequencies and Entropy
        cluster_sizes = gamma.sum(dim=0).cpu().numpy()
        frequencies = cluster_sizes / np.sum(cluster_sizes)
        entropy = -np.sum(frequencies * np.log(frequencies + 1e-10))

        # Variance of Individual Latent Dimensions
        #latent_variances = np.var(z_np, axis=0)
        
        metrics = {
            'global_silhouette': global_sil,
            'per_cluster_silhouette': per_cluster_sil,  # Dictionary with per-cluster scores
            'davies_bouldin_index': db_index,
            'calinski_harabasz_index': ch_index,
            'balance': balance_val,
            'local_density': local_density_val,
            'smoothness': smoothness_val,
            'density_variation': density_variation_val,
            'gaussian_similarity': gaussian_similarity_val,
            'cluster_entropy': float(entropy),
            'cluster_frequencies': frequencies.tolist(),
        }
        return metrics

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE-GMM model.
        """
        x, _ = batch
        with torch.no_grad():
            x_recon, mu, log_var, z = self(x)
            # Compute the loss (including reconstruction, global KLD, etc.)
            losses = self.loss_function(x, x_recon, mu, log_var, z, prefix='val')

            # Compute metrics that can always be determined
            smoothness, density_variation, gaussian_similarity = self.compute_latent_smoothness(z)
            local_density = self.compute_local_density(z)
            losses['val/metric/smoothness'] = smoothness
            losses['val/metric/density_variation'] = density_variation
            losses['val/metric/gaussian_similarity'] = gaussian_similarity
            losses['val/metric/local_density'] = local_density
            
            if self.current_epoch >= self.training_setup.kmeans_init_epoch:
                # Once enough epochs have passed, also compute the cluster-related metrics:
                _, gamma = self.gaussian_mixture_log_prob(z)
                balance_score = self.cluster_balance_metric(gamma)
                silhouette = self.compute_silhouette(z, gamma)
                # Compute additional cluster metrics
                cluster_metrics = self.compute_cluster_metrics(z, gamma)
                
                losses['val/metric/balance'] = balance_score
                losses['val/metric/silhouette'] = silhouette
                # Add all keys from cluster_metrics
                for key, value in cluster_metrics.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            losses[f'val/metric/{key}/{subkey}'] = subvalue
                    elif isinstance(value, list):
                        for i, v in enumerate(value):
                            losses[f'cluster/{key}/{i}'] = v
                    else:
                        losses[f'val/metric/{key}'] = value
            else:
                # Default values if not enough epochs have passed yet:
                losses['val/metric/balance'] = -1.0
                losses['val/metric/silhouette'] = -1.0
                losses['val/metric/global_silhouette'] = -1.0
                losses['val/metric/davies_bouldin_index'] = 1000.0
                losses['val/metric/calinski_harabasz_index'] = 0.0
                losses['val/metric/cluster_entropy'] = -1.0
                # Per-Cluster Silhouette Scores all set to -1.0
                for i in range(self.model_config.num_clusters):
                    losses[f'cluster/per_cluster_silhouette/{i}'] = -1.0

            
            self.log_dict(losses, batch_size=self.trainer.datamodule.batch_size,
                        on_step=False, on_epoch=True, sync_dist=True)
            return losses['val/loss/total']
        

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Updates the learning rate schedulers and logs the annealing factors.
        """
        opt_vae, opt_cluster = self.optimizers()
        cluster_scheduler = self.lr_schedulers()[1]
        cluster_scheduler.step()

        # Update the dynamic multiplier if necessary:
        if isinstance(self.training_config.dynamic_update_epoch, (list, tuple)):
            if self.current_epoch in self.training_config.dynamic_update_epoch:
                self.training_config.dynamic_multiplier *= self.training_config.dynamic_reduction_factor
                print(f"Dynamic multiplier updated to {self.training_config.dynamic_multiplier} at epoch {self.current_epoch}")
        else:
            if self.current_epoch == self.training_config.dynamic_update_epoch:
                self.training_config.dynamic_multiplier *= self.training_config.dynamic_reduction_factor
                print(f"Dynamic multiplier updated to {self.training_config.dynamic_multiplier} at epoch {self.current_epoch}")


        self.log('learning_rate/vae', opt_vae.param_groups[0]["lr"], prog_bar=True)
        self.log('learning_rate/clustering', opt_cluster.param_groups[0]["lr"], prog_bar=True)
        # Log the annealing factors as metrics:
        self.log('annealing/vae_factor', self.factors["vae_factor"])
        self.log('annealing/gmm_factor', self.factors["gmm_factor"]*self.training_config.dynamic_multiplier)
        self.log('annealing/reg_factor', self.factors["reg_factor"]*self.training_config.dynamic_multiplier)
        self.log('annealing/cat_factor', self.factors["cat_factor"]*self.training_config.dynamic_multiplier)


    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        Updates the VAE scheduler and logs the TSNE visualization every 5 epochs.
        """
        annealing_epochs = self.training_setup.warmup_epochs + self.training_setup.vae_epochs

        current_loss = self.trainer.callback_metrics.get('val/loss/recon')
        if current_loss is not None:
            if self.current_epoch > annealing_epochs:
                # VAE Scheduler after the warmup epochs
                vae_scheduler = self.lr_schedulers()[0]  # The first scheduler is for VAE
                vae_scheduler.step(current_loss)


        # 4. TSNE Visualization every 5 epochs
        if self.training_config.log_img and self.current_epoch % 5 == 0:
            samples = self.collect_samples(return_mu=True, return_x=False, return_timestamp=False)
            mu = torch.tensor(samples["mu"])
            self.log_tsne(mu)
            

    def cluster_balance_metric(self, gamma, threshold_factor=0.5):
        """
        Evaluates the cluster distribution and penalizes shrinking clusters.

        Args:
            gamma (Tensor): Soft Assignments (batch_size, num_clusters)
            threshold_factor (float): Threshold below which clusters are considered shrunk
            epsilon (float): Numerical stability

        Returns:
            Tensor: Score (higher is better)
        """
        # 1. Compute cluster sizes
        cluster_sizes = gamma.sum(0)  # Number of assignments per cluster
        avg_cluster_size = cluster_sizes.mean()  # Average cluster size

        # 2. Shrinkage Penalty: Penalty for clusters < threshold_factor * average
        shrinkage_penalty = torch.sum(torch.relu((threshold_factor * avg_cluster_size) - cluster_sizes))

        # 3. Balance Penalty: Penalty for imbalanced clusters (standard deviation)
        balance_penalty = torch.std(cluster_sizes)

        # 4. Combined Score (the smaller, the worse)
        score = - (balance_penalty + 5.0 * shrinkage_penalty)

        return score  # Higher score = better clustering


    def compute_silhouette(self, z, gamma):
        """
        Computes the silhouette score for the given latent variables and soft assignments.
        Args:
            z (Tensor): Latent variables (batch_size, latent_dim).
            gamma (Tensor): Soft assignments (batch_size, num_clusters).
        Returns:
            float: Silhouette score for the clusters.
        """
        # Convert z to NumPy and determine hard cluster assignments
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()

        # Check number of unique labels
        unique_labels = len(np.unique(cluster_labels))
        
        # If less than 2 clusters, return a default value
        if unique_labels < 2:
            return torch.tensor(-1.0).to(self.device)  # or another meaningful default value
        return silhouette_score(z.cpu().numpy(), cluster_labels)

    def compute_local_density(self, z, k = 10):
        """
        Computes the local density for each sample in the latent space.
        A lower value indicates a more uniform distribution.
        Args:
            z (Tensor): Latent variables (batch_size, latent_dim).
            k (int): Number of nearest neighbors to consider for local density calculation.
        Returns:
            float: Local density for the latent space.
        """
        distances = torch.cdist(z, z)
        k_nearest = torch.topk(distances, k=k+1, largest=False)[0]
        k_nearest = k_nearest[:, 1:]  # Remove distance to self
        local_density = torch.std(k_nearest, dim=1).mean()
        return local_density
        

    def compute_latent_smoothness(self, z, k=10):
        """
        Computes the smoothness of the latent space using:
        - Local Density Variation
        - Global Distribution Metrics (Gaussian similarity)
        Args:
            z (Tensor): Latent variables (batch_size, latent_dim).
        Returns:
            Tuple[float, float, float]: Smoothness score, local density variation, and Gaussian similarity.
        """
        # Local Density Variation
        distances = torch.cdist(z, z)
        knn_distances, _ = torch.topk(distances, k, largest=False)
        density_variation = torch.std(knn_distances[:, 1:])  # Erste Distanz ist immer 0 (zu sich selbst)
        
        # Gaussian Similarity
        z_mean = torch.mean(z, dim=0)
        z_std = torch.std(z, dim=0)
        gaussian_similarity = -torch.mean(torch.abs(z_std - 1.0)) # Compares the std of z to 1.0, which is the ideal for a Gaussian distribution

        # Combined Metric: Higher values = better, smoother distribution
        smoothness = -density_variation + gaussian_similarity
        
        return smoothness, density_variation, gaussian_similarity

    def collect_samples(self, return_mu=True, return_x=True, return_timestamp=True, dataloader = None):
        """
        Collects samples from the encoder and returns them as a dictionary.
        Args:
            return_mu (bool): Whether to return the latent means.
            return_x (bool): Whether to return the original inputs.
            return_timestamp (bool): Whether to return the timestamps.
            dataloader (DataLoader, optional): Custom dataloader to use for sampling.
        Returns:
            Dict[str, np.ndarray]: Dictionary containing the collected samples.
        - "mu": Latent means (if return_mu is True)
        - "x": Original inputs (if return_x is True)
        - "timestamp": Timestamps (if return_timestamp is True)
        """
        self.encoder.eval() # Set the encoder to evaluation mode
        if dataloader is None:
            dataloader = self.trainer.datamodule.val_dataloader()

        mu_list, x_list, timestamp_list = [], [], []
        for batch in dataloader:
            # Unpack the batch
            x, info = batch
            if return_mu:
                with torch.no_grad():
                    mu, _ = self.encoder(x.to(self.device))
                mu_list.append(mu.cpu().numpy())
            if return_x:
                x_list.append(x.cpu().numpy())
            if return_timestamp:
                timestamp_list.append(np.array(info["timestamp"]))

        self.encoder.train()

        results = {}
        if return_mu:
            results["mu"] = np.concatenate(mu_list, axis=0)
        if return_x:
            results["x"] = np.concatenate(x_list, axis=0)
        if return_timestamp:
            results["timestamp"] = np.concatenate(timestamp_list, axis=0)

        return results


    def log_tsne(self, mu):
        """
        Logs a t-SNE visualization of the latent space.
        Args:
            mu (Tensor): Latent means (batch_size, latent_dim).
        """
        if not self.training_config.log_img:
            return
        # t-SNE Transformation
        mu = mu.to(self.device)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        mu_tsne = tsne.fit_transform(mu.cpu().numpy())

        # Cluster assignments
        _, gamma = self.gaussian_mixture_log_prob(mu)
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()

        # Compute the cluster centers in the t-SNE space
        tsne_centers = np.array([
            mu_tsne[cluster_labels == i].mean(axis=0) for i in np.unique(cluster_labels)
        ])

        # Colors for the clusters from the Seaborn palette
        unique_clusters = sorted(np.unique(cluster_labels))  # Unique, sorted cluster labels
        palette = sns.color_palette("Set1", len(unique_clusters))

        # Visualization
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=mu_tsne[:, 0], 
            y=mu_tsne[:, 1], 
            hue=cluster_labels,
            palette=palette,
            alpha=0.4,
            legend="full"
        )

        # Draw cluster centers (color according to the respective cluster)
        for cluster_id, center in enumerate(tsne_centers):
            plt.scatter(
                center[0], center[1],
                marker='X',
                color=palette[cluster_id],  # Color of the cluster
                s=200, 
                label=f'Cluster {unique_clusters[cluster_id]} Center'
            )

        # Legend and Logging
        plt.legend()
        # Check for logger availability
        if hasattr(self, "logger") and self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(f't-SNE_epoch_{self.current_epoch}', plt.gcf(), close=True)
        else:
            print(f"t-SNE for epoch {self.current_epoch} is not being logged (no logger available)")
        # Save the figure
        plt.close()


    def get_latent_variances(self):
        """Computes the variance of each dimension in the latent space after a forward pass."""
        samples = self.collect_samples(return_mu=True, return_x=False, return_timestamp=False)
        mu_all = samples["mu"]

        variances = np.var(mu_all, axis=0)  # Compute variance of each latent dimension
        return variances


    def on_fit_start(self):
        """ Called at the very beginning of fit, after checkpoint restore if any. """
        # Access the optimizers
        opt_vae, opt_cluster = self.optimizers()

        # Manually set the learning rate for the clustering optimizer
        for pg in opt_cluster.param_groups:
            pg["lr"] = self.training_config.clustering_lr
            pg["initial_lr"] = self.training_config.clustering_lr

        # Access the clustering scheduler
        cluster_scheduler = self.lr_schedulers()[1]
        cluster_scheduler.last_epoch = -1  # Reset scheduler




    def get_annealing_factor(self, current_epoch, start_epoch, duration, end_value):
        """
        Computes the annealing factor with defined start and end boundaries.

        Args:
            current_epoch (int): Current epoch.
            start_epoch (int): Start point of the annealing.
            duration (int): Duration of the annealing.
            end_value (float): Target value at the end of the annealing.
            annealing_type (str): Type of the annealing, either "sigmoid" or "linear".

        Returns:
            float: The computed annealing factor.
        """
        if not self.training_config.use_annealing:
            return end_value

        min_value = 1e-2 * end_value  # Start value of the annealing

        if current_epoch < start_epoch:
            return min_value  # Before the start, the value remains minimal

        # Compute progress (between 0 and 1)
        progress = (current_epoch - start_epoch) / duration
        progress = np.clip(progress, 0.0, 1.0)  # Clip to [0, 1]

        if self.training_setup.annealing_type == "sigmoid":
            # Sigmoid annealing (smooth transition from min_value to end_value)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            return min_value + (sigmoid_progress * (end_value - min_value))

        elif self.training_setup.annealing_type == "linear":
            # Linear annealing (uniform increase from min_value to end_value)
            return min_value + (progress * (end_value - min_value))
        
        else:
            raise ValueError("annealing_type must be 'sigmoid' or 'linear'")

        

    def get_kld_weight(self):
        """
        Dynamically adjusts the KLD weight based on the validation loss history.
        If the current validation loss is lower than the minimum in the history,
        the KLD weight is increased by 10%.
        """
        # Load the validation loss
        val_loss = self.trainer.callback_metrics.get('val/loss/recon')

        # Check if val_loss is valid (not None)
        if val_loss is not None:
            # Convert the tensor to a float (if necessary)
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()

            # Remove all None values from the history to safely use min()
            valid_losses = [loss for loss in self.val_loss_hist if loss is not None]

            # If there are valid values in the history, check if val_loss is smaller
            if valid_losses and min(valid_losses) >= val_loss:
                self.training_config.kld_weight *= 1.1

            self.training_config.kld_weight = min(self.training_config.kld_weight, self.training_config.vae_end_value)
            # Store the current loss in the history
            self.val_loss_hist.append(val_loss)

            # Log the current KLD weight value
            self.log('kld_weight', self.training_config.kld_weight)


    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the VAE-GMM model.
        Returns:
            Tuple[List[Optimizer], List[Dict[str, Any]]]: A tuple containing the optimizers and their respective schedulers.
        """
        # VAE parameters: encoder and decoder
        vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # Cluster parameters (e.g., pi, mu_c, log_var_c, or other dummy-based parameters)
        #cluster_params = [self.pi, self.mu_c, self.log_var_c]
        cluster_params = [self.mu_c, self.pi, self.log_var_c]

        # Optimizer for the two areas
        opt_vae = torch.optim.Adam(vae_params, lr=self.training_config.vae_lr)
        opt_cluster = torch.optim.AdamW(cluster_params, lr=self.training_config.clustering_lr)

        # Warmup configuration: In the first 25 epochs, the cluster LR should remain 0,
        # then linear increase over 'warmup_epochs' (here 90 epochs) to full LR.
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_cluster,
            lr_lambda=lambda epoch: lr_lambda(epoch, warmup_epochs=self.training_setup.clustering_warmup, linear_epochs=self.training_setup.linear_epochs)
        )

        # Afterwards: Cosine Annealing over the remaining epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_cluster,
            T_max=self.training_setup.cosine_T_max,  # Adjust this value to your training duration
            eta_min=self.training_setup.cosine_eta_min,
        )

        # Combine the two schedulers: First Warmup (until epoch 25 + warmup_epochs),
        # then Cosine Annealing.
        cluster_scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt_cluster,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.training_setup.clustering_warmup]
        )

        # VAE scheduler (e.g., a ReduceLROnPlateau)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_vae, 
            mode="min", 
            factor=self.training_setup.vae_lr_factor, 
            patience=self.training_setup.vae_lr_patience, 
            min_lr=1e-6
        )

        return (
            [opt_vae, opt_cluster],
            [
                {"scheduler": vae_scheduler, "interval": "epoch", "monitor": "val_loss", "name": "vae_scheduler"},
                {"scheduler": cluster_scheduler, "interval": "epoch", "name": "cluster_scheduler"},
            ],
        )








if __name__ == "__main__":
    #### Main Script for Training and Testing the VaDE Model ####
    # This script initializes the model, data module, and trainer,
    # and runs either a learning rate finder or a full training session.
    parser = argparse.ArgumentParser(description='Test Script for VaDE.')
    parser.add_argument('--find_lr', type=bool, default=False, help='Flag to find the optimal learning rate.')
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    default_model_config = ModelConfig()
    default_training_config = TrainingConfig()
    default_training_setup = TrainingSetup()
    default_data_config = DataConfig()
    default_hardware_config = HardwareConfig()

    data_module = DataModule(
        data_dir = default_data_config.data_dir,
        batch_size = default_training_config.batch_size,
        num_workers = default_data_config.num_workers
    )

    vade = VAE(
        default_model_config,
        default_training_config,
        default_training_setup,
    )   

    if args.find_lr:
        trainer = pl.Trainer(
            accelerator=default_hardware_config.accelerator,
            devices=default_hardware_config.devices,
            max_epochs=1,
            val_check_interval=1,
            enable_progress_bar=True,
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(vade, data_module, min_lr=1e-7, max_lr=1e-3, num_training=800)
        print(lr_finder.suggestion())

    else:

        trainer = pl.Trainer(
            accelerator= 'cpu',
            devices= 4,
            max_epochs=50,
            val_check_interval=1,
            enable_progress_bar=True,
            fast_dev_run=True,
        )

        trainer.fit(vade, data_module)