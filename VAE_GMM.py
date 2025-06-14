import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR

from lightning.pytorch.tuner import Tuner

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import argparse

import numpy as np
import os
from dataset import DataModule
import config as config
from config import ModelConfig, TrainingConfig, TrainingSetup, HardwareConfig, DataConfig
from typing import List, Tuple, Dict, Any, Optional


def lr_lambda(epoch, warmup_epochs = 35, linear_epochs=65):
            # Beispiel: In den ersten warmup_epochs (z. B. 25) LR=0
            if epoch < warmup_epochs:
                return 0.0
            elif epoch < (warmup_epochs+linear_epochs):
                return float(epoch - warmup_epochs) / linear_epochs
            else:
                return 1.0



class Encoder(nn.Module):
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
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.decoder(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x.view(x.size(0), *self.model_config.input_shape)

        

class VAE(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, training_setup: TrainingSetup):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.training_setup = training_setup

        self.save_hyperparameters()
        # Modell-Parameter
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

        # Setze Dummy-Werte für die Clusterparameter
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
        """Initialisiert nur die Cluster-Zentren mit K-Means, ohne PCA oder weitere Transformationen.
        Speichert zusätzlich die z_sample und die zugehörigen Clusterlabels im Log-Verzeichnis.
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
        Berechnet den Annealing-Faktor mit definierter Start- und Endgrenze.

        Args:
            current_epoch (int): Aktuelle Epoche.
            start_epoch (int): Startpunkt des Annealings.
            duration (int): Dauer des Annealings.
            end_value (float): Zielwert am Ende des Annealings.
            annealing_type (str): Art des Annealings, entweder "sigmoid" oder "linear".

        Returns:
            float: Der berechnete Annealing-Faktor.
        """
        if not self.training_config.use_annealing:
            return end_value
        
        min_value = 1e-3 * end_value  # Startwert des Annealings

        if current_epoch < start_epoch:
            return min_value  # Vor dem Start bleibt der Wert minimal
        
        # Fortschritt berechnen (zwischen 0 und 1)
        progress = (current_epoch - start_epoch) / duration
        progress = np.clip(progress, 0.0, 1.0)  # Begrenzen auf [0, 1]

        if self.training_setup.annealing_type == "sigmoid":
            # Sigmoid-Annealing (sanfter Übergang)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            return min_value + (sigmoid_progress * (end_value - min_value))

        elif self.training_setup.annealing_type == "linear":
            # Lineares Annealing (gleichmäßige Erhöhung von min_value zu end_value)
            return min_value + (progress * (end_value - min_value))
        
        else:
            raise ValueError("annealing_type muss 'sigmoid' oder 'linear' sein")

        

    def get_kld_weight(self):
        # Lade den Validation Loss
        val_loss = self.trainer.callback_metrics.get('val/loss/recon')

        # Überprüfen, ob val_loss gültig ist (kein None)
        if val_loss is not None:
            # Konvertiere den Tensor zu einem float (falls erforderlich)
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()

            # Entferne alle None-Werte aus der Historie, um min() sicher verwenden zu können
            valid_losses = [loss for loss in self.val_loss_hist if loss is not None]

            # Wenn es gültige Werte in der Historie gibt, prüfe, ob val_loss kleiner ist
            if valid_losses and min(valid_losses) >= val_loss:
                self.training_config.kld_weight *= 1.1

            self.training_config.kld_weight = min(self.training_config.kld_weight, self.training_config.vae_end_value)
            # Speichere den aktuellen Loss in der Verlaufs-Historie
            self.val_loss_hist.append(val_loss)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    def gaussian_mixture_log_prob(self, z):
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
        std_c = torch.exp(0.5 * log_var_c)
        max_std = std_c.max(dim=0)[0]
        min_std = std_c.min(dim=0)[0]
        return (max_std / (min_std + 1e-6)).mean()
    



    def cluster_balance_metric(self, gamma, threshold_factor=0.5):
        """
        Bewertet die Clusterverteilung und bestraft schrumpfende Cluster.

        Args:
            gamma (Tensor): Soft Assignments (batch_size, num_clusters)
            threshold_factor (float): Schwelle, ab der Cluster als geschrumpft gelten
            epsilon (float): Numerische Stabilität

        Returns:
            Tensor: Score (höher ist besser)
        """
        # 1. Clustergrößen berechnen
        cluster_sizes = gamma.sum(0)  # Anzahl der Zuweisungen je Cluster
        avg_cluster_size = cluster_sizes.mean()  # Durchschnittliche Clustergröße

        # 2. Shrinkage Penalty: Strafe für Cluster < threshold_factor * Durchschnitt
        shrinkage_penalty = torch.sum(torch.relu((threshold_factor * avg_cluster_size) - cluster_sizes))

        # 3. Balance Penalty: Strafe für unausgewogene Cluster (Standardabweichung)
        balance_penalty = torch.std(cluster_sizes)

        # 4. Kombinierter Score (je kleiner, desto schlechter)
        score = - (balance_penalty + 5.0 * shrinkage_penalty)

        return score  # Höherer Score = besseres Clustering
    

    
    def compute_silhouette(self, z, gamma):
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()
        
        # Prüfe Anzahl der uniquen Labels
        unique_labels = len(np.unique(cluster_labels))
        
        # Wenn weniger als 2 Cluster, gebe einen Default-Wert zurück
        if unique_labels < 2:
            return torch.tensor(-1.0).to(self.device)  # oder einen anderen sinnvollen Default-Wert
            
        return silhouette_score(z.cpu().numpy(), cluster_labels)


    def compute_loss_components(self, x, x_recon, mu, log_var, z):
        """
        Berechnet die einzelnen Loss-Komponenten und gibt sie als Dictionary zurück.
        """
        components = {}

        # Rekonstruktionsloss (bereits mit dem konfigurierten Gewicht)
        components['recon'] = self.loss_func(x_recon, x) * self.training_config.recon_weight

        # Globaler KLD-Loss
        components['global_kld'] = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # Falls der Clustering-Teil aktiv ist:
        if self.current_epoch >= self.training_setup.warmup_epochs:
            log_p_z, gamma = self.gaussian_mixture_log_prob(z)
            kl_loss_clusters = torch.zeros(1, device=self.device)
            for i in range(self.model_config.num_clusters):
                cluster_kl = 1 + log_var - (z - self.mu_c[i]).pow(2) - log_var.exp()
                cluster_kl = torch.sum(cluster_kl, dim=-1) * gamma[:, i]
                kl_loss_clusters -= 0.5 * cluster_kl.mean()
            components['cluster_kld'] = kl_loss_clusters

            # Kategorischer Loss (z.B. für die Cluster-Prioren)
            cluster_props = gamma.mean(0)
            cat_kl = torch.sum(cluster_props * torch.log(cluster_props / (self.pi + 1e-10)))
            if self.training_config.log_scaled:
                cat_kl = torch.log(1 + cat_kl)
            components['cat_kld'] = cat_kl

            # Varianz-regularisierung der Cluster-Parameter
            components['var_reg'] = self.variance_regularization(self.log_var_c)

        return components

    def loss_function(self, x, x_recon, mu, log_var, z, prefix='train'):
        """
        Berechnet den Gesamtloss als Summe der einzelnen Komponenten, gewichtet mit
        dynamisch berechneten Faktoren.
        """
        # Berechne einzelne Loss-Komponenten
        components = self.compute_loss_components(x, x_recon, mu, log_var, z)

        # Basis-Epoche für den Clustering-Teil
        gmm_epoch = self.training_setup.vae_epochs + self.training_setup.adapt_epochs + self.training_setup.warmup_epochs

        # Berechne den VAE-Faktor (globaler KLD)
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
            reduction_factor = 0.7  # Beispielwert: 0.7 für 30% Reduktion
            vae_factor = self.training_config.vae_end_value * (reduction_factor + (1 - reduction_factor) * (1 - gmm_progress))

        # Berechne die weiteren Faktoren
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

        # Verwende den dynamischen Multiplikator aus der Config (z. B. aus training_config)
        dynamic_multiplier = self.training_config.dynamic_multiplier

        # Füge die gewichteten Loss-Komponenten in ein Dictionary ein:
        losses = {}
        losses[f'{prefix}/loss/recon'] = components['recon']
        losses[f'{prefix}/loss/global_kld'] = components['global_kld'] * self.factors["vae_factor"]
        losses[f'{prefix}/loss/cluster_kld'] = components.get('cluster_kld', 0) * self.factors["gmm_factor"] * dynamic_multiplier
        losses[f'{prefix}/loss/cat_kld'] = components.get('cat_kld', 0) * self.factors["cat_factor"] * dynamic_multiplier
        losses[f'{prefix}/loss/var_reg'] = components.get('var_reg', 0) * self.factors["reg_factor"] * dynamic_multiplier

        # Kombiniere die Loss-Komponenten zum Gesamtloss.
        # Wenn der Clustering-Teil aktiv ist, summiere alle gewichteten Komponenten;
        # ansonsten nur die Rekonstruktion und den globalen KLD.
        if self.current_epoch >= self.training_setup.warmup_epochs:
            total_loss = sum(value for key, value in losses.items() if not key.endswith('total'))
        else:
            total_loss = components['recon'] + self.factors["vae_factor"] * components["global_kld"]

        losses[f'{prefix}/loss/total'] = total_loss
        return losses


    def on_train_epoch_start(self):

        self.get_kld_weight()

        # if self.current_epoch == self.training_setup.warmup_epochs:
        #     print("Initializing clustering parameters with KMeans++...")
        #     self.initialize_cluster_parameters()

        if self.current_epoch == self.training_setup.kmeans_init_epoch:
            print("Initializing clustering parameters with KMeans++...")
            self.initialize_cluster_parameters()


    def training_step(self, batch, batch_idx):
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
        Berechnet verschiedene Metriken zur Bewertung der Cluster-Stabilität und -Qualität:
        - Globaler Silhouette Score
        - Per-Cluster Silhouette Scores (als Dictionary)
        - Davies-Bouldin Index
        - Calinski-Harabasz Index
        - Cluster Balance
        - Lokale Dichte
        - Latente Smoothness, Dichtevariation und Gaußsche Ähnlichkeit
        - Cluster-Frequenzen und Verteilungsentropie
        - Varianz der einzelnen Latent-Dimensionen
        """
        # Konvertiere z in NumPy und bestimme harte Clusterzuweisungen
        z_np = z.cpu().numpy()
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()
        unique_labels = np.unique(cluster_labels)
        
        # Globaler Silhouette Score
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
        
        # Davies-Bouldin und Calinski-Harabasz Index (nur wenn mindestens 2 Cluster existieren)
        if len(unique_labels) < 2:
            db_index = np.nan
            ch_index = np.nan
        else:
            db_index = davies_bouldin_score(z_np, cluster_labels)
            ch_index = calinski_harabasz_score(z_np, cluster_labels)
        
        # Cluster Balance (wie bereits definiert)
        balance = self.cluster_balance_metric(gamma)
        balance_val = balance.item() if isinstance(balance, torch.Tensor) else balance
        
        # Lokale Dichte im Latent Space
        local_density = self.compute_local_density(z)
        local_density_val = local_density.item() if isinstance(local_density, torch.Tensor) else local_density
        
        # Latente Smoothness und zugehörige Metriken
        smoothness, density_variation, gaussian_similarity = self.compute_latent_smoothness(z)
        smoothness_val = smoothness.item() if isinstance(smoothness, torch.Tensor) else smoothness
        density_variation_val = density_variation.item() if isinstance(density_variation, torch.Tensor) else density_variation
        gaussian_similarity_val = gaussian_similarity.item() if isinstance(gaussian_similarity, torch.Tensor) else gaussian_similarity
        
        # Cluster-Frequenzen und Entropie
        cluster_sizes = gamma.sum(dim=0).cpu().numpy()
        frequencies = cluster_sizes / np.sum(cluster_sizes)
        entropy = -np.sum(frequencies * np.log(frequencies + 1e-10))
        
        # Varianz der einzelnen Latent-Dimensionen
        #latent_variances = np.var(z_np, axis=0)
        
        metrics = {
            'global_silhouette': global_sil,
            'per_cluster_silhouette': per_cluster_sil,  # Dictionary mit Werten pro Cluster
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
        x, _ = batch
        with torch.no_grad():
            x_recon, mu, log_var, z = self(x)
            # Berechne den Loss (inklusive Rekonstruktion, globaler KLD etc.)
            losses = self.loss_function(x, x_recon, mu, log_var, z, prefix='val')
            
            # Berechne Metriken, die immer bestimmt werden können
            smoothness, density_variation, gaussian_similarity = self.compute_latent_smoothness(z)
            local_density = self.compute_local_density(z)
            losses['val/metric/smoothness'] = smoothness
            losses['val/metric/density_variation'] = density_variation
            losses['val/metric/gaussian_similarity'] = gaussian_similarity
            losses['val/metric/local_density'] = local_density
            
            if self.current_epoch >= self.training_setup.kmeans_init_epoch:
                # Sobald genügend Epochen vergangen sind, berechne auch die clusterbezogenen Metriken:
                _, gamma = self.gaussian_mixture_log_prob(z)
                balance_score = self.cluster_balance_metric(gamma)
                silhouette = self.compute_silhouette(z, gamma)
                # Berechne weitere Cluster-Metriken
                cluster_metrics = self.compute_cluster_metrics(z, gamma)
                
                losses['val/metric/balance'] = balance_score
                losses['val/metric/silhouette'] = silhouette
                # Füge alle Schlüssel aus cluster_metrics hinzu
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
                # Default-Werte setzen, wenn noch nicht genügend Epochen gelaufen sind:
                losses['val/metric/balance'] = -1.0
                losses['val/metric/silhouette'] = -1.0
                losses['val/metric/global_silhouette'] = -1.0
                losses['val/metric/davies_bouldin_index'] = 1000.0
                losses['val/metric/calinski_harabasz_index'] = 0.0
                losses['val/metric/cluster_entropy'] = -1.0
                # Per-Cluster Silhouette für alle Cluster als -1
                for i in range(self.model_config.num_clusters):
                    losses[f'cluster/per_cluster_silhouette/{i}'] = -1.0

            # Debug-Ausgabe: welche Keys werden geloggt?
            #print("Logged metric keys:", list(losses.keys()))
            
            self.log_dict(losses, batch_size=self.trainer.datamodule.batch_size,
                        on_step=False, on_epoch=True, sync_dist=True)
            return losses['val/loss/total']
        

    def on_train_epoch_end(self):
        opt_vae, opt_cluster = self.optimizers()
        cluster_scheduler = self.lr_schedulers()[1]
        cluster_scheduler.step()

        # Aktualisiere den dynamischen Multiplikator, falls nötig:
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
        # Logge die annealing Faktoren als Metriken:
        self.log('annealing/vae_factor', self.factors["vae_factor"])
        self.log('annealing/gmm_factor', self.factors["gmm_factor"]*self.training_config.dynamic_multiplier)
        self.log('annealing/reg_factor', self.factors["reg_factor"]*self.training_config.dynamic_multiplier)
        self.log('annealing/cat_factor', self.factors["cat_factor"]*self.training_config.dynamic_multiplier)


    def on_validation_epoch_end(self):
        annealing_epochs = self.training_setup.warmup_epochs + self.training_setup.vae_epochs

        current_loss = self.trainer.callback_metrics.get('val/loss/recon')
        if current_loss is not None:
            if self.current_epoch > annealing_epochs:
                # VAE Scheduler erst nach der Annealing-Phase aktivieren
                vae_scheduler = self.lr_schedulers()[0]  # Der erste Scheduler ist für VAE
                vae_scheduler.step(current_loss)


        # 4. TSNE Visualisierung alle 5 Epochen
        if self.training_config.log_img and self.current_epoch % 5 == 0:
            samples = self.collect_samples(return_mu=True, return_x=False, return_timestamp=False)
            mu = torch.tensor(samples["mu"])
            self.log_tsne(mu)
            

    def cluster_balance_metric(self, gamma, threshold_factor=0.5):
        """
        Bewertet die Clusterverteilung und bestraft schrumpfende Cluster.

        Args:
            gamma (Tensor): Soft Assignments (batch_size, num_clusters)
            threshold_factor (float): Schwelle, ab der Cluster als geschrumpft gelten
            epsilon (float): Numerische Stabilität

        Returns:
            Tensor: Score (höher ist besser)
        """
        # 1. Clustergrößen berechnen
        cluster_sizes = gamma.sum(0)  # Anzahl der Zuweisungen je Cluster
        avg_cluster_size = cluster_sizes.mean()  # Durchschnittliche Clustergröße

        # 2. Shrinkage Penalty: Strafe für Cluster < threshold_factor * Durchschnitt
        shrinkage_penalty = torch.sum(torch.relu((threshold_factor * avg_cluster_size) - cluster_sizes))

        # 3. Balance Penalty: Strafe für unausgewogene Cluster (Standardabweichung)
        balance_penalty = torch.std(cluster_sizes)

        # 4. Kombinierter Score (je kleiner, desto schlechter)
        score = - (balance_penalty + 5.0 * shrinkage_penalty)

        return score  # Höherer Score = besseres Clustering
    
    
    def compute_silhouette(self, z, gamma):
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()
        
        # Prüfe Anzahl der uniquen Labels
        unique_labels = len(np.unique(cluster_labels))
        
        # Wenn weniger als 2 Cluster, gebe einen Default-Wert zurück
        if unique_labels < 2:
            return torch.tensor(-1.0).to(self.device)  # oder einen anderen sinnvollen Default-Wert
            
        return silhouette_score(z.cpu().numpy(), cluster_labels)

    def compute_local_density(self, z):
        """
        Berechnet die lokale Dichte für jedes Sample im Latent Space.
        Ein niedriger Wert bedeutet eine gleichmäßigere Verteilung.
        """
        distances = torch.cdist(z, z)
        k = 10
        k_nearest = torch.topk(distances, k=k+1, largest=False)[0]
        k_nearest = k_nearest[:, 1:]  # Entferne Distanz zu sich selbst
        local_density = torch.std(k_nearest, dim=1).mean()
        return local_density
        

    def compute_latent_smoothness(self, z):
        # 1. Lokale Dichte-Variation
        distances = torch.cdist(z, z)
        k = 10  # k nächste Nachbarn
        knn_distances, _ = torch.topk(distances, k, largest=False)
        density_variation = torch.std(knn_distances[:, 1:])  # Erste Distanz ist immer 0 (zu sich selbst)
        
        # 2. Globale Verteilungs-Metrik
        z_mean = torch.mean(z, dim=0)
        z_std = torch.std(z, dim=0)
        gaussian_similarity = -torch.mean(torch.abs(z_std - 1.0))  # Nähe zur Standard-Normalverteilung
        
        # Kombinierte Metrik: Höhere Werte = bessere, glattere Verteilung
        smoothness = -density_variation + gaussian_similarity
        
        return smoothness, density_variation, gaussian_similarity

    def collect_samples(self, return_mu=True, return_x=True, return_timestamp=True, dataloader = None):
        """
        Sammelt für alle Batches im Validierungs-Dataloader:
        - z_sample: Die enkodierten Repräsentationen (mu) vom Encoder,
        - x_sample: Die Originaleingaben,
        - timestamp: Einen Zeitstempel (aus info, z. B. info["timestamp"]).

        Die Rückgabe erfolgt als Dictionary. Über die booleschen Flags
        kann gesteuert werden, welche Werte berechnet und zurückgegeben werden.
        """
        self.encoder.eval()  # Eval-Modus für konsistente Ausgaben
        if dataloader is None:
            dataloader = self.trainer.datamodule.val_dataloader()

        mu_list, x_list, timestamp_list = [], [], []
        for batch in dataloader:
            # Wir nehmen an, dass der Batch so aufgebaut ist: (x, info)
            x, info = batch
            if return_mu:
                with torch.no_grad():
                    mu, _ = self.encoder(x.to(self.device))
                mu_list.append(mu.cpu().numpy())
            if return_x:
                x_list.append(x.cpu().numpy())
            if return_timestamp:
                # Annahme: info enthält einen Schlüssel "timestamp", der für jedes Sample existiert.
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
        # t-SNE Transformation
        mu = mu.to(self.device)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        mu_tsne = tsne.fit_transform(mu.cpu().numpy())

        # Clusterzuweisungen
        _, gamma = self.gaussian_mixture_log_prob(mu)
        cluster_labels = torch.argmax(gamma, dim=1).cpu().numpy()

        # Berechne die Clusterzentren im t-SNE Raum
        tsne_centers = np.array([
            mu_tsne[cluster_labels == i].mean(axis=0) for i in np.unique(cluster_labels)
        ])

        # Farben für die Cluster aus der Seaborn-Palette
        unique_clusters = sorted(np.unique(cluster_labels))  # Eindeutige, sortierte Cluster-Labels
        palette = sns.color_palette("Set1", len(unique_clusters))   

        # Visualisierung
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=mu_tsne[:, 0], 
            y=mu_tsne[:, 1], 
            hue=cluster_labels,
            palette=palette,
            alpha=0.4,
            legend="full"
        )

        # Clusterzentren zeichnen (Farbe der entsprechenden Cluster)
        for cluster_id, center in enumerate(tsne_centers):
            plt.scatter(
                center[0], center[1],
                marker='X', 
                color=palette[cluster_id],  # Farbe des Clusters
                s=200, 
                label=f'Cluster {unique_clusters[cluster_id]} Center'
            )

        # Legende und Logging
        plt.legend()
        self.logger.experiment.add_figure(f't-SNE_epoch_{self.current_epoch}', plt.gcf(), close=True)
        plt.close()


    def get_latent_variances(self):
        """Berechnet die Varianz jeder Dimension im Latent Space nach einem Forward-Pass."""
        samples = self.collect_samples(return_mu=True, return_x=False, return_timestamp=False)
        mu_all = samples["mu"]

        variances = np.var(mu_all, axis=0)  # Varianz jeder Latent-Dimension berechnen
        return variances


    def on_fit_start(self):
        """ Called at the very beginning of fit, after checkpoint restore if any. """
        # Zugriff auf Optimizer
        opt_vae, opt_cluster = self.optimizers()

        # Manuelles Setzen der Lernrate für den Clustering-Optimizer
        for pg in opt_cluster.param_groups:
            pg["lr"] = self.training_config.clustering_lr
            pg["initial_lr"] = self.training_config.clustering_lr

        # Zugriff auf den Clustering-Scheduler
        cluster_scheduler = self.lr_schedulers()[1]  # Kein ["scheduler"] hier
        cluster_scheduler.last_epoch = -1  # Scheduler zurücksetzen




    def get_annealing_factor(self, current_epoch, start_epoch, duration, end_value):
        """
        Berechnet den Annealing-Faktor mit definierter Start- und Endgrenze.

        Args:
            current_epoch (int): Aktuelle Epoche.
            start_epoch (int): Startpunkt des Annealings.
            duration (int): Dauer des Annealings.
            end_value (float): Zielwert am Ende des Annealings.
            annealing_type (str): Art des Annealings, entweder "sigmoid" oder "linear".

        Returns:
            float: Der berechnete Annealing-Faktor.
        """
        if not self.training_config.use_annealing:
            return end_value
        
        min_value = 1e-2 * end_value  # Startwert des Annealings

        if current_epoch < start_epoch:
            return min_value  # Vor dem Start bleibt der Wert minimal
        
        # Fortschritt berechnen (zwischen 0 und 1)
        progress = (current_epoch - start_epoch) / duration
        progress = np.clip(progress, 0.0, 1.0)  # Begrenzen auf [0, 1]

        if self.training_setup.annealing_type == "sigmoid":
            # Sigmoid-Annealing (sanfter Übergang)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            return min_value + (sigmoid_progress * (end_value - min_value))

        elif self.training_setup.annealing_type == "linear":
            # Lineares Annealing (gleichmäßige Erhöhung von min_value zu end_value)
            return min_value + (progress * (end_value - min_value))
        
        else:
            raise ValueError("annealing_type muss 'sigmoid' oder 'linear' sein")

        

    def get_kld_weight(self):
        # Lade den Validation Loss
        val_loss = self.trainer.callback_metrics.get('val/loss/recon')

        # Überprüfen, ob val_loss gültig ist (kein None)
        if val_loss is not None:
            # Konvertiere den Tensor zu einem float (falls erforderlich)
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()

            # Entferne alle None-Werte aus der Historie, um min() sicher verwenden zu können
            valid_losses = [loss for loss in self.val_loss_hist if loss is not None]

            # Wenn es gültige Werte in der Historie gibt, prüfe, ob val_loss kleiner ist
            if valid_losses and min(valid_losses) >= val_loss:
                self.training_config.kld_weight *= 1.1

            self.training_config.kld_weight = min(self.training_config.kld_weight, self.training_config.vae_end_value)
            # Speichere den aktuellen Loss in der Verlaufs-Historie
            self.val_loss_hist.append(val_loss)

            # Logge den aktuellen KLD-Gewichtswert
            self.log('kld_weight', self.training_config.kld_weight)


    def configure_optimizers(self):
        # VAE-Parameter: Encoder und Decoder
        vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # Cluster-Parameter (zum Beispiel: pi, mu_c, log_var_c oder weitere Dummy-basierte Parameter)
        #cluster_params = [self.pi, self.mu_c, self.log_var_c]
        cluster_params = [self.mu_c, self.pi, self.log_var_c]

        # Optimizer für die beiden Bereiche
        opt_vae = torch.optim.Adam(vae_params, lr=self.training_config.vae_lr)
        opt_cluster = torch.optim.AdamW(cluster_params, lr=self.training_config.clustering_lr)

        # Warmup-Konfiguration: In den ersten 25 Epochen soll der Cluster-LR 0 bleiben,
        # danach lineare Steigerung über 'warmup_epochs' (hier 90 Epochen) bis zur vollen LR.
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_cluster,
            lr_lambda=lambda epoch: lr_lambda(epoch, warmup_epochs=self.training_setup.clustering_warmup, linear_epochs=self.training_setup.linear_epochs)
        )

        # Danach: Cosine Annealing über die restlichen Epochen
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_cluster,
            T_max=self.training_setup.cosine_T_max,  # Passe diesen Wert an deine Trainingsdauer an
            eta_min=self.training_setup.cosine_eta_min,
        )

        # Kombiniere die beiden Scheduler: Zuerst Warmup (bis Epoche 25 + warmup_epochs),
        # danach Cosine Annealing.
        cluster_scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt_cluster,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.training_setup.clustering_warmup]
        )

        # VAE-Scheduler (zum Beispiel ein ReduceLROnPlateau)
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
    parser = argparse.ArgumentParser(description='Test Script for VaDE.')
    parser.add_argument('--find_lr', type=bool, default=False, help='Flag to find the optimal learning rate.')
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    default_model_config = ModelConfig()
    default_training_config = TrainingConfig()
    defaul_training_setup = TrainingSetup()
    default_data_config = DataConfig()
    default_hardware_config = HardwareConfig()

    data_module = DataModule(
        data_dir = default_data_config.data_dir,
        batch_size = default_training_config.batch_size,
        num_workers = default_data_config.num_workers
    )

    vade = VaDE(
        default_model_config,
        default_training_config,
        defaul_training_setup,
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