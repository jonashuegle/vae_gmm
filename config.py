from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch.nn as nn


@dataclass
class ModelConfig:
    """Model configuration for the Variational Autoencoder (VAE) and Gaussian Mixture Model (GMM) components.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of the input data.
        layer_sizes (Tuple[int, ...]): Sizes of the layers in the VAE.
        dropout_prob (Tuple[float, ...]): Dropout probabilities for each layer.
        num_clusters (int): Number of clusters for the GMM.
        loss_func_name (str): Name of the loss function to be used.
        cluster_init_algorithm (str): Algorithm for cluster initialization ('kmeans' or 'hierarchical').
        cluster_init_source (str): Source for cluster initialization ('latent_space', 'full_data', or 'pca').
        cluster_init_pca_components (int): Number of PCA components if source is 'pca'.
        cluster_init_save (bool): Whether to save the cluster initialization.
        cluster_init_seed (int): Seed for reproducible initialization.

    """
    input_shape: Tuple[int, ...] = (1, 61, 181)
    layer_sizes: Tuple[int, ...] = (4000, 2000, 800, 200, 100, 14)
    dropout_prob: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1, 0)
    num_clusters: int = 5
    loss_func_name: str = "MSELoss"


@dataclass
class TrainingConfig:
    """Training configuration for the VAE and GMM components.

    Attributes:
        batch_size (int): Batch size for training.
        vae_lr (float): Learning rate for the VAE.
        clustering_lr (float): Learning rate for the clustering.
        use_annealing (bool): Whether to use learning rate annealing.
        recon_weight (float): Weight for the reconstruction loss.
        kld_weight (float): Weight for the Kullback-Leibler divergence loss.
        gmm_end_value (float): End value for the GMM loss.
        reg_end_value (float): End value for the regularization loss.
        vae_end_value (float): End value for the VAE loss.
        cat_end_value (float): End value for the categorical loss.
        dynamic_multiplier (float): Multiplier for dynamic loss adjustment (used for the adaption phase in the end).
        dynamic_update_epoch (Tuple[int, ...]): Epochs at which the dynamic loss adjustment is updated.
        dynamic_reduction_factor (float): Reduction factor for the dynamic loss adjustment.
        seed (int): Random seed for reproducibility.
        log_img (bool): Whether to log images during training.
        log_scaled (bool): Whether to log scaled categorical loss values during training.
    """
    batch_size: int = 400
    vae_lr: float = 0.000193066
    clustering_lr: float = 5.929e-06
    use_annealing: bool = True
    recon_weight: float = 0.1
    kld_weight: float = 0.0001
    gmm_end_value: float = 0.005220209
    reg_end_value: float = 0.385072058
    vae_end_value: float = 0.001
    cat_end_value: float = 0.005362321
    log_img: bool = True
    log_scaled: bool = True
    dynamic_multiplier: float = 1.0
    dynamic_reduction_factor: float = 0.9        
    dynamic_update_epoch: Tuple[int, ...] = (300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320)
    seed: int = None

@dataclass
class TrainingSetup:
    """Training setup for the VAE and GMM components.
    
    Attributes:
        warmup_epochs (int): Number of warmup epochs.
        vae_epochs (int): Number of epochs for VAE training.
        adapt_epochs (int): Number of epochs for adaptation.
        clustering_warmup (int): Number of warmup epochs for clustering.
        gmm_epochs (int): Number of epochs for GMM training.
        cat_epochs (int): Number of epochs for categorical loss training.
        reg_epochs (int): Number of epochs for regularization training.
        kmeans_init_epoch (int): Epoch at which KMeans initialization is performed.
        annealing_type (str): Type of annealing ('linear' or 'cosine').
        vae_lr_factor (float): Learning rate factor for VAE.
        vae_lr_patience (int): Patience for VAE learning rate scheduler.
        linear_epochs (int): Number of epochs for linear learning rate scheduler.
        cosine_T_max (int): Maximum number of epochs for cosine annealing.
        cosine_eta_min (float): Minimum learning rate for cosine annealing.

    """
    warmup_epochs: int = 25
    vae_epochs: int = 25
    adapt_epochs: int = 15
    gmm_epochs: int = 80
    cat_epochs: int = 250
    reg_epochs: int = 250
    kmeans_init_epoch : int = 25
    annealing_type: str = 'linear'
    vae_lr_factor: float = 0.777187766
    vae_lr_patience: int = 30
    clustering_warmup: int = 25
    linear_epochs: int = 25
    cosine_T_max: int = 400
    cosine_eta_min: float = 1.2e-8

@dataclass
class DataConfig:
    """Data configuration for the VAE and GMM components.

    Attributes:
        data_dir (str): Directory containing the training data.
        nam_csv_path (str): Path to the NAM index CSV file.
        log_dir (str): Directory for logging.
        experiment (str): Name of the experiment for logging.
        num_workers (int): Number of workers for data loading.
    """
    data_dir: str = '/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc'
    log_dir: str = '/work/aa0238/a271125/logs/Correct_Normalization'
    experiment: str = 'Experiment_'
    num_workers: int = 64


@dataclass
class HardwareConfig:
    """Hardware configuration for the VAE and GMM components.

    Attributes:
        accelerator (str): Type of accelerator to use ('gpu', 'cpu', etc.).
        devices (Tuple[int, ...]): Tuple of device IDs to use for training. Set to (0,) for slurm single GPU training.
    """
    accelerator: str = "gpu"
    devices: Tuple[int, ...] = (0,)


# Example for initializing default configurations
# default_model_config = ModelConfig()
# default_training_config = TrainingConfig()
# default_data_config = DataConfig()
# default_hardware_config = HardwareConfig()
    