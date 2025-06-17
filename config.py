from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch.nn as nn


@dataclass
class ModelConfig:
    input_shape: Tuple[int, ...] = (1, 61, 181)
    layer_sizes: Tuple[int, ...] = (4000, 2000, 800, 200, 100, 14)
    dropout_prob: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1, 0)
    num_clusters: int = 5
    loss_func_name: str = "MSELoss"


@dataclass
class TrainingConfig:
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
    data_dir: str = '/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc'
    log_dir: str = "MULTI_VaDE_logs"
    experiment: str = "Master_thesis_results_VAE_GMM"
    num_workers: int = 128


@dataclass
class HardwareConfig:
    accelerator: str = "gpu"
    devices: Tuple[int, ...] = (0,)


# Standardkonfigurationen
default_model_config = ModelConfig()
default_training_config = TrainingConfig()
default_data_config = DataConfig()
default_hardware_config = HardwareConfig()
