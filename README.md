VAE-GMM — Variational Autoencoder with Gaussian Mixture Clustering

Overview
- Train a Variational Autoencoder (VAE) with a Gaussian Mixture Model (GMM) on atmospheric SLP fields or similar data for regime clustering.
- Implemented with PyTorch Lightning: versioned logging and checkpoints per run.
- Includes single-run training, SLURM jobs, Ray Tune parameter scan, and rich analysis/visualization utilities.

Requirements
- Conda env (e.g. environment.yml)
- NetCDF dataset (default SLP) and GPU recommended.

Data & Logs
- Dataset path: set in `config.py` via `DataConfig.data_dir` (default: `/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc`).
- Logs & checkpoints: `DataConfig.log_dir` / `DataConfig.experiment` / `version_X/`.

Key Files
- `training.py`: entry point, versioning, checkpoints, callbacks.
- `VAE_GMM.py`: model (encoder/decoder, GMM, losses, schedulers, metrics, t-SNE logging).
- `dataset.py`: `CustomDataset` + Lightning `DataModule` (xarray NetCDF, normalization, splits).
- `config.py`: central config (ModelConfig, TrainingConfig, TrainingSetup, DataConfig, HardwareConfig).
- `loaders.py`: `ClusteringLoader` to load a trained model, extract labels, auto/manual mapping, plotting.
- `plotting.py`: Basemap plots, isolines, custom colormaps.
- `visualize_tsne.py`: t-SNE visualization (Matplotlib/Plotly).
- `pattern_reference_manager.py`: PCA+KMeans reference patterns, caching, manual naming (mapping).
- `pattern_taylor.py`: pattern matching (Hungarian) and Taylor diagrams.
- `tb_utils.py`: TensorBoard scalar loading and diagnostics figure.
- SLURM: `job.slurm`, `multijob.slurm`, `multijob_scan.slurm`, `parameter_scan.slurm`.
- `parameter_scan.py`: Ray Tune multi-objective scan.
- `check_ray_progress.py`: quick progress check for Ray experiments.

Quick Start (local)
1) Review and adjust `config.py` (see next section).
2) Activate your env.
3) Examples:
	 - New run (auto versioning):
		 python3 training.py --max_epochs 400 --seed 42
	 - Resume latest version (if checkpoints exist):
		 python3 training.py --resume True --seed 42
	 - Fixed version:
		 python3 training.py --version 3 --max_epochs 300 --seed 42

Note: `TrainingConfig.seed` must not be None. Set in `config.py` or pass `--seed` and set a default in config.

Training via SLURM
- Single job:
	sbatch job.slurm --max_epochs 400 --seed 42
- Array (version == array id):
	sbatch multijob.slurm --max_epochs 400
- Grid over GMM weights × seeds (predefined):
	sbatch multijob_scan.slurm

Parameter Scan (Ray Tune)
- Local/interactive:
	python3 parameter_scan.py --version 5
- SLURM with auto versioning:
	sbatch parameter_scan.slurm  
	(optional: pass version as first argument)
- Check progress:
	python3 check_ray_progress.py --dir /work/aa0238/a271125/logs_ray/vae_gmm_multi_objective_scan/version_5

Configure `config.py`
- ModelConfig
	- `input_shape`: e.g., (1, 61, 181)
	- `layer_sizes`, `dropout_prob`: MLP layers & dropout
	- `num_clusters`: GMM clusters
	- `loss_func_name`: e.g., `MSELoss`
- TrainingConfig
	- `batch_size`, `vae_lr`, `clustering_lr`, `use_annealing`
	- Loss weights: `recon_weight`, `kld_weight`, `vae_end_value`, `gmm_end_value`, `reg_end_value`, `cat_end_value`
	- `dynamic_multiplier`, `dynamic_update_epoch`, `dynamic_reduction_factor`
	- `seed` (required), `log_img` for t-SNE images
- TrainingSetup (phases)
	- `warmup_epochs`, `vae_epochs`, `adapt_epochs`, `gmm_epochs`, `cat_epochs`, `reg_epochs`
	- `kmeans_init_epoch`: when to KMeans++ init cluster params
	- annealing and schedulers (`annealing_type`, `vae_lr_factor`, `vae_lr_patience`, `clustering_warmup`, `cosine_*`)
- DataConfig
	- `data_dir`, `log_dir`, `experiment`, `num_workers`
- HardwareConfig
	- `accelerator` ('gpu'/'cpu'), `devices` (tuple of device ids)

Outputs & Logging
- Checkpoints: `version_X/checkpoints/` (`last.ckpt` and epoch snapshots).
- TensorBoard: in `DataConfig.log_dir / experiment/version_X` (scalars and optional images/t-SNE).
- Metrics include reconstruction, global/cluster/categorical KLD, variance regularization, silhouette/DB/CH, entropy, density/smoothness.

Post-training Analysis & Plots
1) ClusteringLoader
	 from loaders import ClusteringLoader
	 import torch
	 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	 CL = ClusteringLoader(
			 log_dir='/work/aa0238/a271125/logs/Correct_Normalization',
			 experiment='Experiment_',
			 version='0',
			 nc_path='/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc',
			 device=device,
			 pca_kmeans_path='pca_km.pkl',
			 mapping_path='cluster_mapping.json')
	 CL.auto_map_by_correlation()
	 CL.plot_model_composition()
	 CL.plot_tsne(use_kmeans=False)
	 periods_df = CL.cluster_periods()

2) Reference patterns & Taylor diagrams
	 - `pattern_reference_manager.py` builds/caches PCA+KMeans reference patterns (+ optional naming).
	 - `pattern_taylor.py` compares model vs reference (Hungarian) and draws Taylor diagrams.

3) Training diagnostics figure from TensorBoard
	 - `tb_utils.py` provides `plot_training_diagnostics(...)` (losses, annealing weights, phase timeline).

Notable Callbacks & Schedulers
- Versioning: pass `--version` or auto-increment; `--resume True` reloads `last.ckpt`.
- Specific epoch checkpoints and “last” checkpoint.
- Optional shuffle switch at `kmeans_init_epoch`.
- KMeans++ init of clusters in latent space.
- LR scheduling: LambdaLR→Cosine for clustering; ReduceLROnPlateau for VAE.