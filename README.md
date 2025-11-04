# VAE-GMM — Variational Autoencoder with Gaussian Mixture Clustering

## Overview
- Train a Variational Autoencoder (VAE) with a Gaussian Mixture Model (GMM) on atmospheric SLP fields (or similar gridded NetCDF data) to obtain regime / pattern clusters.
- Implemented with PyTorch Lightning → automatic versioned logging (`experiment/version_X/`) and checkpoints.
- Includes: single-run training, SLURM job scripts, optional Ray Tune parameter scan, and utilities for post-processing (pattern mapping, t-SNE, regime composition).
- Code lives in `src/`, configuration in `config.py`.

## Requirements
- Conda / Python env with PyTorch, PyTorch Lightning, xarray, scikit-learn, matplotlib, …
- A NetCDF file with the SLP data.
- GPU is strongly recommended.

## Data & Logs
This project expects data and log paths via environment variables.

Set them **before** running:

export DATA_PATH=/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc
export LOG_DIR=./logs

- DATA_PATH → path to the NetCDF file used for training
- LOG_DIR → base directory where Lightning will create `experiment/version_X/...`

On the cluster you pass them via `sbatch --export=...` (see below).
If they are not set, `config.py` may fall back to local defaults, but explicit is better.

## Project Structure
- `config.py` — central config (`ModelConfig`, `TrainingConfig`, `TrainingSetup`, `DataConfig`, `HardwareConfig`)
- `src/training.py` — training entry point (Lightning trainer, version handling, callbacks)
- `src/VAE_GMM.py` — model: encoder/decoder, GMM part, losses, schedulers, metrics, t-SNE logging
- `src/dataset.py` — `CustomDataset` (xarray NetCDF) + Lightning `DataModule` (train/val split, full-data loader)
- `src/loaders.py` / `src/clustering_loader.py` — load trained model, extract gamma/mu, auto/manual mapping, plotting
- `src/plotting.py`, `src/visualize_tsne.py`, `src/pattern_reference_manager.py`, `src/pattern_taylor.py`, `src/tb_utils.py` — analysis & visualization
- `slurm/*.slurm` — example SLURM scripts (training, multijob, parameter scan)

Important: since code is in `src/`, run modules like this:

python -m src.training ...


## Quick Start (local)

1. Set your paths:

export DATA_PATH=/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc
export LOG_DIR=./logs

2. Activate env:

conda activate MA

3. Start training:

python -m src.training --max_epochs 400 --seed 42

4. Resume latest version (if checkpoints exist):

python -m src.training --resume True --seed 42

5. Run a fixed version:

python -m src.training --version 3 --max_epochs 300 --seed 42

Notes:
- logs/checkpoints will appear under

$LOG_DIR / <experiment_name> / version_X / ...

where `<experiment_name>` comes from `DataConfig.experiment` in `config.py`.

## Training via SLURM

Example job script (save e.g. as `slurm/training.slurm`):

#!/bin/bash
#SBATCH --account=aa0238_gpu
#SBATCH --job-name=training
#SBATCH --output=/home/a/a271125/VAE-GMM/logs/training_%j.log
#SBATCH --error=/home/a/a271125/VAE-GMM/logs/training_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1

eval "$(conda shell.bash hook)"
conda activate MA

# go to project root (script lives in slurm/)
cd "$SLURM_SUBMIT_DIR/.."

# make project importable (for src.*)
export PYTHONPATH="$(pwd):$PYTHONPATH"

# require that DATA_PATH and LOG_DIR are provided
: "${DATA_PATH:?DATA_PATH not set}"
: "${LOG_DIR:?LOG_DIR not set}"

python3 -m src.training "$@"

Submit it like this:

sbatch --export=./data/slp.nc,LOG_DIR=./log slurm/training.slurm --max_epochs 1 --seed 42

- everything after `slurm/training.slurm` is forwarded to `python -m src.training …`
- everything in `--export=...` becomes an environment variable and is picked up by your `DataConfig`

## Parameter Scan (Ray Tune) (optional)

Local / interactive:

python -m src.parameter_scan --version 5

SLURM with auto versioning (according to your script):

sbatch slurm/parameter_scan.slurm
sbatch slurm/parameter_scan.slurm 7   # explicit version

## Configure `config.py`

- ModelConfig
  - `input_shape` (e.g. `(1, 61, 181)`)
  - `layer_sizes`, `dropout_prob`
  - `num_clusters`
  - `loss_func_name` (e.g. `"MSELoss"`)

- TrainingConfig
  - `batch_size`, `vae_lr`, `clustering_lr`, `use_annealing`
  - loss weights: `recon_weight`, `kld_weight`, `vae_end_value`, `gmm_end_value`, `reg_end_value`, `cat_end_value`
  - `dynamic_multiplier`, `dynamic_update_epoch`, `dynamic_reduction_factor`
  - `seed` is now set from the CLI in `src/training.py`

- TrainingSetup
  - `warmup_epochs`, `vae_epochs`, `adapt_epochs`, `gmm_epochs`, `cat_epochs`, `reg_epochs`
  - `kmeans_init_epoch`
  - annealing / scheduler params

- DataConfig
  - `data_dir` (overridden by `DATA_PATH`)
  - `log_dir` (overridden by `LOG_DIR`)
  - `experiment`
  - `num_workers`

- HardwareConfig
  - `accelerator` (`"gpu"` / `"cpu"`)
  - `devices` (e.g. `(0,)`)

## Outputs & Logging

- Checkpoints:

$LOG_DIR / <experiment> / version_X / checkpoints / last.ckpt

plus epoch-specific checkpoints (due to the custom checkpoint callback).

- TensorBoard:

$LOG_DIR / <experiment> / version_X /

- Metrics include:
  - reconstruction loss
  - global / cluster / categorical KLD
  - variance regularization
  - after cluster init: silhouette, Davies–Bouldin, Calinski–Harabasz, entropy, density, smoothness

## Post-training Analysis

from src.loaders import ClusteringLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CL = ClusteringLoader(
    log_dir='./log',
    experiment='Experiment_',
    version='0',
    nc_path='./data/slp.nc',
    device=device,
    pca_kmeans_path='pca_km.pkl',
    mapping_path='cluster_mapping.json'
)

CL.auto_map_by_correlation()
CL.plot_model_composition()
CL.plot_tsne(use_kmeans=False)
periods_df = CL.cluster_periods()

Notes:
- run from project root so that `src/` is on the path (the SLURM script already does this)
- on SLURM you must pass `DATA_PATH` and `LOG_DIR` via `--export=...`
- Lightning’s “srun is available” warning can be ignored for single-GPU runs
