#!/bin/bash
#SBATCH --account= ...
#SBATCH --job-name=multi_scan
#SBATCH --output=./logs_slurm/train_%A_%a.out
#SBATCH --error=./logs_slurm/train_%A_%a.err
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1

eval "$(conda shell.bash hook)"
conda activate MA

# du hast sbatch aus /home/a/a271125/VAE-GMM aufgerufen
# also brauchst du hier KEIN ..
cd "$SLURM_SUBMIT_DIR"

# Projektroot für Python sichtbar machen
export PYTHONPATH="$(pwd):$PYTHONPATH"

# sicherstellen, dass die beiden Variablen gesetzt sind
: "${DATA_PATH:?DATA_PATH not set}"
: "${LOG_DIR:?LOG_DIR not set}"

python3 -m src.training "$@"
