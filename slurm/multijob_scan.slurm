#!/bin/bash
#SBATCH --account=aa0238_gpu
#SBATCH --job-name=multi_scan
#SBATCH --output=/home/a/a271125/VAE-GMM/logs/train_%A_%a.out
#SBATCH --error=/home/a/a271125/VAE-GMM/logs/train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --signal=USR1@600   
#SBATCH --time=08:00:00

### Parameter-Definition ###
# 10 verschiedene GMM-Weights
GMM_VAL=(0.001 0.002 0.003 0.004 0.005220209 0.007 0.008 0.009 0.01 0.015)
# 5 verschiedene Seeds
SEEDS=(42 100 123) #2025 31415

num_gmm=${#GMM_VAL[@]}      # = 10
num_seeds=${#SEEDS[@]}      # = 3
total=$(( num_gmm * num_seeds ))  # = 30

#SBATCH --array=1-${total}

### Index-Berechnung ###
# SLURM_ARRAY_TASK_ID läuft von 1 bis total
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))          # Null-basiert: 0…49
gmm_idx=$(( idx / num_seeds ))              # Ganzzahliger Quotient → 0…9
seed_idx=$(( idx % num_seeds ))             # Rest → 0…4

# Parameter für diesen Job
GMM=${GMM_VAL[gmm_idx]}
SEED=${SEEDS[seed_idx]}
# Eindeutige Version (z.B. g3_s1)
VERSION="g${gmm_idx}_s${seed_idx}"

### Environment aktivieren ###
eval "$(conda shell.bash hook)"
conda activate MA

echo "#### Starting run SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} (GMM=${GMM}, SEED=${SEED}, VERSION=${VERSION}) ####"

cd "$SLURM_SUBMIT_DIR/.."

### Training aufrufen ###
python3 -m src.training.py \
    --version "${VERSION}" \
    --gmm_end_value "${GMM}" \
    --seed "${SEED}"
