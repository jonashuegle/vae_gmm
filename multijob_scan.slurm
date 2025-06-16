#!/bin/bash
#SBATCH --account=aa0238_gpu
#SBATCH --job-name=multi_scan
#SBATCH --output=/home/a/a271125/VAE-GMM/logs/train_%A_%a.out
#SBATCH --error=/home/a/a271125/VAE-GMM/logs/train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=1-10

GMM_VAL=(0.001 0.002 0.003 0.004 0.005220209 0.007 0.008 0.009 0.01 0.015)

# Wähle basierend auf SLURM_ARRAY_TASK_ID den passenden Wert
GMM=${GMM_VAL[$SLURM_ARRAY_TASK_ID]}

eval "$(conda shell.bash hook)"
conda activate MA

echo "#### Starting run $SLURM_ARRAY_TASK_ID####"

python3 training.py \
    --version $SLURM_ARRAY_TASK_ID \
    --gmm_end_value $GMM \
    "$@"
