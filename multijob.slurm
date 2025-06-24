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
#SBATCH --array=1-5


eval "$(conda shell.bash hook)"
conda activate MA

echo "#### Starting run $SLURM_ARRAY_TASK_ID####"

python3 training.py \
    --version $SLURM_ARRAY_TASK_ID \
    "$@"
