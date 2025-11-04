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

# alle übergebenen Parameter ($@) an Python weiterreichen
python3 training.py "$@"
