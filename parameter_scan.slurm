#!/bin/bash
#SBATCH --account=aa0238_gpu
#SBATCH --job-name=param_scan
#SBATCH --output=/home/a/a271125/VAE-GMM/logs/parameter_scan_%j.out
#SBATCH --error=/home/a/a271125/VAE-GMM/logs/parameter_scan_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@600

# 1) Ins Arbeitsverzeichnis wechseln
cd /home/a/a271125/VAE-GMM

# 2) Conda-Umgebung aktivieren
eval "$(conda shell.bash hook)"
conda activate MA

# 3) PYTHONPATH setzen (falls benötigt)
export PYTHONPATH="${PYTHONPATH}:/home/a/a271125/VAE-GMM"

# 4) GPUs sichtbar machen
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE-1)))

# 5) OMP-Threads beschränken
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 6) Logs-Verzeichnis (für Ray-Resultate) erstellen
mkdir -p /work/aa0238/a271125/logs_ray/vae_gmm_multi_objective_scan/version_1
mkdir -p /work/aa0238/a271125/logs_ray/vae_gmm/version_1/best_results

echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK, GPUs: $SLURM_GPUS_PER_NODE"

# 7) Parameter-Scan starten — ray.init() in Deinem Script startet den Head selbst
python parameter_scan.py

echo "✅ parameter_scan.py finished"
