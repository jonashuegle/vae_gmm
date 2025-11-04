#!/bin/bash
#SBATCH --account= ...
#SBATCH --job-name=multi_scan
#SBATCH --output=./logs_slurm/train_%A_%a.out
#SBATCH --error=./logs_slurm/train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=71-100


module load python3
eval "$(conda shell.bash hook)"
conda activate MA

echo "#### Starting run $SLURM_ARRAY_TASK_ID####"

cd "$SLURM_SUBMIT_DIR/.."

python3 -m src.training \
    --version $SLURM_ARRAY_TASK_ID \
    "$@"
