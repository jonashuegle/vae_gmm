#!/bin/bash
#SBATCH --account=aa0238_gpu
#SBATCH --job-name=final_multi_objective_scan
#SBATCH --output=/home/a/a271125/VAE-GMM/logs/final_scan%j.out
#SBATCH --error=/home/a/a271125/VAE-GMM/logs/final_scan%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@600

# Basis-Verzeichnis für deine Ray-Logs auf dem Cluster
BASE_DIR="/work/aa0238/a271125/logs_ray"

# --- Funktion zur automatischen Bestimmung der nächsten Version ---
get_next_version() {
    local max_version=0

    if [ -d "${BASE_DIR}/vae_gmm_multi_objective_scan" ]; then
        for dir in "${BASE_DIR}/vae_gmm_multi_objective_scan"/version_*; do
            if [ -d "$dir" ]; then
                version=$(echo "$dir" | grep -oP 'version_\K\d+')
                if [[ "$version" =~ ^[0-9]+$ ]] && [ "$version" -gt "$max_version" ]; then
                    max_version=$version
                fi
            fi
        done
    fi

    echo $((max_version + 1))
}

# --- Version bestimmen ---
if [ -z "$1" ]; then
    EXPERIMENT_VERSION=$(get_next_version)
    echo "Keine Version angegeben, verwende automatisch nächste Version: $EXPERIMENT_VERSION"
else
    EXPERIMENT_VERSION=$1
    echo "Verwende angegebene Version: $EXPERIMENT_VERSION"
fi

echo "DEBUG: Gesetzte Experiment-Version: $EXPERIMENT_VERSION"
echo "DEBUG: Durchsuche Versionsordner in ${BASE_DIR}/vae_gmm_multi_objective_scan"
for dir in "${BASE_DIR}/vae_gmm_multi_objective_scan"/version_*; do
    echo "  Gefunden: $dir"
done

# Logs-Verzeichnisse für diesen Run anlegen
mkdir -p "${BASE_DIR}/vae_gmm_multi_objective_scan/version_${EXPERIMENT_VERSION}"
mkdir -p "${BASE_DIR}/vae_gmm/version_${EXPERIMENT_VERSION}/best_results"

# Ray tmp säubern
if [ -d "/tmp/ray" ]; then
    echo "Cleaning Ray temporary files..."
    rm -rf /tmp/ray
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Graceful Shutdown für SLURM-Signal
_term() {
  echo "Received SLURM TERM signal. Cleaning up and exiting..."
  exit 0
}
trap _term USR1

# Ray-Umgebungsvariablen
export RAY_OBJECT_STORE_MEMORY=8000000000
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=0.95
export RAY_memory_usage_threshold=0.95
export PL_DISABLE_FORK="1"

# Conda-Umgebung
eval "$(conda shell.bash hook)"
conda activate MA

# ⬇️ WICHTIG: in den Projektroot hochgehen
cd "$SLURM_SUBMIT_DIR/.."  # slurm/ -> projektroot/

# falls dein Python-Code jetzt Umgebungsvariablen nutzt (wie wir es im config.py vorgesehen haben),
# kannst du sie hier setzen. Beispiel:
# export LOG_DIR="${BASE_DIR}/vae_gmm_multi_objective_scan/version_${EXPERIMENT_VERSION}"

# Parameter-Scan starten – jetzt mit Modul-Aufruf aus src/
python3 -m src.parameter_scan --version "$EXPERIMENT_VERSION"

RETVAL=$?

echo "parameter_scan.py finished with status: $RETVAL"
echo "End time: $(date)"

exit $RETVAL
