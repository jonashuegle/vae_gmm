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

# Dynamische Versionsverwaltung
BASE_DIR="/work/aa0238/a271125/logs_ray"

# Funktion zur automatischen Bestimmung der nächsten Version
get_next_version() {
    local max_version=0
    
    # Prüfe alle vorhandenen Versionsordner
    if [ -d "${BASE_DIR}/vae_gmm_multi_objective_scan" ]; then
        for dir in "${BASE_DIR}/vae_gmm_multi_objective_scan"/version_*; do
            if [ -d "$dir" ]; then
                # Extrahiere Versionsnummer aus Ordnernamen
                version=$(echo "$dir" | grep -oP 'version_\K\d+')
                if [[ "$version" =~ ^[0-9]+$ ]] && [ "$version" -gt "$max_version" ]; then
                    max_version=$version
                fi
            fi
        done
    fi
    
    # Nächste Version = höchste vorhandene Version + 1
    echo $((max_version + 1))
}

# Versionsnummer bestimmen: Entweder vom Argument oder automatisch die nächste
if [ -z "$1" ]; then
    EXPERIMENT_VERSION=$(get_next_version)
    echo "Keine Version angegeben, verwende automatisch nächste Version: $EXPERIMENT_VERSION"
else
    EXPERIMENT_VERSION=$1
    echo "Verwende angegebene Version: $EXPERIMENT_VERSION"
fi

# Debug-Ausgabe der erkannten Version
echo "DEBUG: Gesetzte Experiment-Version: $EXPERIMENT_VERSION"

# Debug: Durchsuche vorhandene Versionsordner 
echo "DEBUG: Durchsuche Versionsordner in ${BASE_DIR}/vae_gmm_multi_objective_scan"
for dir in "${BASE_DIR}/vae_gmm_multi_objective_scan"/version_*; do
    echo "  Gefunden: $dir"
done

# Logs-Verzeichnisse dynamisch erstellen (mit der bestimmten Version)
mkdir -p "${BASE_DIR}/vae_gmm_multi_objective_scan/version_${EXPERIMENT_VERSION}"
mkdir -p "${BASE_DIR}/vae_gmm/version_${EXPERIMENT_VERSION}/best_results" 

# Ray Temp-Verzeichnis reinigen um eventuellen Datei-Korruptionen vorzubeugen
if [ -d "/tmp/ray" ]; then
    echo "Cleaning Ray temporary files..." 
    rm -rf /tmp/ray
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Graceful Shutdown-Handler für SLURM-Signal (USR1)
_term() {
  echo "Received SLURM TERM signal. Cleaning up and exiting..."
  exit 0
}

trap _term USR1

# WICHTIG: Ray-spezifische Umgebungsvariablen
export RAY_OBJECT_STORE_MEMORY=8000000000
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=0.95
export RAY_memory_usage_threshold=0.95
export PL_DISABLE_FORK="1"

# Conda-Umgebung aktivieren
eval "$(conda shell.bash hook)"
conda activate MA

# Parameter-Scan starten mit direkter Versionsübergabe als Argument
python parameter_scan.py --version $EXPERIMENT_VERSION

RETVAL=$?

echo "✅ parameter_scan.py finished with status: $RETVAL"
echo "End time: $(date)"

exit $RETVAL