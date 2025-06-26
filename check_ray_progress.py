#!/usr/bin/env python3

import os
import json
import argparse
from datetime import datetime

def check_ray_progress(experiment_dir):
    """Überprüft den Fortschritt eines Ray-Tune-Experiments."""
    
    # Prüfen ob das Verzeichnis existiert
    if not os.path.exists(experiment_dir):
        print(f"❌ Experiment-Verzeichnis nicht gefunden: {experiment_dir}")
        return
    
    # Trials-Verzeichnis durchsuchen
    trials_dir = os.path.join(experiment_dir, "vae_gmm")
    if not os.path.exists(trials_dir):
        print(f"❌ Keine Trials gefunden in: {trials_dir}")
        return
    
    # Trial-Verzeichnisse auflisten
    trial_dirs = [d for d in os.listdir(trials_dir) if d.startswith("train_vae_")]
    
    if not trial_dirs:
        print(f"❌ Keine Trial-Verzeichnisse gefunden in: {trials_dir}")
        return
    
    print(f"📊 Experiment: {os.path.basename(experiment_dir)}")
    print(f"📁 Pfad: {experiment_dir}")
    print(f"🔍 Gefundene Trials: {len(trial_dirs)}")
    
    # Status zählen
    completed = 0
    running = 0
    error = 0
    
    # Beste Metriken
    best_silhouette = -float('inf')
    best_trial_id = None
    
    # Alle Trials durchgehen
    for trial_dir in trial_dirs:
        full_path = os.path.join(trials_dir, trial_dir)
        
        # Nach result.json suchen
        result_file = os.path.join(full_path, "result.json")
        params_file = os.path.join(full_path, "params.json")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                # Trial ist abgeschlossen
                completed += 1
                
                # Silhouette-Score prüfen
                if "silhouette" in result:
                    silhouette = result["silhouette"]
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_trial_id = trial_dir
            except:
                running += 1
        elif os.path.exists(params_file):
            # Trial wurde gestartet, aber noch nicht abgeschlossen
            running += 1
        else:
            # Unbekannter Status
            error += 1
    
    # Zusammenfassung ausgeben
    print(f"\n📊 ZUSAMMENFASSUNG:")
    print(f"   Abgeschlossene Trials: {completed}")
    print(f"   Laufende Trials: {running}")
    print(f"   Fehlerhafte Trials: {error}")
    
    if best_trial_id:
        print(f"\n🏆 Bester Trial: {best_trial_id}")
        print(f"   Silhouette Score: {best_silhouette:.4f}")
    
    print(f"\n⏱️  Letzte Aktualisierung: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray Tune Fortschritt überprüfen')
    parser.add_argument('--dir', type=str, 
                        default='/work/aa0238/a271125/logs_ray/vae_gmm_multi_objective_scan/version_4',
                        help='Experiment-Verzeichnis')
    
    args = parser.parse_args()
    check_ray_progress(args.dir)
