#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import numpy as np
import os
import json
import hashlib
import argparse
from datetime import datetime

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from dataset import DataModule
from config import ModelConfig, TrainingConfig, TrainingSetup, DataConfig
from VAE_GMM import VAE
from pytorch_lightning.loggers import TensorBoardLogger

# Einfacher Callback ohne komplexe Integration
class SimpleCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoche {trainer.current_epoch}/{trainer.max_epochs} abgeschlossen")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {k: round(v.item(), 4) if isinstance(v, torch.Tensor) else v 
                 for k, v in trainer.callback_metrics.items() 
                 if isinstance(v, (torch.Tensor, float))}
        print(f"Validierungsmetriken: {metrics}")

def train_vae(config, checkpoint_dir=None):
    # Erzeugen einer eindeutigen ID für diesen Trial basierend auf den Hyperparametern
    config_str = str(sorted([(k, str(v)) for k, v in config.items()]))
    trial_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Pfade für Zwischenergebnisse und Checkpoints
    interim_results_dir = os.path.join(EXPERIMENT_DIR, "interim_results")
    os.makedirs(interim_results_dir, exist_ok=True)
    result_file = os.path.join(interim_results_dir, f"trial_{trial_id}.json")
    
    if os.path.exists(result_file):
        print(f"Lade vorhandene Ergebnisse für Trial {trial_id}")
        with open(result_file, 'r') as f:
            result = json.load(f)
        metrics = result["metrics"] if "metrics" in result else result
        # Reporte die Metriken explizit, damit Ray Tune sie findet:
        tune.report(
            silhouette=metrics["silhouette"],
            loss_recon=metrics["loss_recon"],
            calinski_harabasz=metrics["calinski_harabasz"],
            davies_bouldin=metrics["davies_bouldin"],
            cluster_entropy=metrics["cluster_entropy"],
            smoothness=metrics["smoothness"],
        )
        return metrics
    
    print(f"\n{'='*50}")
    print(f"Starte Training für Trial {trial_id} mit Konfiguration: {config}")
    
    # Model-Setup
    latent_dim = config["latent_dim"]
    model_config = ModelConfig(layer_sizes=(4000, 2000, 800, 200, 100, latent_dim))
    training_config = TrainingConfig(
        clustering_lr=config["clustering_lr"],
        vae_lr=config["vae_lr"],
        recon_weight=config["recon_weight"],
        kld_weight=config["kld_weight"],
        gmm_end_value=config["gmm_end_value"],
        reg_end_value=config["reg_end_value"],
        cat_end_value=config["cat_end_value"],
        seed=42,
        log_img=False,  # Kein Logging von Bildern für schnellere Ausführung
    )
    training_setup = TrainingSetup(
        gmm_epochs=config["gmm_epochs"],
        warmup_epochs=config["warmup_epochs"],
        adapt_epochs=config["adapt_epochs"],
        vae_lr_factor=config["vae_lr_factor"],
        vae_lr_patience=config["vae_lr_patience"],
    )
    
    data_config = DataConfig()
    num_epochs = 300
    
    # Daten und Model vorbereiten
    data_module = DataModule(
        data_config.data_dir, 
        batch_size=training_config.batch_size, 
        num_workers=0,
    )
    
    # Wichtig: Setup ausführen, bevor auf train_dataset zugegriffen wird
    data_module.setup()
    
    model = VAE(model_config=model_config, training_config=training_config, training_setup=training_setup)
    
    # Checkpoint laden wenn vorhanden
    custom_checkpoint_dir = os.path.join(interim_results_dir, f"checkpoint_{trial_id}")
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    custom_checkpoint_path = os.path.join(custom_checkpoint_dir, "model.ckpt")
    
    if os.path.exists(custom_checkpoint_path):
        print(f"Lade Checkpoint: {custom_checkpoint_path}")
        checkpoint = torch.load(custom_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    
    # Vereinfachter Trainer mit direktem Output
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=num_epochs,
        logger=False,  # Kein Logger für direkten Output
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks=[SimpleCallback()],
        # profiler="simple",
        precision=32,
        enable_model_summary=True,
        num_sanity_val_steps=0  # Sanity-Check überspringen für schnelleren Start
    )
    
    # Training starten
    print(f"⚡ DataLoader hat {len(data_module.train_dataset)} Training, {len(data_module.val_dataset)} Validation Samples")
    print(f"🚀 TRAINING STARTET JETZT mit {num_epochs} Epochen")
    
    trainer.fit(model, data_module)
    
    print(f"✅ TRAINING BEENDET nach {trainer.current_epoch} Epochen")    
    
    # Ergebnisse erfassen
    try:
        metrics = {
            "loss_recon": trainer.callback_metrics["val/loss/recon"].item(),
            "silhouette": trainer.callback_metrics["val/metric/silhouette"].item(),
            "calinski_harabasz": trainer.callback_metrics["val/metric/calinski_harabasz_index"].item(),
            "davies_bouldin": trainer.callback_metrics["val/metric/davies_bouldin_index"].item(),
            "cluster_entropy": trainer.callback_metrics["val/metric/cluster_entropy"].item(),
            "smoothness": trainer.callback_metrics.get("val/metric/smoothness", torch.tensor(-1e3)).item(),
        }
    except Exception:
        # Fallback-Metriken…
        metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v)
                   for k, v in {
                       "loss_recon": 1e3,
                       "silhouette": -1e3,
                       "calinski_harabasz": -1e3,
                       "davies_bouldin": 1e3,
                       "cluster_entropy": -1e3,
                       "smoothness": -1e3,
                   }.items()}
    
    # Kombiniere config + metrics in ein einziges JSON
    result = {
        "trial_id": trial_id,
        "config": config,
        "metrics": metrics
    }
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    tune.report(
        silhouette=metrics["silhouette"],
        loss_recon=metrics["loss_recon"],
        calinski_harabasz=metrics["calinski_harabasz"],
        davies_bouldin=metrics["davies_bouldin"],
        cluster_entropy=metrics["cluster_entropy"],
        smoothness=metrics["smoothness"],
    )

    return metrics


if __name__ == "__main__":
    # Konfiguration
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=int(os.environ.get("VAE_EXPERIMENT_VERSION", 1)))
    parser.add_argument("--gpus", type=int, default=0, help="Anzahl zu verwendender GPUs (0=auto)")
    args = parser.parse_args()
    
    # Experiment-Setup
    EXPERIMENT_VERSION = args.version
    EXPERIMENT_NAME = f"vae_gmm_scan_v{EXPERIMENT_VERSION}"
    BASE_DIR = '/work/aa0238/a271125/logs_ray'
    EXPERIMENT_DIR = f"{BASE_DIR}/vae_gmm_multi_objective_scan/version_{EXPERIMENT_VERSION}"
    
    # Verzeichnisse erstellen
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # Ressourcenerkennung
    requested_gpus = args.gpus
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested_gpus > 0:
        gpus_to_use = min(requested_gpus, available_gpus)
    else:
        gpus_to_use = available_gpus
    
    print(f"GPUs verfügbar: {available_gpus}, verwende: {gpus_to_use}")
    
    # Ray initialisieren - vereinfacht
    ray.init(
        num_cpus=min(8, os.cpu_count()),
        num_gpus=gpus_to_use,
        _memory=32 * 1024 * 1024 * 1024,  # 32GB RAM-Limit
        _redis_max_memory=1 * 1024 * 1024 * 1024,  # 1GB für Redis
        object_store_memory=8 * 1024 * 1024 * 1024,  # 8GB
        local_mode=False,
        log_to_driver=True, 
        ignore_reinit_error=True,
        include_dashboard=False
    )
    
    # # Vereinfachter Search Space
    # search_space = {
    #     "clustering_lr": tune.uniform(1e-06, 1e-05),
    #     "gmm_end_value": tune.uniform(0.0045, 0.0055),
    #     "reg_end_value": tune.uniform(0.35, 0.45),
    #     "cat_end_value": tune.uniform(0.0020, 0.1),
    #     "gmm_epochs": tune.choice([60, 80, 100]),
    #     "cosine_eta_min": tune.loguniform(1e-8, 3e-8),
    #     "vae_lr_factor": tune.uniform(0.75, 0.85),
    #     "vae_lr_patience": tune.choice([20, 25, 30]),
    # }

    search_space = {
        # Learning Rates
        "vae_lr": tune.loguniform(1e-5, 5e-3),
        "clustering_lr": tune.loguniform(1e-6, 5e-4),

        # Loss Weights
        "recon_weight": tune.loguniform(5e-2, 1.0),  # 0.1 ist guter Default
        
        "vae_end_value": tune.loguniform(1e-5, 1e-2),    # 0.0001 - 0.001 je nach Fall
        "gmm_end_value": tune.uniform(0.003, 0.01),
        "reg_end_value": tune.uniform(0.02, 0.5),
        "cat_end_value": tune.loguniform(0.001, 0.05),

        # Scheduler
        "vae_lr_factor": tune.uniform(0.7, 0.95),
        "vae_lr_patience": tune.choice([10, 20, 30, 40]),
        # Latent Space
        "latent_dim": tune.choice([14, 20, 30, 40, 50, 60, 70]),

        # Epochen für Phasen
        "gmm_epochs": tune.choice([50, 80, 100, 150]),
        "warmup_epochs": tune.choice([15, 20, 25, 30]),
        "adapt_epochs": tune.choice([10, 15, 20]),
        # ... weitere falls du magst!
    }

    points_to_evaluate = [
        {
            "latent_dim": 14, 
            "vae_lr": 0.000193066,
            "clustering_lr": 5.929e-06,
            "recon_weight": 0.1,
            "vae_end_value": 0.001,
            "gmm_end_value": 0.005220209,
            "reg_end_value": 0.04072058,
            "cat_end_value": 0.005362321,
            "vae_lr_factor": 0.777187766,
            "vae_lr_patience": 30,
            "gmm_epochs": 80,
            "warmup_epochs": 25,
            "adapt_epochs": 15,
            # usw., je nach Parameter die du explizit vergleichen willst
        }
    ]

    
    # Scheduler
    scheduler = ASHAScheduler(
        max_t=300,
        grace_period=120,
        reduction_factor=2,
        metric="silhouette",
        mode="max"
    )
    
    # Reporter
    reporter = tune.CLIReporter(
        parameter_columns=[
            "clustering_lr", "gmm_end_value", "reg_end_value",
            "cat_end_value", "gmm_epochs", "vae_lr_factor", 
        ],
        metric_columns=[
            "loss_recon", "silhouette", "calinski_harabasz",
            "davies_bouldin", "cluster_entropy", "smoothness", "training_iteration"
        ],
        max_report_frequency=60,  # Alle 60 Sekunden berichten
    )
    
    # Suchkonfiguration
    search_alg = OptunaSearch(
        metric=["silhouette", "loss_recon", "smoothness"],
        mode=["max", "min", "max"],
        points_to_evaluate=points_to_evaluate,
    )
    
    # Parameter für Multi-GPU-Setup
    num_samples = 256# if gpus_to_use <= 1 else min(gpus_to_use * 10, 20)
    resources_per_trial = {"gpu": 1, "cpu": 4}
    max_concurrent = max(1, gpus_to_use)
    
    # Ray Tune ausführen
    print(f"Start Parameter-Scan Version {EXPERIMENT_VERSION} mit {num_samples} Samples, max {max_concurrent} parallel")
    result = tune.run(
        train_vae,
        search_alg=search_alg,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=EXPERIMENT_NAME,
        local_dir=EXPERIMENT_DIR,
        config=search_space,
        # Unsere eigene Fortsetzungslogik ist implementiert, daher kein resume="AUTO"
        max_failures=2,
        max_concurrent_trials=max_concurrent,
    )
    
    # Ergebnisse speichern
    best_trial = result.get_best_trial("silhouette", "max")
    print(f"Bester Trial: {best_trial.trial_id}")
    print(f"Beste Metriken: {best_trial.last_result}")
    
    # Ergebnisse als JSON speichern
    best_results_path = os.path.join(EXPERIMENT_DIR, "best_results.json")
    with open(best_results_path, "w") as f:
        json.dump({
            "config": best_trial.config,
            "metrics": {k: v for k, v in best_trial.last_result.items() if isinstance(v, (int, float))}
        }, f, indent=2)
    
    print(f"Ergebnisse gespeichert unter: {best_results_path}")
    ray.shutdown()