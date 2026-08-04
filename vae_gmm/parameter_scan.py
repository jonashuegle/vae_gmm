#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from vae_gmm.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    TrainingSetup,
)
from vae_gmm.dataset import DataModule
from vae_gmm.VAE_GMM import VAE


class SimpleCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}/{trainer.max_epochs} finished")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            k: round(v.item(), 4) if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (torch.Tensor, float))
        }
        print(f"Validation metrics: {metrics}")


def train_vae(config, checkpoint_dir=None):
    # Trial id derived from the hyperparameters, so a resumed run maps to the same directory.
    config_str = str(sorted([(k, str(v)) for k, v in config.items()]))
    trial_id = hashlib.md5(config_str.encode()).hexdigest()[:8]

    interim_results_dir = os.path.join(EXPERIMENT_DIR, "interim_results")
    os.makedirs(interim_results_dir, exist_ok=True)
    result_file = os.path.join(interim_results_dir, f"trial_{trial_id}.json")

    if os.path.exists(result_file):
        print(f"Loading existing results for trial {trial_id}")
        with open(result_file) as f:
            result = json.load(f)
        metrics = result["metrics"] if "metrics" in result else result
        # Report explicitly so Ray Tune picks the metrics up.
        tune.report(
            silhouette=metrics["silhouette"],
            loss_recon=metrics["loss_recon"],
            calinski_harabasz=metrics["calinski_harabasz"],
            davies_bouldin=metrics["davies_bouldin"],
            cluster_entropy=metrics["cluster_entropy"],
            smoothness=metrics["smoothness"],
            balance=metrics["balance"],
        )
        return metrics

    print(f"\n{'=' * 50}")
    print(f"Starting trial {trial_id} with config: {config}")

    latent_dim = config["latent_dim"]
    model_config = ModelConfig(layer_sizes=(4000, 2000, 800, 200, 100, latent_dim))
    training_config = TrainingConfig(
        clustering_lr=config["clustering_lr"],
        vae_lr=config["vae_lr"],
        recon_weight=config["recon_weight"],
        vae_end_value=config["vae_end_value"],
        gmm_end_value=config["gmm_end_value"],
        reg_end_value=config["reg_end_value"],
        cat_end_value=config["cat_end_value"],
        seed=42,
        log_img=False,  # no image logging, keeps the sweep fast
    )
    training_setup = TrainingSetup(
        gmm_epochs=config["gmm_epochs"],
        warmup_epochs=config["vae_epochs"],
        vae_epochs=config["vae_epochs"],
        kmeans_init_epoch=config["vae_epochs"],
        clustering_warmup=config["vae_epochs"],
        vae_lr_factor=config["vae_lr_factor"],
        vae_lr_patience=config["vae_lr_patience"],
    )

    data_config = DataConfig()
    num_epochs = 300

    data_module = DataModule(
        data_config.data_dir,
        batch_size=training_config.batch_size,
        num_workers=0,
    )

    # setup() has to run before train_dataset exists.
    data_module.setup()

    model = VAE(model_config=model_config, training_config=training_config, training_setup=training_setup)

    custom_checkpoint_dir = os.path.join(interim_results_dir, f"checkpoint_{trial_id}")
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    custom_checkpoint_path = os.path.join(custom_checkpoint_dir, "model.ckpt")

    if os.path.exists(custom_checkpoint_path):
        print(f"Loading checkpoint: {custom_checkpoint_path}")
        checkpoint = torch.load(custom_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=num_epochs,
        logger=False,  # no logger, print directly
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks=[SimpleCallback()],
        # profiler="simple",
        precision=32,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # skip the sanity check for a faster start
    )

    print(
        f"DataLoader: {len(data_module.train_dataset)} training, {len(data_module.val_dataset)} validation samples"
    )
    print(f"Training starts, {num_epochs} epochs")

    trainer.fit(model, data_module)

    print(f"Training finished after {trainer.current_epoch} epochs")

    try:
        metrics = {
            "loss_recon": trainer.callback_metrics["val/loss/recon"].item(),
            "silhouette": trainer.callback_metrics["val/metric/silhouette"].item(),
            "calinski_harabasz": trainer.callback_metrics["val/metric/calinski_harabasz_index"].item(),
            "davies_bouldin": trainer.callback_metrics["val/metric/davies_bouldin_index"].item(),
            "cluster_entropy": trainer.callback_metrics["val/metric/cluster_entropy"].item(),
            "smoothness": trainer.callback_metrics.get("val/metric/smoothness", torch.tensor(-1e3)).item(),
            "balance": trainer.callback_metrics.get("val/metric/balance", torch.tensor(-1e4)).item(),
        }
    except Exception:
        # Fallback metrics
        metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in {
                "loss_recon": 1e3,
                "silhouette": -1e3,
                "calinski_harabasz": -1e3,
                "davies_bouldin": 1e3,
                "cluster_entropy": -1e3,
                "smoothness": -1e3,
                "balance": -1e4,
            }.items()
        }

    # config + metrics in a single JSON file
    result = {"trial_id": trial_id, "config": config, "metrics": metrics}
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # tune.report(
    #     silhouette=metrics["silhouette"],
    #     loss_recon=metrics["loss_recon"],
    #     calinski_harabasz=metrics["calinski_harabasz"],
    #     davies_bouldin=metrics["davies_bouldin"],
    #     cluster_entropy=metrics["cluster_entropy"],
    #     smoothness=metrics["smoothness"],
    #     balance=metrics["balance"],
    # )

    tune.report(**metrics)

    return metrics


if __name__ == "__main__":
    # Konfiguration
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=int(os.environ.get("VAE_EXPERIMENT_VERSION", 1)))
    parser.add_argument("--gpus", type=int, default=0, help="Anzahl zu verwendender GPUs (0=auto)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Experiment-Setup
    EXPERIMENT_VERSION = args.version
    EXPERIMENT_NAME = f"vae_gmm_scan_v{EXPERIMENT_VERSION}"
    BASE_DIR = os.getenv("RAY_LOG_DIR", "./logs_ray")
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
        _redis_max_memory=1 * 1024 * 1024 * 1024,  # 1 GB for Redis
        object_store_memory=8 * 1024 * 1024 * 1024,  # 8GB
        local_mode=False,
        log_to_driver=True,
        ignore_reinit_error=True,
        include_dashboard=False,
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
        "vae_end_value": tune.loguniform(1e-5, 1e-2),  # 0.0001 - 0.001 depending on the run
        "gmm_end_value": tune.uniform(0.003, 0.01),
        "reg_end_value": tune.uniform(0.02, 0.5),
        "cat_end_value": tune.loguniform(0.001, 0.05),
        # Scheduler
        "vae_lr_factor": tune.uniform(0.7, 0.95),
        "vae_lr_patience": tune.choice([10, 20, 30, 40]),
        # Latent Space
        "latent_dim": tune.choice([14, 20, 30, 40, 50, 60, 70]),
        # epochs per phase
        "gmm_epochs": tune.choice([50, 80, 100, 150]),
        "vae_epochs": tune.choice([25, 30, 40, 50]),  # VAE-Training
        # extend as needed
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
            "vae_epochs": 25,
        }
    ]

    # Scheduler
    scheduler = ASHAScheduler(
        max_t=300, grace_period=120, reduction_factor=2, metric="silhouette", mode="max"
    )

    # Reporter
    reporter = tune.CLIReporter(
        parameter_columns=[
            "clustering_lr",
            "gmm_end_value",
            "reg_end_value",
            "cat_end_value",
            "gmm_epochs",
            "vae_lr_factor",
        ],
        metric_columns=[
            "loss_recon",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cluster_entropy",
            "smoothness",
            "balance",
            "training_iteration",
        ],
        max_report_frequency=60,  # Alle 60 Sekunden berichten
    )

    # Suchkonfiguration
    search_alg = OptunaSearch(
        metric=["silhouette", "loss_recon", "balance"],
        mode=["max", "min", "max"],
        points_to_evaluate=points_to_evaluate,
    )

    # multi-GPU setup
    num_samples = 256  # if gpus_to_use <= 1 else min(gpus_to_use * 10, 20)
    resources_per_trial = {"gpu": 1, "cpu": 4}
    max_concurrent = max(1, gpus_to_use)

    # run Ray Tune
    print(
        f"Start Parameter-Scan Version {EXPERIMENT_VERSION} mit {num_samples} Samples, max {max_concurrent} parallel"
    )
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
        # resume is handled by our own logic, so no resume="AUTO" here
        max_failures=2,
        max_concurrent_trials=max_concurrent,
    )

    # Ergebnisse speichern
    best_trial = result.get_best_trial("silhouette", "max")
    print(f"Bester Trial: {best_trial.trial_id}")
    print(f"Beste Metriken: {best_trial.last_result}")

    # write results as JSON
    best_results_path = os.path.join(EXPERIMENT_DIR, "best_results.json")
    with open(best_results_path, "w") as f:
        json.dump(
            {
                "config": best_trial.config,
                "metrics": {k: v for k, v in best_trial.last_result.items() if isinstance(v, (int, float))},
            },
            f,
            indent=2,
        )

    print(f"Ergebnisse gespeichert unter: {best_results_path}")
    ray.shutdown()
