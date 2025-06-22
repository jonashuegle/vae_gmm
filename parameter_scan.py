import pytorch_lightning as pl
import torch
import numpy as np

import ray
from ray import tune
from ray import train
from ray.air import session

from ray.tune.tune_config import TuneConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Stopper


import os
import json
import random
import time

from dataset import DataModule
from config import ModelConfig, TrainingConfig, TrainingSetup, DataConfig, HardwareConfig
from VAE_GMM import VAE
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')


class IdleTrialStopper(Stopper):
    """Stoppe Trials, die länger als max_idle_s keine Fortschritts-Metrik reporten."""
    def __init__(self, max_idle_s: float):
        self.max_idle_s = max_idle_s
        # Pro Trial speichern wir den Zeitpunkt der letzten echten Report-Metrik
        self._last_report = {}  # trial_id -> (last_iter, timestamp)

    def __call__(self, trial_id: str, result: dict) -> bool:
        now = time.time()
        it = result.get("training_iteration", None)
        if trial_id not in self._last_report:
            # Erster Report: initialisieren
            self._last_report[trial_id] = (it, now)
            return False
        last_it, last_t = self._last_report[trial_id]
        if it is not None and it > last_it:
            # Iteration hat sich erhöht: updaten
            self._last_report[trial_id] = (it, now)
            return False
        # Keine neue Iteration: prüfen, ob Idle-Zeit überschritten
        if now - last_t > self.max_idle_s:
            print(f"⏱️ Trial {trial_id} idle for {now-last_t:.1f}s > {self.max_idle_s}s → stopping")
            return True
        return False

    def stop_all(self) -> bool:
        return False
class CombinedStopper(Stopper):
    def __init__(self, max_iter, max_total_s, max_idle_s):
        self.max_iter     = max_iter
        self.max_total_s  = max_total_s
        self.idle_stopper = IdleTrialStopper(max_idle_s)
        self.start_times  = {}  # für time_total_s

    def __call__(self, trial_id, result):
        now = time.time()
        # initialisiere Startzeit
        if trial_id not in self.start_times:
            self.start_times[trial_id] = now

        # 1) Gesamtlaufzeit überschritten?
        if now - self.start_times[trial_id] >= self.max_total_s:
            return True

        # 2) Max-Iterationen erreicht?
        if result.get("training_iteration", 0) >= self.max_iter:
            return True

        # 3) Idle-Timeout erreicht?
        return self.idle_stopper(trial_id, result)

    def stop_all(self):
        return False



class EvaluationCallback(TuneReportCallback):
    def on_validation_end(self, trainer, pl_module):
        try:
            metrics = {
                # Achte darauf, dass die Keys exakt mit denen übereinstimmen, 
                # die du in self.log(...) gesetzt hast
                "loss_recon": trainer.callback_metrics["val/loss/recon"].item(),
                "silhouette": trainer.callback_metrics["val/metric/silhouette"].item(),
                "calinski_harabasz": trainer.callback_metrics["val/metric/calinski_harabasz_index"].item(),
                "davies_bouldin": trainer.callback_metrics["val/metric/davies_bouldin_index"].item(),
                "cluster_entropy": trainer.callback_metrics["val/metric/cluster_entropy"].item()
            }
            print(f"Reporting metrics: {metrics}")
            session.report(metrics)
        except KeyError as e:
            print(f"Missing metric in callback_metrics: {e}")
            print(f"Available metrics: {trainer.callback_metrics.keys()}")
            raise
        except Exception as e:
            print(f"Error in callback: {e}")
            raise



def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Beachte: Vorzeichen für max/min Optimierung
            dominates = np.any((-costs[is_efficient] > -c), axis=1)
            is_efficient[is_efficient] = dominates
            is_efficient[i] = True
    return is_efficient


def train_vae(config, num_epochs=None, pretrained_path=None):
    try:
        model_config = ModelConfig()
        training_config = TrainingConfig(
            clustering_lr=config["clustering_lr"],
            gmm_end_value=config["gmm_end_value"],
            reg_end_value=config["reg_end_value"],
            cat_end_value=config["cat_end_value"],
            seed=42
        )
        training_setup = TrainingSetup(
            gmm_epochs=config["gmm_epochs"],
            cosine_eta_min=config["cosine_eta_min"],
            vae_lr_factor=config["vae_lr_factor"],
            vae_lr_patience=config["vae_lr_patience"],
        )
    
        data_config = DataConfig()

        # Epochen festlegen
        num_epochs = training_setup.warmup_epochs + training_setup.vae_epochs + training_setup.adapt_epochs+50
        
        data_module = DataModule(data_config.data_dir, batch_size=training_config.batch_size, num_workers=8)

        model = VAE(model_config=model_config, training_config=training_config, training_setup=training_setup)

        # Vortrainiertes Modell laden, falls Pfad angegeben
        if pretrained_path is not None and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

        logger = TensorBoardLogger(save_dir='/work/aa0238/a271125/logs_ray/vae_gmm',
                                   name="multi_objective_scan", 
                                   version=3)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=logger,
            callbacks=[EvaluationCallback()],
            precision=32,
            enable_progress_bar=False,
        )

        trainer.fit(model, data_module)

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        session.report({
            "loss_recon": 1e3,
            "silhouette": -1e3,
            "calinski_harabasz": -1e3,
            "davies_bouldin": 1e3,
            "cluster_entropy": -1e3
        })



if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    gpus = int(os.environ.get("SLURM_GPUS_PER_NODE", 0))

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,

        _memory=32 * 1024 * 1024 * 1024,  # 32GB RAM-Limit
        _redis_max_memory=1 * 1024 * 1024 * 1024,  # 1GB für Redis
        object_store_memory=4 * 1024 * 1024 * 1024,  # 8GB statt 4GB
        
        # Process Management
        # temp_dir="/work/aa0238/a271125/ray_tmp",  # Entfernt - nicht unterstützt
        local_mode=False,  # Wichtig: explizit auf False für echte Parallelisierung
        
        # Debugging und Fehlerbehandlung
        log_to_driver=True,  # Aktiviere Logging (wichtig für Fehlerdiagnose)
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    scheduler = ASHAScheduler(
        max_t=200,
        grace_period=120,
        reduction_factor=2,
        metric="silhouette",  # Primäre Metrik für den Scheduler
        mode="max"         # Modus für diese Metrik
    )

    reporter = tune.CLIReporter(
        parameter_columns=[
            "clustering_lr",
            "gmm_end_value",
            "reg_end_value",
            "cat_end_value",
            "gmm_epochs",
            "cosine_eta_min",
            "vae_lr_factor",
            "vae_lr_patience",
        ],
        metric_columns=[
            "loss_recon",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cluster_entropy",
            "training_iteration",
        ]
     )

    stopper = CombinedStopper(
        max_iter=200,           # z.B. Ende bei 200 Epochen
        max_total_s=7*3600,     # z.B. 7 Stunden
        max_idle_s=600          # z.B. 10 Minuten
    )


    search_space = {
        "clustering_lr": tune.uniform(1e-06, 1e-05),          # Niedriger Bereich, um den Effekt niedriger LR zu untersuchen
        "gmm_end_value": tune.uniform(0.0045, 0.0055),          # Rund 0.005
        "reg_end_value": tune.uniform(0.35, 0.45),              # Zwischen 0.33 und 0.38
        "cat_end_value": tune.uniform(0.0020, 0.1),          # Sehr schmaler Bereich
        "gmm_epochs": tune.choice([60, 80, 100, 120]),
        "cosine_eta_min": tune.loguniform(1e-8, 3e-8),
        "vae_lr_factor": tune.uniform(0.75, 0.85),
        "vae_lr_patience": tune.choice([20,25,30]),
}




    path = '/work/aa0238/a271125/logs_ray/vae_gmm_multi_objective_scan/version_4'
    os.environ["RAY_RESULTS_DIR"] = path

    result = tune.run(
        train_vae,
        search_alg=OptunaSearch(
            metric=["loss_recon", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_entropy"],
            mode=["min", "max", "max", "min", "max"],
            points_to_evaluate=[{
                "clustering_lr":  5.928647e-06,
                "gmm_end_value": 5.220209e-03,
                "reg_end_value": 3.850721e-01,
                "cat_end_value": 5.362321e-03,
                "gmm_epochs": 80,
                "cosine_eta_min":  1.2e-08,
                "vae_lr_factor":  0.777187766,
                "vae_lr_patience": 30,
            }],
        ),
        num_samples=54,
        resources_per_trial={"gpu": 1},
        scheduler=scheduler,
        progress_reporter=reporter,
        name="vae_gmm",
        local_dir=path,
        config= search_space,
        resume="AUTO",
        stop=stopper,
        raise_on_failed_trial=False,
        max_failures=1,
    )

    # Kostenmatrix erstellen
    all_results = []
    for trial in result.trials:
        last_result = trial.last_result
        if "loss_recon" in last_result and "silhouette" in last_result and "calinski_harabasz" in last_result and "davies_bouldin" in last_result and "cluster_entropy" in last_result:
            all_results.append([
                last_result["loss_recon"],
                last_result["silhouette"],
                last_result["calinski_harabasz"],
                last_result["davies_bouldin"],
                last_result["cluster_entropy"],
            ])
        else:
            print(f"Skipping trial {trial.trial_id} due to missing metrics.")
    costs = np.array(all_results)

    # Pareto-Effizienz berechnen
    pareto_mask = is_pareto_efficient(costs)
    pareto_trials = np.array(result.trials)[pareto_mask]

    # Pareto-optimalen Ergebnisse speichern
    best_results = []
    for trial in pareto_trials:
        best_results.append({
            "config": trial.config,
            "metrics": {
                "loss_recon": trial.last_result["loss_recon"],
                "silhouette": trial.last_result["silhouette"],
                "calinski_harabasz": trial.last_result["calinski_harabasz"],
                "davies_bouldin": trial.last_result["davies_bouldin"],
                "cluster_entropy": trial.last_result["cluster_entropy"],
            }
        })

    best_model_dir = os.path.join('/work/aa0238/a271125/logs_ray/vae_gmm/version_4/best_results', 'pareto_optimal_results')
    os.makedirs(best_model_dir, exist_ok=True)

    with open(os.path.join('/work/aa0238/a271125/logs_ray/vae_gmm/version_4/best_results', 'pareto_results.json'), "w") as f:
        json.dump(best_results, f, indent=4)

    ray.shutdown()