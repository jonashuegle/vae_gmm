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

import os
import json
import random

from dataset import DataModule
from config_vade import ModelConfig, TrainingConfig, TrainingSetup, DataConfig, HardwareConfig
from VaDE_kmeans_init import VaDE
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

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


def train_vade(config, num_epochs=None, pretrained_path=None):
    try:
        model_config = ModelConfig()
        training_config = TrainingConfig(
            clustering_lr=config["clustering_lr"],
            gmm_end_value=config["gmm_end_value"],
            reg_end_value=config["reg_end_value"],
            cat_end_value=config["cat_end_value"],
        )
        training_setup = TrainingSetup(
            gmm_epochs=config["gmm_epochs"],
            cat_epochs=config["cat_epochs"],
            reg_epochs=config["reg_epochs"],
            vae_epochs=config["vae_epochs"],
            warmup_epochs=config["warmup_epochs"],
            kmeans_init_epoch=config["kmeans_init_epoch"],
            clustering_warmup=config["clustering_warmup"],
            linear_epochs=config["linear_epochs"],
            cosine_eta_min=config["cosine_eta_min"],
            vae_lr_factor=config["vae_lr_factor"],
            vae_lr_patience=config["vae_lr_patience"],
        )
    
        data_config = DataConfig()

        # Epochen festlegen
        num_epochs = training_setup.warmup_epochs + training_setup.vae_epochs + training_setup.adapt_epochs+50
        
        data_module = DataModule(data_config.data_dir, batch_size=training_config.batch_size, num_workers=64)

        model = VaDE(model_config=model_config, training_config=training_config, training_setup=training_setup)

        # Vortrainiertes Modell laden, falls Pfad angegeben
        if pretrained_path is not None and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

        logger = TensorBoardLogger(save_dir='/work/aa0238/a271125/logs_ray/full_multi_vade',
                                   name="multi_objective_scan", 
                                   version=0)

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



    ray.init(num_gpus=4, num_cpus=256)

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
            "vae_epochs",
            "gmm_epochs",
            "cat_epochs",
            "reg_epochs",
            "warmup_epochs",
            "kmeans_init_epoch",
            "clustering_warmup",
            "linear_epochs",
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

    search_space = {
        "clustering_lr": tune.uniform(1e-06, 1e-05),          # Niedriger Bereich, um den Effekt niedriger LR zu untersuchen
        "gmm_end_value": tune.uniform(0.0045, 0.0055),          # Rund 0.005
        "reg_end_value": tune.uniform(0.35, 0.45),              # Zwischen 0.33 und 0.38
        "cat_end_value": tune.uniform(0.0020, 0.1),          # Sehr schmaler Bereich
        "vae_epochs": tune.choice([15, 20, 25, 30, 35]),
        "gmm_epochs": tune.choice([80, 100, 120, 150, 200]),
        "cat_epochs": tune.choice([200, 220, 240, 250, 260, 270]),
        "reg_epochs": tune.choice([250, 280, 300, 310]),
        "warmup_epochs": tune.choice([20, 25, 30, 35, 40]),
        "kmeans_init_epoch": tune.choice([10, 15]),
        "clustering_warmup": tune.choice([15, 20, 25, 30, 35]),
        "linear_epochs": tune.choice([50, 60, 70, 80, 90]),
        "cosine_eta_min": tune.loguniform(1e-8, 3e-8),
        "vae_lr_factor": tune.uniform(0.75, 0.85),
        "vae_lr_patience": tune.choice([15, 18, 20, 22, 25, 30]),
}




    path = '/work/aa0238/a271125/logs_ray/vae_gmm_multi_objective_scan/version_2'
    os.environ["RAY_RESULTS_DIR"] = path

    result = tune.run(
        train_vade,
        search_alg=OptunaSearch(
            metric=["loss_recon", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_entropy"],
            mode=["min", "max", "max", "min", "max"],
            points_to_evaluate=[{
                "clustering_lr":  5.928647e-06,
                "gmm_end_value": 5.220209e-03,
                "reg_end_value": 3.850721e-01,
                "cat_end_value": 5.362321e-03,
                "vae_epochs": 25,
                "gmm_epochs": 80,
                "cat_epochs": 250,
                "reg_epochs": 250,
                "warmup_epochs": 25,
                "kmeans_init_epoch": 25,
                "clustering_warmup": 25,
                "linear_epochs": 25,
                "cosine_eta_min":  1.2e-08,
                "vae_lr_factor":  0.777187766,
                "vae_lr_patience": 30,
            }],
        ),
        num_samples=96,
        resources_per_trial={"gpu": 0.25},
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_vade_multi_objective",
        local_dir=path,
        config= search_space,
        #resume=True,
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

    best_model_dir = os.path.join('/work/aa0238/a271125/logs_ray/full_multi_vade/version_2/best_results', 'pareto_optimal_results')
    os.makedirs(best_model_dir, exist_ok=True)

    with open(os.path.join('/work/aa0238/a271125/logs_ray/full_multi_vade/version_2/best_results', 'pareto_results.json'), "w") as f:
        json.dump(best_results, f, indent=4)

    ray.shutdown()