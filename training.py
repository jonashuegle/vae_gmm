import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from dataset import DataModule
from VAE_GMM import VAE
from config import ModelConfig, TrainingConfig, TrainingSetup, DataConfig, HardwareConfig

import os
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import random


class SwitchShuffleCallback(pl.Callback):
    def __init__(self, switch_epoch, new_shuffle_value):
        """
        :param switch_epoch: Die Epoche, ab der das Shuffle umgeschaltet werden soll.
        :param new_shuffle_value: Bool, ob ab diesem Zeitpunkt gemischt werden soll.
        """
        self.switch_epoch = switch_epoch
        self.new_shuffle_value = new_shuffle_value

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.switch_epoch:
            print(f"Switching training dataloader shuffle to {self.new_shuffle_value} at epoch {trainer.current_epoch}")
            # Setze das Flag im Datamodule
            trainer.datamodule.shuffle_train = self.new_shuffle_value
            # Falls verfügbar: den Dataloader neu laden (Lightning 2.x)
            if hasattr(trainer, "reset_train_dataloader"):
                trainer.reset_train_dataloader()


class SpecificEpochCheckpoint(Callback):
    def __init__(self, save_epochs, dirpath, filename_template="epoch={epoch:02d}-step={global_step}.ckpt"):
        """
        speichert Checkpoints, wenn die aktuelle Epoche in save_epochs enthalten ist.
        :param save_epochs: Liste oder Menge von Epochen, z. B. [25, 80, 120, 160]
        :param dirpath: Verzeichnis, in dem die Checkpoints gespeichert werden.
        :param filename_template: Namensvorlage für den Checkpoint.
        """
        self.save_epochs = set(save_epochs)
        self.dirpath = dirpath
        self.filename_template = filename_template

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.save_epochs:
            filename = self.filename_template.format(epoch=epoch, global_step=trainer.global_step)
            ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            print(f"Saved specific checkpoint at epoch {epoch} to {ckpt_path}")



def get_latest_version(base_path):
    versions = sorted(glob.glob(os.path.join(base_path, 'version_*')), key=lambda x: int(x.split('_')[-1]))
    if versions:
        return int(versions[-1].split('_')[-1])
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for VAE.')
    parser.add_argument('--resume', type=bool, default=False, help='Flag to resume training from a checkpoint.') # --resume True
    parser.add_argument('--version', type=int, default=None, help='Version of the experiment for logging.') # --version 0
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of epochs for training.') # --max_epochs 100
    parser.add_argument('--gmm_end_value', type=float, default=0.005220209, help='End value for GMM training.') # --gmm_end_value 0.005220209
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Setze den Seed für Reproduzierbarkeit
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**31)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

    # Konfiguration laden
    model_config = ModelConfig(
        layer_sizes=(4000, 2000, 800, 200, 100, 14),
        num_clusters=5,
        )

    # training_config = TrainingConfig(
    #     vae_lr=0.000193066,
    #     clustering_lr= 5.929e-06,
    #     batch_size=400,
    #     vae_end_value=0.001,
    #     gmm_end_value= args.gmm_end_value,#0.005220209,
    #     reg_end_value=0.385072058,#0.04072058,
    #     cat_end_value=0.005362321,
    #     log_scaled=True,
    #     seed = seed,
    #     )
    

    # training_setup = TrainingSetup(
    #     # Phase I: reines VAE-Warmup (Rekon + globaler KLD-Ramp)
    #     warmup_epochs     = 25,
    #     vae_epochs        = 25,
    #     adapt_epochs      = 15,   # 20-25, je nach Datensatz
    #     # Phase II: Clustering
    #     kmeans_init_epoch = 25,   # direkt nach Warmup
    #     gmm_epochs        = 80,   # kürzer, stabilisiert
    #     cat_epochs        = 250,
    #     reg_epochs        = 250,

    #     # Da LambdaLR/Warmup und LinearLR im Wesentlichen auf warmup_epochs  
    #     # abzielen, können wir die alten Parameter entfernen:
    #     clustering_warmup = 25,   # optional, aber = warmup_epochs
    #     linear_epochs     = 25,   # optional → gleich warmup_epochs 

    #     # verbleibende Scheduler-Settings unverändert:
    #     annealing_type    = "linear",
    #     cosine_T_max      = 400,
    #     cosine_eta_min    = 1.2e-08,
    #     vae_lr_factor     = 0.777187766,
    #     vae_lr_patience   = 30,
    # )

    training_config = TrainingConfig(
        # Lernraten
        vae_lr          = 0.002118,   # aus Run
        clustering_lr   = 0.000335,   # aus Run

        # Gewichtungen
        recon_weight    = 0.093868,   # aus Run
        kld_weight      = 0.000107,   # aus Run

        # Endwerte für Scheduling der Loss-Terme
        vae_end_value   = 0.001,      # unverändert
        gmm_end_value   = 0.006894,   # aus Run
        reg_end_value   = 0.403404,   # aus Run
        cat_end_value   = 0.011894,   # aus Run

        # Sonstiges
        log_scaled      = True,       # unverändert
        seed            = seed,       # unverändert
    )

    training_setup = TrainingSetup(
        # Phase I: reines VAE-Warmup (Rekon + globaler KLD-Ramp)
        warmup_epochs     = 20,       # aus Run
        vae_epochs        = 25,       # unverändert
        adapt_epochs      = 20,       # aus Run

        # Phase II: Clustering
        kmeans_init_epoch = 20,       # = warmup_epochs
        gmm_epochs        = 50,       # aus Run
        cat_epochs        = 250,      # unverändert
        reg_epochs        = 250,      # unverändert

        # Scheduler-Parametrierung (am warmup orientiert)
        clustering_warmup = 20,       # = warmup_epochs
        linear_epochs     = 20,       # = warmup_epochs

        # restliche Scheduler-Settings
        annealing_type    = "linear", # unverändert
        cosine_T_max      = 400,      # unverändert
        cosine_eta_min    = 1.2e-08,  # unverändert

        # VAE-LR-Scheduler
        vae_lr_factor     = 0.775108, # aus Run
        vae_lr_patience   = 30,       # aus Run
    )

    data_config = DataConfig(
        data_dir='/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc',
        log_dir='/work/aa0238/a271125/logs/StableVAE',
        experiment='Exp_test_smoothing',
        num_workers=64,
        )
    hardware_config = HardwareConfig(devices=(0,))


    # Initialisieren des DataModules
    data_module = DataModule(data_config.data_dir, batch_size=training_config.batch_size, num_workers=data_config.num_workers)

    # Logging-Pfade und Version bestimmen
    base_path = f'{data_config.log_dir}/{data_config.experiment}/'

    # Version bestimmen
    if args.version is None:
        latest_version = get_latest_version(base_path)
        if args.resume:
            args.version = latest_version if latest_version is not None else 0
            args.resume = latest_version is not None
        else:
            args.version = (latest_version + 1) if latest_version is not None else 0

    logger = TensorBoardLogger(save_dir=f"{data_config.log_dir}/", name=data_config.experiment, version=args.version)

    # Checkpoint-Pfad
    path = os.path.join(base_path, f'version_{args.version}/checkpoints/')

    
    log_files = glob.glob(os.path.join(path, '*.ckpt'))
    log_files.sort(key=os.path.getmtime)

    max_epochs = args.max_epochs

    # Callbacks
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=path,
        save_last=True,
        save_top_k=0,
        verbose=True,
    )

    specific_checkpoint_callback = SpecificEpochCheckpoint(
        save_epochs=[
            training_setup.kmeans_init_epoch,
            training_setup.warmup_epochs+training_setup.vae_epochs,
            training_setup.warmup_epochs+training_setup.vae_epochs+training_setup.adapt_epochs,
            training_setup.warmup_epochs+training_setup.vae_epochs+training_setup.adapt_epochs+20,
            training_setup.warmup_epochs+training_setup.vae_epochs+training_setup.adapt_epochs+40,
            ],
        dirpath=path,
        filename_template="epoch={epoch:02d}-step={global_step}.ckpt"
    )

    switch_callback = SwitchShuffleCallback(switch_epoch=training_setup.kmeans_init_epoch, new_shuffle_value=False)
    
    if training_config.seed is None:
        raise ValueError("Seed must be set in the training configuration.")

    # Trainer initialisieren
    trainer = pl.Trainer(
        accelerator=hardware_config.accelerator,
        devices=hardware_config.devices,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        callbacks=[last_checkpoint_callback, specific_checkpoint_callback, switch_callback],
        #callbacks=[EarlyStopping(monitor="val/loss/recon", patience=50, min_delta=0.00, mode='min')],
        logger=logger,
        precision=32,
    )

    # Training starten oder fortsetzen
    if args.resume and len(log_files) > 0:
        checkpoint_path = log_files[-1]
        print(f"Checkpoint {checkpoint_path} loaded.")

        # Modell aus Checkpoint laden
        autoencoder = VAE.load_from_checkpoint(checkpoint_path)
        
        # Fortsetzen des Trainings
        trainer.fit(autoencoder, data_module, ckpt_path=checkpoint_path)

    else:
        print("No checkpoint found. Starting new training.")
        # Neues Modell initialisieren
        autoencoder = VAE(
            model_config = model_config,
            training_config = training_config,
            training_setup = training_setup,
        )

        # Neues Training starten
        trainer.fit(autoencoder, data_module)
