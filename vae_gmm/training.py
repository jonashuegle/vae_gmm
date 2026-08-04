import argparse
import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

from vae_gmm.config import (
    DataConfig,
    HardwareConfig,
    ModelConfig,
    TrainingConfig,
    TrainingSetup,
)
from vae_gmm.dataset import DataModule
from vae_gmm.VAE_GMM import VAE


class SwitchShuffleCallback(pl.Callback):
    """
    Callback to switch the shuffle state of the training dataloader at a specific epoch.

    """

    def __init__(self, switch_epoch, new_shuffle_value):
        """
        :param switch_epoch: The epoch at which to switch the shuffle state.
        :param new_shuffle_value: Bool, whether to shuffle from this point onwards.
        """
        self.switch_epoch = switch_epoch
        self.new_shuffle_value = new_shuffle_value

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.switch_epoch:
            print(
                f"Switching training dataloader shuffle to {self.new_shuffle_value} at epoch {trainer.current_epoch}"
            )
            # The DataModule reads this flag when building the loader.
            trainer.datamodule.shuffle_train = self.new_shuffle_value
            # Lightning 2.x only
            if hasattr(trainer, "reset_train_dataloader"):
                trainer.reset_train_dataloader()


class SpecificEpochCheckpoint(Callback):
    """
    Saves checkpoints at specific epochs during training.
    """

    def __init__(self, save_epochs, dirpath, filename_template="epoch={epoch:02d}-step={global_step}.ckpt"):
        """
        :param save_epochs: List or set of epochs, e.g. [25, 80, 120, 160]
        :param dirpath: Directory where the checkpoints will be saved.
        :param filename_template: Filename template for the checkpoint.
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
    """
    Finds the latest version directory in the specified base path.
    Args:
        base_path (str): The base path where version directories are stored.
    Returns:
        int: The latest version number found in the base path.
    """
    versions = sorted(glob.glob(os.path.join(base_path, "version_*")), key=lambda x: int(x.split("_")[-1]))
    if versions:
        return int(versions[-1].split("_")[-1])
    else:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training script for VAE.")
    parser.add_argument(
        "--version", type=int, default=None, help="Version of the experiment for logging."
    )  # --version 0
    parser.add_argument(
        "--max_epochs", type=int, default=400, help="Maximum number of epochs for training."
    )  # --max_epochs 100
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument(
        "--find-lr", action="store_true", help="Run the learning rate finder instead of training."
    )

    args = parser.parse_args(argv)

    # Generate random seed if not provided
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**31)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_config = ModelConfig()
    training_config = TrainingConfig(seed=seed)
    training_setup = TrainingSetup()
    data_config = DataConfig()
    hardware_config = HardwareConfig()

    data_module = DataModule(
        data_config.data_dir, batch_size=training_config.batch_size, num_workers=data_config.num_workers
    )

    # Learning rate finder
    if args.find_lr:
        model = VAE(
            model_config=model_config,
            training_config=training_config,
            training_setup=training_setup,
        )
        lr_trainer = pl.Trainer(
            accelerator=hardware_config.accelerator,
            devices=hardware_config.devices,
            max_epochs=1,
            enable_progress_bar=True,
            logger=False,
        )
        lr_finder = Tuner(lr_trainer).lr_find(
            model, datamodule=data_module, min_lr=1e-7, max_lr=1e-3, num_training=800
        )
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        return

    # Path setup for logging and checkpoints
    base_path = f"{data_config.log_dir}/{data_config.experiment}/"

    # Version handling
    if args.version is None:
        latest_version = get_latest_version(base_path)
        if args.resume:
            args.version = latest_version if latest_version is not None else 0
            args.resume = latest_version is not None
        else:
            args.version = (latest_version + 1) if latest_version is not None else 0

    logger = TensorBoardLogger(
        save_dir=f"{data_config.log_dir}/", name=data_config.experiment, version=args.version
    )

    path = os.path.join(base_path, f"version_{args.version}/checkpoints/")

    # Get all existing checkpoints in the directory for resuming training
    log_files = glob.glob(os.path.join(path, "*.ckpt"))
    log_files.sort(key=os.path.getmtime)

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
            training_setup.warmup_epochs + training_setup.vae_epochs,
            training_setup.warmup_epochs + training_setup.vae_epochs + training_setup.adapt_epochs,
            training_setup.warmup_epochs + training_setup.vae_epochs + training_setup.adapt_epochs + 20,
            training_setup.warmup_epochs + training_setup.vae_epochs + training_setup.adapt_epochs + 40,
        ],
        dirpath=path,
        filename_template="epoch={epoch:02d}-step={global_step}.ckpt",
    )

    # Callback to switch the shuffle state of the training dataloader at the kmeans initialization epoch
    # (impact has to be further investigated)
    switch_callback = SwitchShuffleCallback(
        switch_epoch=training_setup.kmeans_init_epoch, new_shuffle_value=False
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=hardware_config.accelerator,
        devices=hardware_config.devices,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        callbacks=[last_checkpoint_callback, specific_checkpoint_callback, switch_callback],
        # callbacks=[EarlyStopping(monitor="val/loss/recon", patience=50, min_delta=0.00, mode='min')],
        logger=logger,
        precision=32,
    )

    # Start or resume training
    if args.resume and len(log_files) > 0:
        checkpoint_path = log_files[-1]
        print(f"Checkpoint {checkpoint_path} loaded.")

        # Load model from checkpoint
        autoencoder = VAE.load_from_checkpoint(checkpoint_path)

        # Resume training
        trainer.fit(autoencoder, data_module, ckpt_path=checkpoint_path)

    else:
        print("No checkpoint found. Starting new training.")
        # Initialize new model
        autoencoder = VAE(
            model_config=model_config,
            training_config=training_config,
            training_setup=training_setup,
        )

        # Start new training
        trainer.fit(autoencoder, data_module)


if __name__ == "__main__":
    main()
