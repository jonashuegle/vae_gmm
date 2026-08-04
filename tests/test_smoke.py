import pytorch_lightning as pl
import torch

from vae_gmm.config import TrainingConfig, TrainingSetup
from vae_gmm.dataset import DataModule
from vae_gmm.VAE_GMM import VAE


def test_smoke(tiny_config, make_netcdf):

    data_file = make_netcdf(n_time_steps=256)
    data_module = DataModule(data_file, batch_size=16, num_workers=0)

    model = VAE(tiny_config, TrainingConfig(), TrainingSetup())

    trainer = pl.Trainer(accelerator="cpu", devices=1, fast_dev_run=True)
    trainer.fit(model, datamodule=data_module)

    loss = trainer.callback_metrics.get("train/loss/total")
    assert loss is None or torch.isfinite(loss)

    print(trainer.callback_metrics)
