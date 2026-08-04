import torch

from vae_gmm.config import TrainingConfig, TrainingSetup
from vae_gmm.VAE_GMM import VAE


def _create_model(tiny_config):
    model = VAE(tiny_config, TrainingConfig(), TrainingSetup())
    return model


def test_model_dimensions(tiny_config):
    model = _create_model(tiny_config)
    x = torch.randn(2, *tiny_config.input_shape)

    x_recon, mu, logvar, z = model(x)

    assert x_recon.shape == x.shape
    assert mu.shape == (2, tiny_config.layer_sizes[-1])
    assert logvar.shape == (2, tiny_config.layer_sizes[-1])
    assert z.shape == (2, tiny_config.layer_sizes[-1])
