# testing annealing process for VAE-GMM model
from vae_gmm.VAE_GMM import lr_lambda

def test_lr_lambda_is_zero_during_warmup():
    assert lr_lambda(epoch=1, warmup_epochs=5, linear_epochs=10) == 0.0


def test_lr_lambda_is_one_after_annealing():
    assert lr_lambda(epoch=16, warmup_epochs=5, linear_epochs=10) == 1.0


def test_lr_lambda_is_linear_during_annealing():
    assert lr_lambda(epoch=10, warmup_epochs=5, linear_epochs=10) == 0.5