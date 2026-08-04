import numpy as np

from vae_gmm.dataset import CustomDataset


def test_dataset_length(make_netcdf):
    ds = CustomDataset(make_netcdf(n_time_steps=4))
    assert len(ds) == 4


def test_dataset_item(make_netcdf):
    ds = CustomDataset(make_netcdf(n_time_steps=4))
    tensor, timestamp = ds[0]
    assert np.array(tensor).shape == (1, 61, 181)  # lat x lon


def test_no_nans_with_constant_field(make_netcdf):
    ds = CustomDataset(make_netcdf(n_time_steps=4, constant=1013.0), drop_pol=False)
    tensor, timestamp = ds[0]
    assert not np.isnan(tensor).any()
