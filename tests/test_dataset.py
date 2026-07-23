from vae_gmm.dataset import CustomDataset
import numpy as np


def test_dataset_length(make_netcdf):
    ds = CustomDataset(make_netcdf(n_time_steps=4))
    assert len(ds) == 4

def test_dataset_item(make_netcdf):
    ds = CustomDataset(make_netcdf(n_time_steps=4))
    tensor, timestamp = ds[0]
    assert np.array(tensor).shape == (1, 61, 181)  # lat x lon
    