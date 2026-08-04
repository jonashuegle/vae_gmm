from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def make_netcdf(tmp_path: Path) -> Callable[..., str]:
    """Create a dummy netcdf file for testing."""

    def _make(n_time_steps: int = 5, constant: float | None = None, seed: int = 42):

        times = xr.date_range(
            "2000-02-15", periods=n_time_steps, freq="6h", calendar="noleap", use_cftime=True
        )

        lat = np.arange(30, 91, 1)
        lon = np.arange(-90, 91, 1)

        rng = np.random.default_rng(seed)
        data = 1013.0 + rng.normal(loc=0.0, scale=10.0, size=(n_time_steps, len(lat), len(lon)))

        if constant is not None:
            data[:] = constant

        ds = xr.Dataset(
            {"MSL": (["time", "lat", "lon"], data)},
            coords={
                "time": times,
                "lat": lat,
                "lon": lon,
            },
        )

        path = tmp_path / "slp.nc"
        ds.to_netcdf(path)

        return str(path)

    return _make


@pytest.fixture
def tiny_config():
    from vae_gmm.config import ModelConfig

    config = ModelConfig(
        input_shape=(1, 61, 181),
        layer_sizes=(64, 16, 4),
        dropout_prob=(0.0, 0.0),
        num_clusters=2,
    )

    return config
