import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from vae_gmm.dataset import CustomDataset
from vae_gmm.plotting import Plotting


class PatternReferenceManager:
    """
    Builds and manages reference clusters (PCA + k-means) on NetCDF data.
    - sqrt(cos(lat)) weighting via CustomDataset
    - caching of labels and patterns
    - optional manual naming through a mapping
    - plotting delegated to the Plotting class
    """

    def __init__(
        self,
        nc_file_path: str,
        var_name: str = "MSL",
        n_pcs: int = 14,
        n_clusters: int = 5,
        cache_dir: str = "cache_pattern_ref",
        sqrt: bool = True,
        drop_pol: bool = False,
        random_state: int = 42,
    ):
        self.nc_file_path = nc_file_path
        self.var_name = var_name
        self.n_pcs = n_pcs
        self.n_clusters = n_clusters
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.stem = Path(nc_file_path).stem
        self.f_labels = self.cache_dir / f"{self.stem}_kmeans_labels.npy"
        self.f_pattern = self.cache_dir / f"{self.stem}_pattern.nc"
        self.f_mapping = self.cache_dir / f"{self.stem}_manual_mapping.json"

        self.patterns: xr.DataArray | None = None
        self.labels: np.ndarray | None = None
        self.mapping: dict[int, str] | None = None

        self._load_or_compute_patterns()
        if self.f_mapping.exists():
            self.load_mapping()
        else:
            # Without a mapping the clusters are plotted unnamed.
            print("[PatternReferenceManager] Kein Mapping gefunden - Cluster werden mit Nummern geplottet.")
            self.plot_clusters()

    def _load_or_compute_patterns(self):
        """Load patterns and labels from the cache, or recompute them."""
        try:
            if self.f_pattern.exists() and self.f_labels.exists():
                self.patterns = xr.open_dataarray(self.f_pattern)
                self.labels = np.load(self.f_labels)
            else:
                print("[PatternReferenceManager] Kein Cache gefunden, berechne Patterns ...")
                self.fit()
                self.save_patterns()
        except Exception as e:
            print(f"Fehler beim Laden des Caches: {e}. Berechne neu ...")
            self.fit()
            self.save_patterns()

    def fit(self):
        """
        Compute PCA + k-means on the reference data.
        Results: self.patterns (xarray), self.labels (numpy)
        """
        # sqrt latitude weighting is applied by CustomDataset.
        dataset = CustomDataset(
            nc_file_path=self.nc_file_path,
            sqrt=True,
            save_ram=False,
            drop_pol=True,
        )

        msl = np.asarray(dataset.data[self.var_name])
        n_samples = msl.shape[0]
        n_features = np.prod(msl.shape[1:])
        msl_flat = msl.reshape(n_samples, n_features)

        pca = PCA(n_components=self.n_pcs, random_state=self.random_state)
        pca_results = pca.fit_transform(msl_flat)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100, random_state=self.random_state)
        labels = kmeans.fit_predict(pca_results)

        # One composite per cluster, kept as an xarray object.
        ds = xr.open_dataset(self.nc_file_path)
        slp = ds[self.var_name].isel(time=slice(0, len(labels)))
        slp = slp.assign_coords(cluster=("time", labels))
        patterns = slp.groupby("cluster").mean("time").sortby("cluster")

        self.patterns = patterns
        self.labels = labels

    def save_patterns(self):
        """Cache the patterns (xarray) and labels (npy)."""
        assert self.patterns is not None and self.labels is not None
        self.patterns.to_netcdf(self.f_pattern)
        np.save(self.f_labels, self.labels)

    def delete_cache(self):
        """Delete the cached patterns, labels and mapping."""
        for f in [self.f_pattern, self.f_labels, self.f_mapping]:
            if os.path.exists(f):
                os.remove(f)
        print("PatternReferenceManager: Cache gelöscht.")

    def plot_clusters(
        self, plotter: Plotting | None = None, titles: list[str] | None = None, show_colorbar=True
    ):
        """
        Plot the cluster composites.
        """
        if self.patterns is None:
            raise ValueError("Keine Patterns gefunden!")
        if plotter is None:
            lon = self.patterns["lon"].values
            lat = self.patterns["lat"].values
            plotter = Plotting(lon=lon, lat=lat)
        if titles is None:
            if self.mapping is not None:

                def _to_int_if_possible(val):
                    try:
                        return int(val)
                    except Exception:
                        return val

                titles = [
                    self.mapping.get(_to_int_if_possible(i), str(i)) for i in self.patterns.cluster.values
                ]
            else:
                titles = [str(i) for i in self.patterns.cluster.values]

        plotter.plot_isolines(self.patterns.values, titles=titles, show_colorbar=show_colorbar)

    def save_mapping(self, mapping: dict):
        """Persist the index -> name mapping as JSON."""
        with open(self.f_mapping, "w") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
        self.mapping = {int(k): v for k, v in mapping.items()}

    def load_mapping(self):
        """Load a mapping from the JSON cache."""
        with open(self.f_mapping) as f:
            raw = json.load(f)
        self.mapping = {int(k): v for k, v in raw.items()}

    def apply_mapping(self, mapping: dict | None = None):
        """
        Apply a mapping to the cluster names in the DataArray.
        mapping: dict[int, str], e.g. {0: "NAO-", ...}
        """
        if mapping is not None:
            self.save_mapping(mapping)
        if self.patterns is None:
            raise ValueError("Keine Patterns zum Umbenennen!")
        if self.mapping is None:
            raise ValueError("Kein Mapping gefunden!")

        def _to_int_if_possible(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return val

        new_names = [self.mapping.get(_to_int_if_possible(i), str(i)) for i in self.patterns.cluster.values]
        self.patterns = self.patterns.assign_coords(cluster=("cluster", new_names))
        # Persist the renamed patterns.
        self.save_patterns()

    def get_cluster_labels(self):
        """Return the cluster labels for all time steps."""
        return self.labels

    def get_patterns(self):
        """Return the current pattern array (xarray.DataArray)."""
        return self.patterns

    def get_mapping(self):
        """Return the current mapping."""
        return self.mapping


if __name__ == "__main__":
    mgr = PatternReferenceManager(
        os.getenv("REFERENCE_NC", "./data/slp_reference.nc"), var_name="MSL", n_pcs=14, n_clusters=5
    )

    mapping = {0: "SCAN", 1: "ATL-", 2: "NAO+", 3: "NAO-", 4: "DIPOL"}
    mgr.apply_mapping(mapping)

    mgr.plot_clusters()
