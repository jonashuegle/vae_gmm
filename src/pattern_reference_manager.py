import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, List, Dict

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.dataset import CustomDataset
from src.plotting import Plotting

class PatternReferenceManager:
    """
    Erstellt/verwaltet Referenz-Cluster (PCA + KMeans) auf NetCDF-Daten
    - Wurzel(Cos(lat))-Gewichtung via CustomDataset
    - Caching von Labels/Patterns
    - Optionale manuelle Namensgebung (Mapping)
    - Plotting über externe Plotting-Klasse
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

        self.patterns: Optional[xr.DataArray] = None
        self.labels: Optional[np.ndarray] = None
        self.mapping: Optional[Dict[int, str]] = None

        self._load_or_compute_patterns()
        if self.f_mapping.exists():
            self.load_mapping()
        else:
            # Automatisch plotten (ohne Namen) falls kein Mapping existiert
            print("[PatternReferenceManager] Kein Mapping gefunden – Cluster werden mit Nummern geplottet.")
            self.plot_clusters()


    def _load_or_compute_patterns(self):
        """Lädt Patterns & Labels aus Cache oder berechnet sie neu."""
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
        Berechnet PCA + KMeans für Referenzdaten.
        Ergebnis: self.patterns (xarray), self.labels (numpy)
        """
        # Lade Daten (sqrt-Gewichtung via CustomDataset)
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

        # PCA & KMeans
        pca = PCA(n_components=self.n_pcs, random_state=self.random_state)
        pca_results = pca.fit_transform(msl_flat)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100, random_state=self.random_state)
        labels = kmeans.fit_predict(pca_results)

        # Komposition per Cluster: xarray-Objekt
        ds = xr.open_dataset(self.nc_file_path)
        slp = ds[self.var_name].isel(time=slice(0, len(labels)))
        slp = slp.assign_coords(cluster=("time", labels))
        patterns = slp.groupby("cluster").mean("time").sortby("cluster")

        self.patterns = patterns
        self.labels = labels

    def save_patterns(self):
        """Speichert Patterns (xarray) und Labels (npy) im Cache."""
        assert self.patterns is not None and self.labels is not None
        self.patterns.to_netcdf(self.f_pattern)
        np.save(self.f_labels, self.labels)

    def delete_cache(self):
        """Löscht gespeicherte Patterns/Labels/Mapping."""
        for f in [self.f_pattern, self.f_labels, self.f_mapping]:
            if os.path.exists(f):
                os.remove(f)
        print("PatternReferenceManager: Cache gelöscht.")

    def plot_clusters(self, plotter: Optional[Plotting] = None, titles: Optional[List[str]] = None, show_colorbar=True):
        """
        Plottet die Cluster-Kompositionen.
        """
        if self.patterns is None:
            raise ValueError("Keine Patterns gefunden!")
        # Plotter initialisieren falls nicht gegeben
        if plotter is None:
            lon = self.patterns['lon'].values
            lat = self.patterns['lat'].values
            plotter = Plotting(lon=lon, lat=lat)
        if titles is None:
            if self.mapping is not None:
                def _to_int_if_possible(val):
                    try:
                        return int(val)
                    except Exception:
                        return val
                titles = [self.mapping.get(_to_int_if_possible(i), str(i)) for i in self.patterns.cluster.values]
            else:
                titles = [str(i) for i in self.patterns.cluster.values]

        plotter.plot_isolines(self.patterns.values, titles=titles, show_colorbar=show_colorbar)

    def save_mapping(self, mapping: dict):
        """Speichert das Mapping (Nummer → Name) als JSON."""
        with open(self.f_mapping, "w") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
        self.mapping = {int(k): v for k, v in mapping.items()}

    def load_mapping(self):
        """Lädt ein Mapping aus dem JSON-Cache."""
        with open(self.f_mapping, "r") as f:
            raw = json.load(f)
        self.mapping = {int(k): v for k, v in raw.items()}

    def apply_mapping(self, mapping: Optional[dict] = None):
        """
        Wendet Mapping auf die Cluster-Namen im DataArray an.
        mapping: dict[int, str] (z.B. {0: "NAO-", ...})
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
        # Nach Anwendung neues Muster abspeichern:
        self.save_patterns()


    def get_cluster_labels(self):
        """Gibt die aktuellen Cluster-Labels für alle Zeitpunkte zurück."""
        return self.labels

    def get_patterns(self):
        """Gibt das aktuelle Muster-Array zurück (xarray.DataArray)."""
        return self.patterns

    def get_mapping(self):
        """Gibt das aktuelle Mapping zurück (dict)."""
        return self.mapping



if __name__ == "__main__":

    mgr = PatternReferenceManager(
                            "/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc",
                             var_name="MSL", 
                             n_pcs=14, 
                             n_clusters=5
                            )

    # Nach Sichtung: Mapping anlegen und speichern
    mapping = {0: "SCAN", 1: "ATL-", 2: "NAO+", 3: "NAO-", 4: "DIPOL"}
    mgr.apply_mapping(mapping)

    # Jetzt mit Namen geplottet!
    mgr.plot_clusters()