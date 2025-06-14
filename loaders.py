import numpy as np
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
import os, glob, re
import torch
import pandas as pd
from dataset import CustomDataset

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from dataset import CustomDataset
from VAE_GMM import VAE
from plotting import Plotting

class ClusteringLoader:
    def __init__(
        self,
        log_dir: str,
        experiment: str,
        version: str,
        nc_path: str,
        device: torch.device,
        pca_kmeans_path: str = "pca_km.pkl",
        n_pcs: int = 10,
        n_clusters: int = None,
        mapping_path: str = "cluster_mapping.json",
        regime_order: list[str] = ["NAO+", "NAO-", "DIPOL", "ATL-", "SCAN"]
    ):
        self.device = device
        self.nc_path = nc_path
        self.pca_kmeans_path = pca_kmeans_path
        self.mapping_path = mapping_path
        self.regime_order = regime_order

        # 1) Basis-Dataset + Zeit-Index
        self.dataset = CustomDataset(nc_file_path=nc_path)
        self.times = pd.DatetimeIndex(self.dataset.data.time)

        # 2) VAE-Modell laden
        base = os.path.join(log_dir, experiment, f"version_{version}")
        ckpt_folder = os.path.join(base, "checkpoints")
        last_ckpt = os.path.join(ckpt_folder, "last.ckpt")
        if not os.path.isfile(last_ckpt):
            raise FileNotFoundError(f"Kann last.ckpt nicht finden in {ckpt_folder}")
        self.model = VAE.load_from_checkpoint(last_ckpt, map_location=device)
        self.model.to(device).eval()

        # 3) Gamma, mu extrahieren
        self.gamma, self.mu = self._extract_all()
        self.vae_labels = self.gamma.argmax(axis=1)

        # 4) PCA + KMeans
        self._load_or_compute_pca_kmeans(n_pcs, n_clusters or self.gamma.shape[1])
        self.kmeans_labels = self.km.labels_

        # Standard-Labels auf VAE
        self.labels = self.vae_labels

        # 6) KMeans→Regime-Mapping laden
        if os.path.isfile(self.mapping_path):
            with open(self.mapping_path, 'r') as f:
                raw = json.load(f)
            self.km_to_regime = {int(k): v for k, v in raw.items()}
        else:
            self.km_to_regime = None
            print("Kein KMeans→Regime-Mapping gefunden. Bitte zuerst assign_manual_mapping(...) aufrufen.")
            self.plot_composition(self.kmeans_composition,
                        fig_scale_factor=5)


        # Platzhalter für Modell-Mapping
        self.model_to_regime = None
        self.named_labels = None

        # 7) Automatisches Mapping direkt nach Init
        if self.km_to_regime is not None:
            print("Führe automatisches Mapping via flächengewichteter Korrelation nach Initialisierung aus...")
            self.auto_map_by_correlation()

    def _extract_all(self):
        dl = DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=4)
        all_gamma, all_mu = [], []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(self.device)
                x_recon, mu, log_var, z = self.model(x)
                _, gamma = self.model.gaussian_mixture_log_prob(z)
                all_gamma.append(gamma.cpu())
                all_mu.append(mu.cpu())
        gamma = torch.cat(all_gamma, dim=0).numpy()
        mu = torch.cat(all_mu, dim=0).numpy()
        return gamma, mu

    def _load_or_compute_pca_kmeans(self, n_pcs: int, n_clusters: int):
        if os.path.isfile(self.pca_kmeans_path):
            with open(self.pca_kmeans_path, 'rb') as f:
                self.pca, self.km = pickle.load(f)
        else:
            self.pca = PCA(n_components=n_pcs)
            mu_pca = self.pca.fit_transform(self.mu)
            self.km = KMeans(n_clusters=n_clusters, random_state=42)
            self.km.fit(mu_pca)
            with open(self.pca_kmeans_path, 'wb') as f:
                pickle.dump((self.pca, self.km), f)
            print(f"PCA+KMeans berechnet und gespeichert in {self.pca_kmeans_path}.")

    def assign_manual_mapping(self, mapping: dict[int, str]):
        """ 
        Assigns a manual mapping from KMeans cluster labels to regime names,
        persists it to disk, and updates the loader’s internal state.
        Parameters:
            mapping (dict[int, str]):
                A dictionary mapping integer cluster labels (e.g., 0, 1, 2, …)
                to human-readable regime names (e.g., "NAO+", "NAO-", "DIPOL", …).
        Side Effects:
            1. Writes the mapping as JSON to the file at self.mapping_path,
               converting integer keys to strings.
            2. Updates the instance attribute `self.km_to_regime` with the provided mapping.
            3. Prints a confirmation message indicating where the mapping was saved.
        Example:
            manual_map = {
                0: "NAO+",
                1: "NAO-",
                2: "DIPOL",
                3: "ATL-",
                4: "SCAN"
            }
            object.assign_manual_mapping(manual_map)
        """
        with open(self.mapping_path, 'w') as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
        self.km_to_regime = mapping
        print(f"KMeans→Regime-Mapping gespeichert in {self.mapping_path}.")

    @staticmethod
    def area_weights(lat: np.ndarray) -> np.ndarray:
        return np.cos(np.deg2rad(lat))

    @staticmethod
    def weighted_stats(ref: np.ndarray, exp: np.ndarray, w: np.ndarray):
        w = w / np.sum(w)
        ref = ref - np.sum(w * ref)
        exp = exp - np.sum(w * exp)
        sigma_r = np.sqrt(np.sum(w * ref**2))
        sigma_e = np.sqrt(np.sum(w * exp**2))
        R = np.sum(w * ref * exp) / (sigma_r * sigma_e)
        crmse = np.sqrt(sigma_r**2 + sigma_e**2 - 2 * sigma_r * sigma_e * R)
        return sigma_r, sigma_e, R, crmse

    def auto_map_by_correlation(self):
        comp_km = self.kmeans_composition
        comp_vd = self.vae_composition
        nC = comp_km.sizes['cluster']

        # 1) Daten flachziehen
        arr_km = comp_km.values.reshape(nC, -1)
        arr_vd = comp_vd.values.reshape(nC, -1)

        # 2) Flächengewichte
        lat = comp_km.coords['lat'].values
        lon = comp_km.coords.get('lon', comp_km.coords.get('longitude')).values
        w_flat = (self.area_weights(lat)[:,None] * np.ones((lat.size, lon.size))).ravel()
        w_flat /= w_flat.sum()

        # 3) Korrelationsmatrix berechnen
        R = np.zeros((nC, nC))
        for i in range(nC):
            for j in range(nC):
                ref =   arr_km[j] -   np.sum(w_flat *   arr_km[j])
                exp =   arr_vd[i] -   np.sum(w_flat *   arr_vd[i])
                sigma_r = np.sqrt(np.sum(w_flat * ref**2))
                sigma_e = np.sqrt(np.sum(w_flat * exp**2))
                R[i,j] = np.sum(w_flat * ref * exp) / (sigma_r * sigma_e)

        # 4) Hungarian auf -R (weil linear_sum_assignment minimalisiert)
        row_ind, col_ind = linear_sum_assignment(-R)

        # 5) mapping bauen
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            mapping[i] = self.km_to_regime[j]

        self.model_to_regime = mapping
        self.named_labels    = np.array([mapping[int(l)] for l in self.vae_labels])
        print("Automatisches Mapping (global) via Hungarian-Algorithmus gesetzt.")

    @property
    def vae_composition(self) -> xr.DataArray:
        ds = xr.open_dataset(self.nc_path)
        slp = ds['MSL']
        times = pd.to_datetime([str(t) for t in slp.time.values])
        slp = slp.assign_coords(time=("time", times))
        T = len(self.vae_labels)
        sel = slp.isel(time=slice(0, T))
        sel = sel.assign_coords(cluster=("time", self.vae_labels))
        comp = sel.groupby('cluster').mean('time').sortby('cluster')
        comp.attrs['composition_type'] = 'vae'
        return comp

    @property
    def kmeans_composition(self) -> xr.DataArray:
        ds = xr.open_dataset(self.nc_path)
        slp = ds['MSL']
        times = pd.to_datetime([str(t) for t in slp.time.values])
        slp = slp.assign_coords(time=("time", times))
        T = len(self.kmeans_labels)
        sel = slp.isel(time=slice(0, T))
        sel = sel.assign_coords(cluster=("time", self.kmeans_labels))
        comp = sel.groupby('cluster').mean('time').sortby('cluster')
        comp.attrs['composition_type'] = 'kmeans'
        return comp

    def plot_composition(
        self,
        comp: xr.DataArray,
        fig_scale_factor: float = 6,
        titles: list[str] = None
    ):
        """
        Zeichnet ein DataArray (cluster, lat, lon) als Isolinien.
        Optional: `titles` (Liste von Strings) wird pro Cluster angezeigt.
        Für VAE-Composition wird bei Bedarf automatisch auto_map_by_correlation aufgerufen,
        um `model_to_regime` zu setzen. Danach Standard-Titles aus Mapping:
          - `model_to_regime` für `composition_type=='vae'`
          - `km_to_regime`   für `composition_type=='kmeans'`
        """

        mapping = (self.model_to_regime if comp.attrs["composition_type"] == "vae"
               else self.km_to_regime)
        # 1) Wenn eine fixe Reihenfolge von Regime-Namen übergeben wurde,
        #    remappe sie auf Cluster-Indizes und reorder das DataArray:
        if self.regime_order is not None and mapping is not None:
            # invertiere das mapping: regime_name -> cluster_index
            name_to_cluster = {v: k for k, v in mapping.items()}
            # baue Liste von cluster-IDs in genau dieser Reihenfolge
            cluster_order = [name_to_cluster[name] for name in self.regime_order]
            comp = comp.sel(cluster=cluster_order)

        comp_type = comp.attrs.get('composition_type')
        # Automatisches Mapping für VAE, falls noch nicht vorhanden
        if comp_type == 'vae' and self.model_to_regime is None:
            print("Kein Modell-Mapping gefunden, führe auto_map_by_correlation aus...")
            self.auto_map_by_correlation()
        # Titel-Zuordnung
        if titles is None:
            if comp_type == 'vae' and self.model_to_regime is not None:
                mapping = self.model_to_regime
            elif comp_type == 'kmeans' and self.km_to_regime is not None:
                mapping = self.km_to_regime
            else:
                mapping = None
            if mapping is not None:
                idxs = comp.coords['cluster'].values
                titles = [mapping[int(i)] for i in idxs]
        # Fallback numerische Labels
        if titles is None:
            titles = [str(int(i)) for i in comp.coords['cluster'].values]

        lon = comp.coords.get('lon', comp.coords.get('longitude'))
        lat = comp.coords.get('lat', comp.coords.get('latitude'))
        plotter = Plotting(lon=lon.values, lat=lat.values, fig_scale_factor=fig_scale_factor)
        fig, ax = plotter.plot_isolines(comp, titles=titles)
        plt.tight_layout()
        plt.show()

    def plot_model_composition(self, fig_scale_factor: float = 6):
        """
        Shortcut: Plotte VAE-Komposition mit modellbezogenen Regime-Titeln.
        """
        if self.model_to_regime is None:
            print("Führe automatisches Mapping via Korrelation aus...")
            self.auto_map_by_correlation()
        self.plot_composition(self.vae_composition, fig_scale_factor=fig_scale_factor)

    def cluster_periods(self) -> pd.DataFrame:
        labels, times = self.labels, self.times
        records = []
        if len(labels) == 0:
            return pd.DataFrame(columns=['cluster','start_time','end_time','duration','mid_time'])
        cur, start = labels[0], 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                st, en = times[start], times[i-1]
                dur, mid = en - st, st + (en-st)/2
                records.append({
                    'cluster': int(cur),
                    'start_time': st,
                    'end_time': en,
                    'duration': dur,
                    'mid_time': mid
                })
                cur, start = labels[i], i
        st, en = times[start], times[-1]
        dur, mid = en - st, st + (en-st)/2
        records.append({
                    'cluster': int(cur),
                    'start_time': st,
                    'end_time': en,
                    'duration': dur,
                    'mid_time': mid
                })
        return pd.DataFrame(records)

