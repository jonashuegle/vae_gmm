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
from visualize_tsne import visualize_tsne_matplotlib

from typing import Optional

class ClusteringLoader:
    def __init__(
        self,
        log_dir: str,
        experiment: str,
        version: str,
        nc_path: str,
        device: torch.device,
        pca_kmeans_path: str = "pca_km.pkl",
        mapping_path: str = "cluster_mapping.json",
        
        n_pcs: int = 10,
        n_clusters: int = None,
        
        regime_order: Optional[list[str]] = None,
    ):
        self.device = device
        self.nc_path = nc_path
        self.pca_kmeans_path = pca_kmeans_path
        self.mapping_path = mapping_path
        self.regime_order = regime_order

        self.dataset = CustomDataset(nc_file_path=nc_path, save_ram=False)
        self.times = pd.DatetimeIndex(self.dataset.all_times)
        self.plotter = Plotting(lon=self.dataset.data['lon'].values,
                                lat=self.dataset.data['lat'].values)

        base = os.path.join(log_dir, experiment, f"version_{version}")
        ckpt_folder = os.path.join(base, "checkpoints")
        last_ckpt = os.path.join(ckpt_folder, "last.ckpt")
        if not os.path.isfile(last_ckpt):
            raise FileNotFoundError(f"Kann last.ckpt nicht finden in {ckpt_folder}")
        self.model = VAE.load_from_checkpoint(last_ckpt, map_location=device)
        self.model.to(device).eval()

        self.gamma, self.mu = self._extract_all()
        self.vae_labels = self.gamma.argmax(axis=1)

        self.dataset_pca = CustomDataset(nc_file_path=nc_path, save_ram=False, sqrt=True)
        self._load_or_compute_pca_kmeans(n_pcs, n_clusters or self.gamma.shape[1])
        self.kmeans_labels = self.km.labels_
        self.labels = self.vae_labels

        # 1) Laden oder leeres Mapping für KMeans → Regime
        if os.path.isfile(self.mapping_path):
            with open(self.mapping_path, 'r') as f:
                raw = json.load(f)
            self.km_to_regime = {int(k): v for k, v in raw.items()}
        else:
            self.km_to_regime = None
            print("Kein KMeans→Regime-Mapping gefunden. Bitte zuerst assign_manual_mapping(...) aufrufen.")
            self.plot_composition(self.kmeans_composition)

        # 2) Platzhalter für Model → Regime (Auto-Mapping) und User Mapping
        self.model_to_regime: Optional[dict[int, str]] = None
        self.named_labels: Optional[np.ndarray] = None
        self.user_mapping: Optional[dict[int, str]] = None
        self.user_mapping_correlations: Optional[dict[str, float]] = None
        self.similarity_matrix: Optional[np.ndarray] = None

        # 3) Auto Mapping bei Start, falls km_to_regime vorhanden
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
            msl = np.asarray(self.dataset_pca.data['MSL'])
            n_samples = msl.shape[0]
            n_features = np.prod(msl.shape[1:])
            msl_flat = msl.reshape(n_samples, n_features)

            self.pca = PCA(n_components=n_pcs)
            self.features = self.pca.fit_transform(msl_flat)
            self.km = KMeans(n_clusters=n_clusters, random_state=42)
            self.km.fit(self.features)
            with open(self.pca_kmeans_path, 'wb') as f:
                pickle.dump((self.pca, self.km), f)
            print(f"PCA+KMeans berechnet und gespeichert in {self.pca_kmeans_path}.")

    def set_user_mapping(self, mapping: dict[int, str]):
        """Setzt das User-Mapping und berechnet die zugehörigen Korrelationen."""
        if self.model_to_regime is None:
            self.auto_map_by_correlation()  # sicherstellen, dass similarity_matrix existiert
        self.user_mapping = mapping
        self.user_mapping_correlations = self._correlations_for_mapping(mapping)
        self.named_labels = np.array([mapping.get(int(l), str(l)) for l in self.vae_labels])
        print("User Mapping gesetzt. Korrelationen:", self.user_mapping_correlations)

    def _correlations_for_mapping(self, mapping: dict[int, str]) -> dict:
        if self.model_to_regime is None or self.similarity_matrix is None:
            self.auto_map_by_correlation()
        R = self.similarity_matrix
        vae_name_to_idx = {v: k for k, v in self.model_to_regime.items()}
        corrs = {}
        for km_idx, regime in mapping.items():
            vae_idx = vae_name_to_idx[regime]
            corrs[regime] = R[vae_idx, km_idx]
        return corrs

    def assign_manual_mapping(self, mapping: dict[int, str]):
        """Speichert das KMeans→Regime-Mapping und lädt es als Standard ein."""
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

        arr_km = comp_km.values.reshape(nC, -1)
        arr_vd = comp_vd.values.reshape(nC, -1)

        lat = comp_km.coords['lat'].values
        lon = comp_km.coords.get('lon', comp_km.coords.get('longitude')).values
        w_flat = (self.area_weights(lat)[:, None] * np.ones((lat.size, lon.size))).ravel()
        w_flat /= w_flat.sum()

        R = np.zeros((nC, nC))
        for i in range(nC):
            for j in range(nC):
                ref = arr_km[j] - np.sum(w_flat * arr_km[j])
                exp = arr_vd[i] - np.sum(w_flat * arr_vd[i])
                sigma_r = np.sqrt(np.sum(w_flat * ref ** 2))
                sigma_e = np.sqrt(np.sum(w_flat * exp ** 2))
                R[i, j] = np.sum(w_flat * ref * exp) / (sigma_r * sigma_e)

        row_ind, col_ind = linear_sum_assignment(-R)
        self.similarity_matrix = R
        self.mapping_assignment = (row_ind, col_ind)

        mapping = {}
        for i, j in zip(row_ind, col_ind):
            mapping[i] = self.km_to_regime[j]
        self.model_to_regime = mapping
        self.named_labels = np.array([mapping[int(l)] for l in self.vae_labels])
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
        fig=None,
        axes=None,
        override_mapping: Optional[dict[int, str]] = None,
        titles: Optional[list[str]] = None,
        show_colorbar=True,
        colorbar_kwargs=None,
        **plot_kwargs
    ):
        comp_type = comp.attrs.get('composition_type')
        if comp_type == 'vae' and self.model_to_regime is None:
            print("Kein Modell-Mapping gefunden, führe auto_map_by_correlation aus...")
            self.auto_map_by_correlation()

        # Mapping-Priorität abhängig vom Typ
        if override_mapping is not None:
            mapping = override_mapping
        elif comp_type == "vae":
            mapping = self.user_mapping if self.user_mapping is not None else self.model_to_regime
        elif comp_type == "kmeans":
            mapping = self.km_to_regime
        else:
            mapping = None

        # regime_order Handling wie gehabt
        if self.regime_order is not None and mapping is not None:
            name_to_cluster = {v: k for k, v in mapping.items()}
            try:
                cluster_order = [name_to_cluster[name] for name in self.regime_order]
                comp = comp.sel(cluster=cluster_order)
            except KeyError:
                print("Warnung: regime_order stimmt nicht mit Mapping überein. Reihenfolge wird ignoriert.")

        if titles is None:
            if mapping is not None:
                idxs = comp.coords['cluster'].values
                titles = [f"{mapping.get(int(i), str(int(i)))}" for i in idxs]
            else:
                titles = [str(int(i)) for i in comp.coords['cluster'].values]

        lon = comp.coords.get('lon', comp.coords.get('longitude'))
        lat = comp.coords.get('lat', comp.coords.get('latitude'))
        fig, axs, cf_handles, cb = self.plotter.plot_isolines(
            comp.values,
            fig=fig,
            axes=axes,
            titles=titles,
            show_colorbar=show_colorbar,
            colorbar_kwargs=colorbar_kwargs or {},
            **plot_kwargs
        )
        return fig, axs, cf_handles, cb


    def plot_model_composition(self):
        """Shortcut: Plotte VAE-Komposition mit dem gerade aktiven Mapping."""
        self.plot_composition(self.vae_composition)

    def cluster_periods(self) -> pd.DataFrame:
        labels, times = self.labels, self.times
        records = []
        if len(labels) == 0:
            return pd.DataFrame(columns=['cluster', 'start_time', 'end_time', 'duration', 'mid_time'])
        cur, start = labels[0], 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                st, en = times[start], times[i - 1]
                dur, mid = en - st, st + (en - st) / 2
                records.append({
                    'cluster': int(cur),
                    'start_time': st,
                    'end_time': en,
                    'duration': dur,
                    'mid_time': mid
                })
                cur, start = labels[i], i
        st, en = times[start], times[-1]
        dur, mid = en - st, st + (en - st) / 2
        records.append({
            'cluster': int(cur),
            'start_time': st,
            'end_time': en,
            'duration': dur,
            'mid_time': mid
        })
        return pd.DataFrame(records)

    def plot_tsne(self, fig=None, ax=None, use_kmeans=True, n_components=2, fig_size=(10, 10), **kwargs):
        """
        Plot t-SNE-Embedding mit Regime-Namen als Cluster-Labels.
        """
        if use_kmeans:
            labels = self.kmeans_labels
            msl = np.asarray(self.dataset_pca.data['MSL'])
            n_samples = msl.shape[0]
            n_features = np.prod(msl.shape[1:])
            msl_flat = msl.reshape(n_samples, n_features)
            features = self.pca.fit_transform(msl_flat)
            self.km.fit(features)
            cluster_to_name = self.km_to_regime
        else:
            labels = self.vae_labels
            features = self.mu
            cluster_to_name = self.model_to_regime

        unique_clusters = sorted(set(labels))
        if cluster_to_name is not None:
            cluster_names = [cluster_to_name.get(c, f"Cluster {c}") for c in unique_clusters]
        else:
            cluster_names = [f"Cluster {c}" for c in unique_clusters]

        fig, ax = visualize_tsne_matplotlib(
            features,
            cluster_probabilities=self.gamma,
            cluster_names=cluster_names,
            fig=fig,
            ax=ax,
            figsize=fig_size,
            **kwargs
        )
        return fig, ax

    def show_similarity_matrix(self):
        """Gibt die aktuelle Similarity-Matrix als DataFrame aus."""
        if self.similarity_matrix is not None:
            df = pd.DataFrame(self.similarity_matrix)
            print(df.round(3))
            return df
        else:
            print("Noch keine Similarity-Matrix berechnet.")
            return None

    def get_named_and_sorted_vae_pattern(self):
        """
        Gibt das VAE-Pattern zurück – umbenannt und in regime_order sortiert!
        """
        comp = self.vae_composition
        # Hole das aktuelle Mapping: model_to_regime oder user_mapping
        mapping = self.user_mapping if self.user_mapping is not None else self.model_to_regime
        if mapping is None:
            # Automatisches Mapping, falls noch nicht passiert!
            self.auto_map_by_correlation()
            mapping = self.user_mapping if self.user_mapping is not None else self.model_to_regime


        # Cluster-Koordinate umbenennen
        comp_named = comp.assign_coords(cluster=("cluster", [mapping[int(i)] for i in comp.cluster.values]))
        # Optional: jetzt nach regime_order sortieren
        if self.regime_order is not None:
            present = [r for r in self.regime_order if r in comp_named.cluster.values]
            comp_named = comp_named.sel(cluster=present)
        return comp_named
