# pattern_taylor.py
# ---------------------------------------------------------------------
# (c) 2025 – Taylor‑Diagram‑/Pattern‑Utility
#
# ▸ erstellt Referenz‑Pattern via flächen‑gewichteter PCA+KMeans
# ▸ speichert sie als   <cache>/<filename>_ref.nc  (Pattern)
#                       <cache>/<filename>_labels.npy  (Zeitlabels)
#                       <cache>/<filename>_meta.json  (Parameter)
# ▸ matcht neue Pattern (max. Korrelation, Hungarian) auf die Referenz
# ▸ zeichnet einen Taylor‑Plot (area‑weighted)
# ---------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, List, Optional

import json
import numpy as np
import xarray as xr

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder

# ---------------------------------------------------------------------
# 0)  Hilfsfunktionen  -------------------------------------------------
# ---------------------------------------------------------------------


def area_weights(lat: np.ndarray) -> np.ndarray:
    """cos(lat)‑Gewichte (1‑D)."""
    return np.cos(np.deg2rad(lat))


def to_pca_matrix(
    da: xr.DataArray,
    *,
    weight_lat: bool = True,
    demean: bool = True,
) -> Tuple[np.ndarray, xr.DataArray]:
    """
    (time, lat, lon) ➜ X (n_time, n_space)  +  2‑D‑Gewichte (lat,lon)

    * Spalten, die komplett NaN sind, werden entfernt
    * verbleibende NaNs werden spaltenweise mit 0 (=Mittel) gefüllt
    """
    if demean:
        da = da - da.mean("time")

    if weight_lat:
        w1d = np.sqrt(np.cos(np.deg2rad(da["lat"]))).rename("sqrt_coslat")
        da = da * w1d
    else:
        w1d = xr.ones_like(da.isel(time=0))

    #   NaN‑Robustes Stapeln
    st = da.stack(space=("lat", "lon")).transpose("time", "space")
    valid = ~st.isnull().all("time")
    st = st.isel(space=valid)

    #   Restliche NaNs = 0
    st = st.fillna(0.0)

    X = st.values
    w2d = (w1d**2).broadcast_like(da.isel(time=0))  # Gewichte zum Auspacken später
    return X, w2d


def pca_kmeans_clusters(
    da: xr.DataArray,
    *,
    n_pc: int,
    n_cluster: int,
    random_state: int = 0,
    weight_lat: bool = True,
    demean: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Gibt (patterns, labels) zurück."""
    X, _ = to_pca_matrix(da, weight_lat=weight_lat, demean=demean)
    pcs = PCA(n_components=n_pc, random_state=random_state).fit_transform(X)

    km = KMeans(
        n_clusters=n_cluster,
        n_init=100,
        random_state=random_state,
    ).fit(pcs)

    labels = xr.DataArray(km.labels_, coords=[da["time"]], dims=["time"])

    patterns = (
        da.assign_coords(cluster=labels)
        .groupby("cluster")
        .mean("time")
        .sortby("cluster")
    )
    return patterns, labels


def weighted_stats(ref: np.ndarray, exp: np.ndarray, w: np.ndarray):
    """σ_ref, σ_exp, corr, CRMSE (alle gewichtet)."""
    w = w / w.sum()
    ref = ref - (w * ref).sum()
    exp = exp - (w * exp).sum()
    s_ref = np.sqrt((w * ref**2).sum())
    s_exp = np.sqrt((w * exp**2).sum())
    corr = (w * ref * exp).sum() / (s_ref * s_exp)
    crmse = np.sqrt(s_ref**2 + s_exp**2 - 2 * s_ref * s_exp * corr)
    return s_ref, s_exp, corr, crmse


# ---------------------------------------------------------------------
# 1)  Mini‑Taylor‑Diagramm  -------------------------------------------
# ---------------------------------------------------------------------


class _TaylorDiagram:
    """kleinst‑mögliche Taylor‑Diagramm‑Klasse."""

    def __init__(self, refstd: float = 1.0, fig=None, rect=111, srange=(0, 2.5)):
        tr = PolarAxes.PolarTransform()
        rs = np.concatenate([np.linspace(0, 1, 6)[1:], [0.95, 0.99]])
        ts = np.arccos(rs)
        gl = grid_finder.FixedLocator(ts)
        tf = grid_finder.DictFormatter({t: f"{r:.2f}" for t, r in zip(ts, rs)})
        helper = floating_axes.GridHelperCurveLinear(
            tr, extremes=(0, np.pi / 2, *srange), grid_locator1=gl, tick_formatter1=tf
        )

        fig = plt.figure() if fig is None else fig
        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=helper)
        fig.add_subplot(ax)

        ax.axis["top"].label.set_text("Correlation")
        ax.axis["right"].label.set_text("σ / σ$_{ref}$")
        ax.axis["bottom"].set_visible(False)

        self.ax = ax.get_aux_axes(tr)
        theta = np.linspace(0, np.pi / 2, 100)
        self.ax.plot(theta * 0, np.linspace(*srange, 100), "k-", lw=0.3)
        self.ax.plot([0], [refstd], "ko", label="Reference")
        self.refstd = refstd

    def add(self, std_ratio: float, corr: float, label: str, color):
        self.ax.plot(np.arccos(corr), std_ratio, "o", ms=10, label=label, c=color)


# ---------------------------------------------------------------------
# 2)  Haupt‑Klasse  ----------------------------------------------------
# ---------------------------------------------------------------------


class PatternComparator:
    """
    Erstellt/liest PCA‑KMeans‑Referenzmuster und vergleicht neue Cluster‑Pattern.
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        nc_path: str | Path,
        *,
        var_name: str,
        n_pc: int,
        n_cluster: int,
        cache_dir: str | Path = "cache",
        random_state: int = 0,
        weight_lat: bool = True,
        demean: bool = True,
    ):
        self.nc_path = Path(nc_path)
        self.var_name = var_name
        self.n_pc = n_pc
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.weight_lat = weight_lat
        self.demean = demean

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        stem = self.nc_path.stem
        self.f_nc = self.cache_dir / f"{stem}_ref.nc"
        self.f_lbl = self.cache_dir / f"{stem}_labels.npy"
        self.f_meta = self.cache_dir / f"{stem}_meta.json"

        if self.f_nc.exists() and self.f_lbl.exists() and self.f_meta.exists():
            self._load_cache()
        else:
            self._compute_reference()
            self._save_cache()

    def rename_reference(self, new_names):
        """Ändert die cluster-Koordinate der Referenz und
           überschreibt die Cache-Datei."""
        if len(new_names) != self.n_cluster:
            raise ValueError("Länge der Namen stimmt nicht")
        self.pat_ref = self.pat_ref.assign_coords(
            cluster=("cluster", list(new_names)))
        # erneut speichern
        self._save_cache()

    # -----------------------------------------------------------------
    # intern
    # -----------------------------------------------------------------



    def _compute_reference(self):
        ds = xr.open_dataset(self.nc_path, decode_times=False)
        da = ds[self.var_name]

        self.pat_ref, lbl_da = pca_kmeans_clusters(
            da,
            n_pc=self.n_pc,
            n_cluster=self.n_cluster,
            random_state=self.random_state,
            weight_lat=self.weight_lat,
            demean=self.demean,
        )
        self.labels_ref = lbl_da  # DataArray (time,)

    def _save_cache(self):
        self.pat_ref.to_netcdf(self.f_nc)
        np.save(self.f_lbl, self.labels_ref.values)
        meta = dict(
            n_pc=self.n_pc,
            n_cluster=self.n_cluster,
            random_state=self.random_state,
            weight_lat=self.weight_lat,
            demean=self.demean,
        )
        with open(self.f_meta, "w") as fh:
            json.dump(meta, fh)

    def _load_cache(self):
        self.pat_ref = xr.open_dataset(self.f_nc, decode_times=False)[self.var_name]
        lbl = np.load(self.f_lbl)
        time_coord = self.pat_ref.coords.get("time", None)
        if time_coord is None:
            time_coord = np.arange(len(lbl))
        self.labels_ref = xr.DataArray(lbl, coords=[time_coord], dims=["time"])

    # -----------------------------------------------------------------
    # public
    # -----------------------------------------------------------------

    def match_patterns(self, patterns: xr.DataArray) -> List[int]:
        """liefert Mapping‐Liste: mapping[i] = j  ➜  pattern_i ↔ ref_j"""
        if patterns.dims[0] != "cluster":
            raise ValueError("patterns muss die Dimension 'cluster' an erster Stelle haben")
        if patterns.shape[0] != self.n_cluster:
            raise ValueError("Anzahl Cluster stimmt nicht mit Referenz überein")

        lat = self.pat_ref["lat"].values
        w2d = area_weights(lat)[:, None] * np.ones_like(self.pat_ref.isel(cluster=0).values)

        # Korrelationsmatrix (Model_i vs Ref_j)
        C = np.empty((self.n_cluster, self.n_cluster))
        for i in range(self.n_cluster):
            for j in range(self.n_cluster):
                _, _, R, _ = weighted_stats(
                    self.pat_ref.isel(cluster=j).values.ravel(),
                    patterns.isel(cluster=i).values.ravel(),
                    w2d.ravel(),
                )
                C[i, j] = -R  # Hungarian = Minimierungs‑Problem

        row, col = linear_sum_assignment(C)
        return col.tolist()  # model‑>ref

    def plot_taylor_per_reference(self,
                                model_patterns: xr.DataArray,
                                title_prefix: str = "TaylorDiagram  Ref"):
        """
        Zeichnet für jede Referenz-Klasse ein eigenes Taylor-Diagramm, in dem
        alle fünf Model-Cluster gegen genau diese Referenz verglichen werden.
        """
        n_clu = self.n_cluster
        lat   = self.pat_ref["lat"].values
        w2d   = area_weights(lat)[:, None] * np.ones((lat.size,
                                                    self.pat_ref["lon"].size))

        colors = plt.cm.tab10(np.linspace(0, 1, n_clu))

        for j_ref in range(n_clu):
            dia = _TaylorDiagram(refstd=1)

            ref = self.pat_ref.isel(cluster=j_ref).values.ravel()
            for i_mod in range(n_clu):
                exp = model_patterns.isel(cluster=i_mod).values.ravel()
                σr, σe, r, _ = weighted_stats(ref, exp, w2d.ravel())
                dia.add(σe/σr, r,
                        label=f"C{i_mod}",      # nur Modell-Label
                        color=colors[i_mod])

            plt.title(f"{title_prefix} R{j_ref}")
            plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
            plt.show()

