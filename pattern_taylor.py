# pattern_taylor.py
# ---------------------------------------------------------------------
# (c) 2025 – Taylor‑Diagram‑/Pattern‑Utility
#
# ▸ erstellt Referenz‑Pattern via flächen‑gewichteter PCA+KMeans
# ▸ speichert sie als   <cache>/<filename>_ref.nc  (Pattern)
#                       <cache>/<filename>_labels.npy  (Zeitlabels)
#                       <cache>/<filename>_meta.json  (Parameter)
# ▸ matcht neue Pattern (max. Korrelation, Hungarian) auf die Referenz
# ▸ zeichnet einen Taylor‑Plot (area‑weighted)
# ---------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches


from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


from plotting import Plotting
from pattern_reference_manager import PatternReferenceManager

import numpy as np

def curved_text(ax, text, radius, center=(0,0), start_angle=-160, end_angle=160, **kwargs):
    # Text wird als Einzelbuchstaben platziert
    n = len(text)
    angles = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), n)
    for i, (char, angle) in enumerate(zip(text, angles)):
        # Position auf dem Kreisrand
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle) * 0.94
        rotation = np.rad2deg(angle) + 90  # +90, damit es entlang des Kreises steht
        ax.text(x, y, char, rotation=rotation, ha='center', va='center', **kwargs)


def area_weights(lat: np.ndarray) -> np.ndarray:
    """cos(lat)‑Gewichte (1‑D)."""
    return np.cos(np.deg2rad(lat))

def to_pca_matrix(
    da: xr.DataArray,
    *,
    weight_lat: bool = True,
    demean: bool = True,
) -> Tuple[np.ndarray, xr.DataArray]:
    if demean:
        da = da - da.mean("time")
    if weight_lat:
        w1d = np.sqrt(np.cos(np.deg2rad(da["lat"]))).rename("sqrt_coslat")
        da = da * w1d
    else:
        w1d = xr.ones_like(da.isel(time=0))
    st = da.stack(space=("lat", "lon")).transpose("time", "space")
    valid = ~st.isnull().all("time")
    st = st.isel(space=valid)
    st = st.fillna(0.0)
    X = st.values
    w2d = (w1d**2).broadcast_like(da.isel(time=0))
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
    w = w / w.sum()
    ref = ref - (w * ref).sum()
    exp = exp - (w * exp).sum()
    s_ref = np.sqrt((w * ref**2).sum())
    s_exp = np.sqrt((w * exp**2).sum())
    corr = (w * ref * exp).sum() / (s_ref * s_exp)
    crmse = np.sqrt(s_ref**2 + s_exp**2 - 2 * s_ref * s_exp * corr)
    return s_ref, s_exp, corr, crmse

class PatternComparator:
    """
    Erstellt/liest PCA‑KMeans‑Referenzmuster und vergleicht neue Cluster‑Pattern.
    """
    def __init__(
        self,
        reference_manager: PatternReferenceManager = None,
    ):
        if reference_manager is None:
            reference_manager = PatternReferenceManager(
                "/home/a/a271125/work/data/slp.N_djfm_6h_aac_detrend_1deg_north_atlantic.nc"
                )
        # Referenz aus Manager übernehmen
        self.manager = reference_manager
        self.pat_ref = reference_manager.get_patterns()
        self.labels_ref = reference_manager.get_cluster_labels()
        self.mapping = reference_manager.get_mapping()
        self.n_cluster = self.pat_ref.sizes["cluster"]
        self.lon = self.pat_ref.lon.values
        self.lat = self.pat_ref.lat.values
        self.cluster_colors = plt.cm.tab10.colors[:self.n_cluster]

    def match_patterns(self, patterns: xr.DataArray) -> list[int]:
        """
        Liefert Mapping-Liste: mapping[i] = j  ➜  pattern_i ↔ ref_j
        (Vergleicht neue Patterns mit der Referenz nach maximaler Korrelation, Hungarian-Algorithmus)
        """
        if patterns.dims[0] != "cluster":
            raise ValueError("patterns muss die Dimension 'cluster' an erster Stelle haben")
        if patterns.shape[0] != self.n_cluster:
            raise ValueError("Anzahl Cluster stimmt nicht mit Referenz überein")
        # Flächengewichtung
        w2d = area_weights(self.lat)[:, None] * np.ones_like(self.pat_ref.isel(cluster=0).values)
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



    # def plot_taylor_halfcircle(self, std_ref, stds, corrs, labels=None, ax=None, r_max=1.5, optimum_idx=None, cluster_colors=None):
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(6,3.7))
    #     ax.set_aspect('equal', 'box')
    #     ax.set_xlim(-r_max*1.05, r_max*1.05)
    #     ax.set_ylim(-r_max*1.05, 0.05*r_max)

    #     # Standardabweichungskreise (explizit Halbkreise, Rand bei r_max!)
    #     sigma_ticks = np.arange(0.5, r_max+0.001, 0.5)
    #     theta = np.linspace(0, np.pi, 300)
    #     for s in sigma_ticks:
    #         x = s * np.cos(theta)
    #         y = -s * np.sin(theta)  # Nach unten!
    #         lw = 2 if np.isclose(s, r_max) else 1
    #         ax.plot(x, y, color='grey', lw=lw, ls='-' if np.isclose(s, r_max) else '--', zorder=1, alpha=0.7)
    #         # Ticks am rechten Rand
    #         ax.text(s-0.01, 0.02, f"{s:.1f}", ha='center', va='bottom', fontsize=7, color='black', alpha=0.8)

    #     # Korrelationslinien (gleichmäßige Winkel, z.B. alle 15°)
    #     for deg in np.arange(0, 181, 15):
    #         angle = np.deg2rad(deg)
    #         r = np.cos(angle)
    #         x = [0, r_max * np.cos(angle)]
    #         y = [0, -r_max * np.sin(angle)]  # Nach unten!
    #         ax.plot(x, y, color='grey', lw=0.9, ls=':', zorder=1, alpha=0.6)
    #         # Korrelationslabel an äußeren Rand
    #         if deg % 30 == 0:
    #             label_corr = np.cos(angle)
    #             lx = 1.2 * r_max * np.cos(angle)
    #             ly = -1.14 * r_max * np.sin(angle)
    #             ax.text(lx, ly, f"{label_corr:.1f}", fontsize=7, ha='center', va='center', color='k', alpha=0.8)

    #     # Punkte plotten (immer gleiche Farben)
    #     if labels is None:
    #         labels = [f"Cluster {i+1}" for i in range(len(stds))]
    #     for i, (std, corr) in enumerate(zip(stds, corrs)):
    #         theta = np.arccos(corr)
    #         r = std / std_ref
    #         x = r * np.cos(theta)
    #         y = -r * np.sin(theta)
    #         marker = 'o'
    #         ms = 5
    #         if optimum_idx is not None and i == optimum_idx:
    #             marker = '*'
    #             ms = 8
    #         color = cluster_colors[i] if cluster_colors is not None else f"C{i}"
    #         ax.plot(x, y, marker, ms=ms, color=color, markeredgecolor='k', label=labels[i], zorder=3)
    #         #ax.text(x+0.1*r_max, y, labels[i], va='center', ha='left', fontsize=11, color=color, zorder=4)

    #     # Referenzpunkt (σ=1, θ=0 → (1,0))
    #     ax.plot([1], [-0.05], 'kv', ms=3, label='Referenz', zorder=5)
    #     #ax.text(1.03, 0.03*r_max, 'Ref', va='bottom', ha='left', fontsize=11, color='k')

    #     ax.axis('off')
    #     # Achsen-Labels (unten)

    #     curved_text(
    #         ax,
    #         "Correlation",
    #         radius=r_max*1.4,  # etwas größer als die Plots, damit es außen steht
    #         center=(0,0),
    #         start_angle=-65,
    #         end_angle=-15,
    #         fontsize=8,
    #         color='k'
    #     )

    #     ax.text(r_max * 0.85, 0.25* r_max, r"$\sigma/\sigma_{\rm ref}$", ha='left', va='center', fontsize=10)

    #     return ax


    def plot_taylor_halfcircle(
        self,
        std_ref,
        stds,
        corrs,
        labels=None,
        ax=None,
        r_max=1.5,
        optimum_idx=None,
        cluster_colors=None,
        tick_kwargs=None,
        marker_kwargs=None,
        ref_marker_kwargs=None,
        axis_label_kwargs=None,
        curved_text_kwargs=None
    ):
        """
        Dynamisch anpassbarer Taylor-Halbkreisplot mit style-dicts für Ticks, Marker, Ref, Axislabel etc.
        """
        import matplotlib.pyplot as plt

        # Default-Styles
        tick_kwargs = tick_kwargs or {
            "fontsize": 11,
            "corrlabel_r_mult": 1.2,   # Default vorher: 1.2
            "corrlabel_y_mult": -1.14  # Default vorher: -1.14
        }
        marker_kwargs = marker_kwargs or {"ms": 7, "zorder": 3}
        ref_marker_kwargs = ref_marker_kwargs or {"marker": "v", "color": "k", "s": 40, "zorder": 5}
        axis_label_kwargs = axis_label_kwargs or {"fontsize": 11}
        curved_text_kwargs = curved_text_kwargs or {"fontsize": 9, "color": "k"}


        tick_kwargs = dict(tick_kwargs or {})
        corrlabel_r_mult = tick_kwargs.pop("corrlabel_r_mult", 1.2)
        corrlabel_y_mult = tick_kwargs.pop("corrlabel_y_mult", -1.14)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.7))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-r_max * 1.05, r_max * 1.05)
        ax.set_ylim(-r_max * 1.05, 0.05 * r_max)

        # Standardabweichungskreise (explizit Halbkreise)
        sigma_ticks = np.arange(0.5, r_max + 0.001, 0.5)
        theta = np.linspace(0, np.pi, 300)
        for s in sigma_ticks:
            x = s * np.cos(theta)
            y = -s * np.sin(theta)
            lw = 2 if np.isclose(s, r_max) else 1
            ax.plot(x, y, color='grey', lw=lw, ls='-' if np.isclose(s, r_max) else '--', zorder=1, alpha=0.7)
            # Ticks am rechten Rand
            ax.text(s - 0.01, 0.02, f"{s:.1f}", ha='center', va='bottom', **tick_kwargs)


        for deg in np.arange(0, 181, 15):
            angle = np.deg2rad(deg)
            x = [0, r_max * np.cos(angle)]
            y = [0, -r_max * np.sin(angle)]
            ax.plot(x, y, color='grey', lw=0.9, ls=':', zorder=1, alpha=0.6)
            if deg % 30 == 0:
                label_corr = np.cos(angle)
                lx = corrlabel_r_mult * r_max * np.cos(angle)
                ly = corrlabel_y_mult * r_max * np.sin(angle)
                ax.text(lx, ly, f"{label_corr:.1f}", ha='center', va='center', **tick_kwargs)


        # Punkte plotten (immer gleiche Farben)
        if labels is None:
            labels = [f"Cluster {i+1}" for i in range(len(stds))]
        for i, (std, corr) in enumerate(zip(stds, corrs)):
            theta = np.arccos(corr)
            r = std / std_ref
            x = r * np.cos(theta)
            y = -r * np.sin(theta)
            # Optimum ggf. hervorheben
            mk = marker_kwargs.copy()
            if optimum_idx is not None and i == optimum_idx:
                mk["marker"] = '*'
                mk["ms"] = mk.get("ms", 7) + 2
            else:
                mk["marker"] = mk.get("marker", 'o')
            color = cluster_colors[i] if cluster_colors is not None else f"C{i}"
            ax.plot(x, y, marker=mk["marker"], color=color, markeredgecolor='k',
                    label=labels[i], ms=mk["ms"], zorder=mk.get("zorder", 3))

        # Referenzpunkt (σ=1, θ=0 → (1,0))
        ax.scatter([1], [0.0], **ref_marker_kwargs, label='Reference')

        ax.axis('off')

        # Achsen-Labels (unten)
        curved_kwargs = dict(curved_text_kwargs or {})
        r_max_mult = curved_kwargs.pop('r_max_multiplyer', 1.4)
        start_angle = curved_kwargs.pop('start_angle', -65)
        end_angle = curved_kwargs.pop('end_angle', -15)
        curved_text(
            ax,
            "Correlation",
            radius=r_max * r_max_mult,
            center=(0, 0),
            start_angle=start_angle,
            end_angle=end_angle,
            **curved_kwargs
        )
        ax.text(r_max * 0.85, 0.25 * r_max, r"$\sigma/\sigma_{\rm ref}$", ha='left', va='center', **axis_label_kwargs)

        return ax



    def plot_comparison_grid(self, pat_mod, mapping, plotter, figsize=(13, 3.5*4), std_ref=1.0):
        """
        Plottet: Referenzkarte – Taylor-Halbkreis – Modellkarte, für jedes Cluster in einer Zeile.
        Und erzeugt eine Legende am unteren Rand!
        """
        pat_ref = self.pat_ref
        nC = pat_ref.sizes["cluster"]
        cluster_colors = self.cluster_colors

        # Setup Figure und GridSpec
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(nC, 3, figure=fig, left=0.04, right=0.98, top=0.97, bottom=0.23, wspace=-0.15, hspace=0.1)

        ref_labels = [str(pat_ref.cluster.values[i]) for i in range(nC)]
        mod_labels = [str(pat_mod.cluster.values[i]) for i in range(nC)]

        handles = []
        for i_mod, j_ref in enumerate(mapping):
            h = Line2D(
                [], [], marker='o', ms=8, mec='k', mfc=cluster_colors[i_mod],
                linestyle='None',
                label=f"Cluster {mod_labels[i_mod]} → {ref_labels[j_ref]}"
            )
            handles.append(h)

        for j in range(nC):
            # -- 1. Referenzkarte (links)
            ax_map_ref = fig.add_subplot(gs[j, 0])
            plotter.plot_isolines(
                data=pat_ref.isel(cluster=j).values,
                axes=[ax_map_ref],
                titles=[f"{ref_labels[j]}"],
                show_colorbar=False
            )

            # -- 2. Taylor-Halbkreis (Mitte, nach unten geöffnet)
            ax_taylor = fig.add_subplot(gs[j, 1])
            stds = []
            corrs = []
            lat = pat_ref["lat"].values
            w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(pat_ref.isel(cluster=0).values)
            for i in range(nC):
                ref = pat_ref.isel(cluster=j).values.ravel()
                mod = pat_mod.isel(cluster=i).values.ravel()
                w = w2d.ravel()
                w = w / w.sum()
                ref_c = ref - (w * ref).sum()
                mod_c = mod - (w * mod).sum()
                s_ref = np.sqrt((w * ref_c**2).sum())
                s_mod = np.sqrt((w * mod_c**2).sum())
                corr = (w * ref_c * mod_c).sum() / (s_ref * s_mod)
                stds.append(s_mod/s_ref)
                corrs.append(corr)
            winner = np.argmax(corrs)

            # Taylorplot
            self.plot_taylor_halfcircle(
                std_ref=1.0,
                stds=stds,
                corrs=corrs,
                labels=mod_labels,
                ax=ax_taylor,
                r_max=1.5,
                optimum_idx=winner,
                cluster_colors=cluster_colors
            )

            # -- 3. Modellkarte (rechts)
            i_mod = mapping.index(j)
            ax_map_mod = fig.add_subplot(gs[j, 2])
            plotter.plot_isolines(
                data=pat_mod.isel(cluster=i_mod).values,
                axes=[ax_map_mod],
                titles=[f"Cluster {mod_labels[i_mod]}"],
                show_colorbar=False
            )

        # ----- LEGENDE -----
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.14),  # jetzt weiter oben, damit sie in den freien Bereich passt
            ncol=min(2, 5),
            frameon=False,
            fontsize=10
        )


        #plt.tight_layout(rect=[0, 0.05, 1, 1])  # unten Platz lassen für Legende


    def compare_and_plot(
        self,
        model_patterns: xr.DataArray,
        plotter: Plotting = None,
        figsize: Tuple[float, float] = (8, 12),
        palette: str = "tab10"
    ):
        mapping = self.match_patterns(model_patterns)
        if plotter is None:
            plotter = Plotting(lon=self.lon, lat=self.lat)
        self.plot_comparison_grid(model_patterns, mapping, plotter, figsize=figsize)


    def plot_taylor_per_reference(
        self,
        model_patterns: List[xr.DataArray],
        model_labels: Optional[List[str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize=(13, 2.7),
        colors=None,
        markers=None,
        r_max=1.5,
        ref_marker="X",
        ref_color="black"
    ):
        """
        Für JEDEN Referenzcluster ein Taylorplot:
        - Referenzpunkt bei (1,0)
        - Alle Modellcluster werden auf diesen einen Referenzcluster projiziert und geplottet.
        Das ist das klassische "Paper"-Taylorplot-Verhalten!
        """
        import matplotlib.pyplot as plt

        ref_patterns = self.pat_ref  # (cluster, lat, lon)
        nC = ref_patterns.sizes["cluster"]
        n_mod = len(model_patterns)

        if model_labels is None:
            model_labels = [f"Modell {i+1}" for i in range(n_mod)]
        if cluster_labels is None:
            cluster_labels = list(ref_patterns.cluster.values)
        if colors is None:
            colors = plt.cm.tab10.colors
        if markers is None:
            markers = ["o", "s", "D", "^", "X", "P"]

        fig, axes = plt.subplots(1, nC, figsize=(figsize[0], figsize[1]), squeeze=False)
        axes = axes[0]

        for ref_idx, (ax, ref_name) in enumerate(zip(axes, cluster_labels)):
            self.plot_taylor_halfcircle(std_ref=1.0, stds=[], corrs=[], ax=ax, r_max=r_max)
            # Referenzpunkt bei (1,0)
            ax.plot(1.0, 0.0, ref_marker, ms=13, color=ref_color, label="Reference", zorder=5)
            # Alle Modellcluster werden auf DIESEN Referenzcluster bezogen!
            for mod_idx, model in enumerate(model_patterns):
                n_modc = model.sizes["cluster"]
                for mod_c_idx in range(n_modc):
                    lat = ref_patterns["lat"].values
                    w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(ref_patterns.isel(cluster=0).values)
                    ref_c = ref_patterns.isel(cluster=ref_idx).values.ravel()      # **immer ref_idx!**
                    mod_c = model.isel(cluster=mod_c_idx).values.ravel()
                    w = w2d.ravel()
                    w = w / w.sum()
                    ref_c_ = ref_c - (w * ref_c).sum()
                    mod_c_ = mod_c - (w * mod_c).sum()
                    s_ref = np.sqrt((w * ref_c_**2).sum())
                    s_mod = np.sqrt((w * mod_c_**2).sum())
                    corr = (w * ref_c_ * mod_c_).sum() / (s_ref * s_mod)
                    theta = np.arccos(np.clip(corr, -1, 1))
                    r = s_mod / s_ref
                    x = r * np.cos(theta)
                    y = -r * np.sin(theta)
                    color = colors[mod_c_idx % len(colors)]
                    marker = markers[mod_c_idx % len(markers)]
                    label = f"{model_labels[mod_idx]} {cluster_labels[mod_c_idx]}"
                    ax.plot(x, y, marker, ms=10, color=color, markeredgecolor="k", label=label, zorder=4, alpha=0.9)
            ax.set_title(f"Referenz: {ref_name}", fontsize=13)
            if ref_idx == 0:
                handles, labels_leg = ax.get_legend_handles_labels()
                seen = set()
                unique = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.add(l))]
                ax.legend(
                    [h for h, l in unique],
                    [l for h, l in unique],
                    fontsize=11,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.4),
                    ncol=3,
                    frameon=False,
                    labelspacing=1.0)
            else:
                ax.legend().set_visible(False)
        plt.tight_layout()
        return fig, axes

    def plot_taylor_best_match_single(
        self,
        model_patterns: List[xr.DataArray],
        model_labels: Optional[List[str]] = None,
        cluster_labels: Optional[List[str]] = None,
        ax=None,
        figsize=(8, 6),
        r_max=1.5,
        colors=None,
        markers=None,
        tick_kwargs=None,
        axis_label_kwargs=None,
        curved_text_kwargs=None,
        marker_kwargs=None,
        ref_marker_kwargs=None
    ):
        """
        Plottet EINEN Taylorplot mit:
        - allen Referenzclustern als Marker (z.B. Stern) auf (σ/σ_ref=1, Corr=1)
        - und jeweils NUR den best-match-Modellcluster je Referenzcluster
        """

        import matplotlib.pyplot as plt
        import numpy as np

        # Referenzmuster laden
        ref_patterns = self.pat_ref  # (cluster, lat, lon)
        nC = ref_patterns.sizes["cluster"]
        n_mod = len(model_patterns)

        # Achse anlegen
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Default-Labels und -Styles
        if model_labels is None:
            model_labels = [f"Modell {i+1}" for i in range(n_mod)]
        if cluster_labels is None:
            cluster_labels = list(ref_patterns.cluster.values)
        if colors is None:
            colors = plt.cm.tab10.colors
        if markers is None:
            markers = ["o", "s", "D", "^", "X", "P"]
        tick_kwargs = tick_kwargs or {}
        axis_label_kwargs = axis_label_kwargs or {}
        curved_text_kwargs = curved_text_kwargs or {}
        marker_kwargs = marker_kwargs or {}
        ref_marker_kwargs = ref_marker_kwargs or {"marker": "v", "color": "k", "s": 40, "zorder": 5}

        # Hintergrund-Taylorplot (ohne Punkte)
        self.plot_taylor_halfcircle(
            std_ref=1.0,
            stds=[],
            corrs=[],
            ax=ax,
            r_max=r_max,
            tick_kwargs=tick_kwargs,
            axis_label_kwargs=axis_label_kwargs,
            curved_text_kwargs=curved_text_kwargs,
            marker_kwargs=marker_kwargs  # wird hier angewendet, falls du stds/corrs übergibst
        )

        # Best-Match-Modellcluster je Referenz einzeichnen
        for mod_idx, model in enumerate(model_patterns):
            n_modc = model.sizes["cluster"]
            for ref_idx in range(nC):
                # besten Clusterindex finden
                best_corr = -np.inf
                best_x = best_y = None
                for mod_c_idx in range(n_modc):
                    lat = ref_patterns["lat"].values
                    w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(ref_patterns.isel(cluster=0).values)
                    ref_c = ref_patterns.isel(cluster=ref_idx).values.ravel()
                    mod_c = model.isel(cluster=mod_c_idx).values.ravel()
                    w = w2d.ravel(); w = w / w.sum()
                    ref_c_ = ref_c - (w * ref_c).sum()
                    mod_c_ = mod_c - (w * mod_c).sum()
                    s_ref = np.sqrt((w * ref_c_**2).sum())
                    s_mod = np.sqrt((w * mod_c_**2).sum())
                    corr = (w * ref_c_ * mod_c_).sum() / (s_ref * s_mod)
                    theta = np.arccos(np.clip(corr, -1, 1))
                    r = s_mod / s_ref
                    x = r * np.cos(theta)
                    y = -r * np.sin(theta)
                    if corr > best_corr:
                        best_corr = corr
                        best_x, best_y = x, y

                # Styles für diesen Punkt aus marker_kwargs übernehmen
                color = colors[ref_idx % len(colors)]
                marker = markers[mod_idx % len(markers)]
                label = f"{model_labels[mod_idx]} {cluster_labels[ref_idx]}"

                mk = dict(marker_kwargs or {})
                mk.setdefault("marker", marker)
                # Wenn der Nutzer 'ms' (Durchmesser) angibt, in 's' umwandeln (Fläche):
                if "ms" in mk and "s" not in mk:
                    mk["s"] = mk.pop("ms") ** 2
                mk.setdefault("s", 7**2)             # Standard-Fläche (bei ms=7)
                mk.setdefault("color", color)
                # edgecolor statt markeredgecolor
                mk.setdefault("edgecolor", "k")
                mk.setdefault("label", label)
                mk.setdefault("zorder", 5)
                mk.setdefault("alpha", 0.6)

                # Scatter statt plot
                ax.scatter(best_x, best_y, **mk)

        # Referenzpunkt mit eigenem kwargs (s, marker etc.)
        #ax.scatter([1], [0.0], **ref_marker_kwargs, label='Refence')

        ax.axis('off')

        # Achsenbeschriftung „σ/σ_ref“
        curved_text(
            ax,
            "Correlation",
            radius=r_max * curved_text_kwargs.get("r_max_multiplyer", 1.4),
            center=(0, 0),
            start_angle=curved_text_kwargs.get("start_angle", -65),
            end_angle=curved_text_kwargs.get("end_angle", -15),
            **{k:v for k,v in curved_text_kwargs.items() if k not in ("r_max_multiplyer","start_angle","end_angle")}
        )
        ax.text(r_max * 0.85, 0.25 * r_max, r"$\sigma/\sigma_{\rm ref}$", ha='left', va='center', **axis_label_kwargs)

        # Legende nur eindeutige Einträge
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
        ax.legend(
            [h for h, _ in unique],
            [l for _, l in unique],
            fontsize=11,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),
            ncol=3,
            frameon=False,
            labelspacing=1.0
        )

        return ax

