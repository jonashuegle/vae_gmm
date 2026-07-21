# tb.utils.py
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Tuple




def load_tensorboard_scalars(
    logdir: str,
    scalar_name: str,
    epoch_length: int = None,
    offset: int = None,
):
    """
    Lädt TensorBoard-Skalare und rechnet optional global steps → epochen um.
    
    :param logdir: Pfad zu den TensorBoard-Logs.
    :param scalar_name: Name des Skalaren, z.B. 'val/loss/recon'.
    :param epoch_length: Anzahl Schritte pro Epoche. Wenn None, wird kein epoch array erzeugt.
    :param offset: Erster geloggter Step, der als Epoche 0 interpretiert wird. 
                   Wenn None, wird offset = min(steps).
    :return: steps (np.array), values (np.array), epochs (np.array oder None)
    """
    # TensorBoard laden
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if scalar_name not in tags:
        raise ValueError(f"Scalar '{scalar_name}' nicht in {logdir}. Gefundene: {tags}")

    events = ea.Scalars(scalar_name)
    steps  = np.array([e.step  for e in events], dtype=int)
    vals   = np.array([e.value for e in events], dtype=float)

    # Offset bestimmen, falls nicht übergeben
    if epoch_length is not None:
        if offset is None:
            offset = steps.min()
        # Epoche = (step – offset) / epoch_length
        epochs = (steps - offset) / epoch_length
    else:
        epochs = None

    return steps, vals, epochs


def plot_training_diagnostics(
    base_dir: str,
    experiments: dict[str, list[int]],
    loss_tags: list[str],
    anneal_tags: list[str],
    phase_bounds_ep: dict[str, tuple[int,int]],
    load_fn,
    epoch_length: int = None,
    epoch_interval: int = 10,
    fig_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> plt.Figure:
    """
    Plots validation losses (log y), annealing weights (log y)
    und eine Phase-Timeline in Epochen (oder in Steps, falls epoch_length=None).
    
    load_fn liefert: steps, vals, epochs = load_fn(logdir, tag, epoch_length, offset=None)
    wobei epochs ein numpy-Array ist, das ab 0 zählt, wenn epoch_length übergeben wurde.
    """

    tag2label = {
        "val/loss/recon":          "Reconstruction Loss",
        "val/loss/global_kld":     "Global KLD Loss",
        "val/loss/cluster_kld":    "Cluster KLD Loss",
        "val/loss/cat_kld":        "Categorical KLD Loss",
        "val/loss/var_reg":        "Variance\n Regularization Loss",
        "annealing/vae_factor":    "VAE KLD Factor",
        "annealing/gmm_factor":    "GMM KLD Factor",
        "annealing/cat_factor":    "Categorical\n Regularization Factor",
        "annealing/reg_factor":    "Variance\n Regularization Factor",
    }

    tag2color = {
        "val/loss/recon":          "C0",
        "val/loss/global_kld":     "C1",
        "val/loss/cluster_kld":    "C2",
        "val/loss/cat_kld":        "C3",
        "val/loss/var_reg":        "C4", 
        "annealing/vae_factor":    "C1",
        "annealing/gmm_factor":    "C2",
        "annealing/cat_factor":    "C3",
        "annealing/reg_factor":    "C4",
    }



    # 1) Defaults
    fig_kwargs  = fig_kwargs or {
        "figsize": (14, 10),
        "gridspec_kw": {"height_ratios": [3, 2, 0.5]}
    }
    plot_kwargs = plot_kwargs or {"alpha": 0.8, "linewidth": 1.5}

    # 2) Setup Subplots
    fig, (ax_loss, ax_ann, ax_phase) = plt.subplots(
        3, 1, sharex=True, **fig_kwargs
    )

    subplot_labels = [f"({c})" for c in string.ascii_lowercase]

    box_style = {
        "boxstyle": "round,pad=0.16",
        "fc": "white",
        "ec": "white",
        "lw": 0.8,
        "alpha": 0.4
    }
    ax_loss.text(
        0.02, 0.95, subplot_labels[0], transform=ax_loss.transAxes,
        fontsize="large", fontweight='bold', va='top', ha='left',
        bbox=box_style,
        zorder=10
    )
    ax_ann.text(
        0.02, 0.95, subplot_labels[1], transform=ax_ann.transAxes,
        fontsize="large", fontweight='bold', va='top', ha='left',
        bbox=box_style,
        zorder=10
    )
    ax_phase.text(
        0.02, 0.95, subplot_labels[2], transform=ax_phase.transAxes,
        fontsize="large", fontweight='bold', va='top', ha='left',
        bbox=box_style,
        zorder=10
    )

    colors     = ["C0","C1","C2","C3","C4"]
    linestyles = ["-","--",":","-.","."]

    max_x = 0.0

    # --- PANEL 1: Validation Losses (log y) ---
    ax_loss.set_yscale("log") 
    for idx, tag in enumerate(loss_tags):
        color = tag2color.get(tag, f"C{idx}")
        label = tag2label.get(tag, tag)
        for exp, vers in experiments.items():
            for i, v in enumerate(vers):
                logdir = os.path.join(base_dir, exp, f"version_{v}")
                steps, vals, epochs = load_fn(
                    logdir, tag,
                    epoch_length=epoch_length,
                    offset=None
                )
                x = epochs if (epochs is not None) else np.array(steps)
                vals = np.array(vals)

                mask = vals > 0
                if not mask.any():
                    continue

                max_x = max(max_x, float(np.max(x[mask])))

                ax_loss.plot(
                    x[mask], vals[mask],
                    color=color,
                    linestyle=linestyles[i % len(linestyles)],
                    label=label if (i==0 and exp==list(experiments)[0]) else "",
                    **plot_kwargs
                )

    ax_loss.set_ylabel("Validation Loss (log)")
    #ax_loss.set_title("Validation Losses")
    for _, (s_ep, e_ep) in phase_bounds_ep.items():
        ax_loss.axvline(e_ep, color='gray', linestyle='--', alpha=0.7)


    # --- PANEL 2: Annealing Weights (log y) ---
    ax_ann.set_yscale("log")
    for idx, tag in enumerate(anneal_tags):
        color = tag2color.get(tag, f"C{idx}")
        label = tag2label.get(tag, tag)
        for exp, vers in experiments.items():
            for i, v in enumerate(vers):
                logdir = os.path.join(base_dir, exp, f"version_{v}")
                steps, vals, epochs = load_fn(
                    logdir, tag,
                    epoch_length=epoch_length,
                    offset=None
                )
                x = epochs if (epochs is not None) else np.array(steps)
                vals = np.array(vals)

                max_x = max(max_x, float(np.max(x)))

                ax_ann.plot(
                    x, vals,
                    color=color,
                    linestyle=linestyles[i % len(linestyles)],
                    label=label if (i==0 and exp==list(experiments)[0]) else "",
                    **plot_kwargs
                )

    ax_ann.set_ylabel("Annealing Weight (log)")
    ax_ann.legend(loc="lower right", fontsize="small")
    for _, (s_ep, e_ep) in phase_bounds_ep.items():
        ax_ann.axvline(e_ep, color='gray', linestyle='--', alpha=0.7)


    # --- PANEL 3: Phase Timeline ---
    ax_phase.set_ylim(0, len(phase_bounds_ep))
    ax_phase.set_yticks(np.arange(len(phase_bounds_ep)) + 0.5)
    ax_phase.set_yticklabels(list(phase_bounds_ep.keys()))
    ax_phase.set_xlabel("Epoch" if epoch_length else "Training Steps")
    phase_colors = plt.cm.Pastel1.colors
    ax_phase.set_autoscale_on(False)

    for j, (name, (s_ep, e_ep)) in enumerate(phase_bounds_ep.items()):
        ax_phase.broken_barh(
            [(s_ep, e_ep - s_ep)],
            (j, 1),
            facecolor=phase_colors[j % len(phase_colors)],
            edgecolor='k',
            alpha=0.6
        )


    # --- Gemeinsame X-Limits & Ticks ---
    for ax in (ax_loss, ax_ann, ax_phase):
        ax.set_xlim(0, max_x)

    if epoch_length:
        ticks     = np.arange(0, np.ceil(max_x)+1, epoch_interval)
        tick_lbls = ticks.astype(int)
    else:
        ticks     = np.arange(0, max_x+1, epoch_interval)
        tick_lbls = ticks

    for ax in (ax_loss, ax_ann, ax_phase):
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lbls)

    # --- Gemeinsame Legende für Loss-Tags ---
    handles, labels = ax_loss.get_legend_handles_labels()
    #ax_loss.legend(handles, labels, ncol=2, fontsize="small", loc ="lower right")
    ax_loss.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        #fontsize="small",
        ncol=1
    )
    ax_ann.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        #fontsize="small",
        ncol=1
    )
    plt.subplots_adjust(right=0.8)
   # plt.tight_layout()
    return fig