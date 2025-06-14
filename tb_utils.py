# tb_utils.py

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from typing import Dict, List, Sequence, Union

def load_tensorboard_scalars(logdir: str, scalar_name: str):
    """
    Lädt TensorBoard-Skalare aus einer gegebenen Logdatei.

    :param logdir: Pfad zu den TensorBoard-Logs (dem Verzeichnis, in dem
                   'events.out.tfevents.xxxx' Dateien liegen).
    :param scalar_name: Name des Skalaren, der extrahiert werden soll
                        (z.B. 'train/loss/total' oder 'val/loss/recon').
    :return: Zwei Listen: steps und values.
    """
    # lade alle events, keine Größenbegrenzung
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if scalar_name not in tags:
        raise ValueError(f"Scalar '{scalar_name}' nicht in {logdir}. Gefundene: {tags}")

    events = ea.Scalars(scalar_name)
    steps  = [e.step  for e in events]
    vals   = [e.value for e in events]
    return steps, vals


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
    Plots validation losses (log y), annealing weights (log y),
    and a separate phase timeline in three stacked panels.
    Adds vertical lines at phase boundaries, and optional epoch ticks every `epoch_interval`.
    """
    fig_kwargs  = fig_kwargs or {
        "figsize": (14, 10),
        "gridspec_kw": {"height_ratios": [3, 2, 0.5]}
    }
    plot_kwargs = plot_kwargs or {"alpha": 0.8, "linewidth": 1.5}
    phase_bounds = {
        name: (start * epoch_length, end * epoch_length)
        for name, (start, end) in phase_bounds_ep.items()
        }

    fig, (ax_loss, ax_ann, ax_phase) = plt.subplots(3, 1, sharex=True, **fig_kwargs)

    # PANEL 1: Losses (log)
    ax_loss.set_yscale("log")
    colors = ["C0","C1","C2","C3","C4"]
    linestyles = ["-","--",":","-.","."]
    handles, labels = [], []
    max_step = 0

    for idx, tag in enumerate(loss_tags):
        for exp, vers in experiments.items():
            for i, v in enumerate(vers):
                steps, vals = load_fn(os.path.join(base_dir, exp, f"version_{v}"), tag)
                steps = np.array(steps)
                vals  = np.array(vals)
                mask = vals > 0
                if not mask.any():
                    continue
                max_step = max(max_step, steps.max())
                h, = ax_loss.plot(
                    steps[mask], vals[mask],
                    color=colors[idx],
                    linestyle=linestyles[i % len(linestyles)],
                    label=tag.split("/")[-1] if (i==0 and exp==list(experiments)[0]) else "",
                    **plot_kwargs
                )
                if i==0 and exp==list(experiments.keys())[0]:
                    handles.append(h)
                    labels.append(tag.split("/")[-1])

    ax_loss.set_ylabel("Validation Loss (log)")
    ax_loss.set_title("Validation Losses")
    # Phase boundary lines
    for _, (s,e) in phase_bounds.items():
        ax_loss.axvline(e, color='gray', linestyle='--', alpha=0.7)

    # PANEL 2: Annealing (log)
    ax_ann.set_yscale("log")
    for idx, tag in enumerate(anneal_tags):
        for exp, vers in experiments.items():
            for i, v in enumerate(vers):
                steps, vals = load_fn(os.path.join(base_dir, exp, f"version_{v}"), tag)
                steps = np.array(steps)
                max_step = max(max_step, steps.max())
                ax_ann.plot(
                    steps, vals,
                    color=colors[idx],
                    linestyle=linestyles[i % len(linestyles)],
                    label=tag.split("/")[-1] if (i==0 and exp==list(experiments)[0]) else "",
                    **plot_kwargs
                )
    ax_ann.set_ylabel("Annealing Weight (log)")
    ax_ann.legend(loc="upper left", fontsize="small")
    # Phase lines in annealing too
    for _, (s,e) in phase_bounds.items():
        ax_ann.axvline(e, color='gray', linestyle='--', alpha=0.7)

    # PANEL 3: Phase Timeline
    ax_phase.set_ylim(0, len(phase_bounds))
    ax_phase.set_yticks(np.arange(len(phase_bounds)) + 0.5)
    ax_phase.set_yticklabels(list(phase_bounds.keys()))
    ax_phase.set_xlabel("Training Steps" if not epoch_length else "Epoch")
    phase_colors = plt.cm.Pastel1.colors
    for j, (name, (s,e)) in enumerate(phase_bounds.items()):
        ax_phase.broken_barh([(s, e-s)], (j,1),
                             facecolor=phase_colors[j % len(phase_colors)],
                             edgecolor='k', alpha=0.6)

    # Optional: Epoch ticks every epoch_interval
    if epoch_length:
        max_epoch = int(np.ceil(max_step / epoch_length))
        # nur jeden epoch_interval
        epoch_idxs = np.arange(0, max_epoch+1, epoch_interval) * epoch_length
        epoch_lbls = (np.arange(0, max_epoch+1, epoch_interval)).astype(int)
        for ax in (ax_loss, ax_ann, ax_phase):
            ax.set_xticks(epoch_idxs)
            ax.set_xticklabels(epoch_lbls)
    
    # Common legend for losses + phases
    phase_patches = [mpatches.Patch(facecolor=phase_colors[i%len(phase_colors)],
                                    alpha=0.6, label=name)
                     for i, name in enumerate(phase_bounds)]
    ax_loss.legend(ncol=2, fontsize="small")

    plt.tight_layout()
    return fig
