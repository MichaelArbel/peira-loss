"""Spectral-ratio and eigenvector-alignment heatmaps for the toy experiments."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

superscripts = dict(zip("0123456789-=", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁼"))


def save_figure(filename, figures_dir):
    plt.savefig(Path(figures_dir) / (filename + ".png"), dpi=300, bbox_inches="tight")


def to_superscript(n):
    return ''.join((superscripts[c] for c in str(n)))

def get_history(row):
    all_keys = row['artifact.pickle.'].keys()
    alignments_indices = {int(k.split('_')[-1]): k for k in all_keys if k.startswith('alignments_')}
    sorted_iterations = sorted(alignments_indices.keys())
    history = {'alignment': [], 'spectrum': [], 'timesteps': []}
    for i in sorted_iterations:
        key = alignments_indices[i]
        data = row['artifact.pickle.'][key].load()
        history['alignment'].append(data['alignment'])
        history['spectrum'].append(data['spectrum_ratio'])
        history['timesteps'].append(i)
    history['alignment'] = np.array(history['alignment'])
    history['spectrum'] = np.array(history['spectrum'])
    history['timesteps'] = np.array(history['timesteps'])
    spectrum = np.zeros_like(history['alignment'])
    min_shape = min(history['alignment'].shape[1], history['spectrum'].shape[1])
    spectrum[:, :min_shape] = history['spectrum'][:, :min_shape]
    history['spectrum'] = spectrum
    history['target_spectrum'] = np.ones(history['alignment'].shape[1])
    history['target_spectrum'][4:] = 0.0
    return history

def heatmap_log(full_heatmap, x):
    if np.any(x <= 0):
        x = x - x.min() + 1.0
    N, T = full_heatmap.shape
    x_log = np.exp(np.linspace(np.log(x[0]), np.log(x[-1]), T))
    heatmap_log = np.empty((N, T), dtype=float)
    for i in range(N):
        heatmap_log[i] = np.interp(x_log, x, full_heatmap[i])
    return (heatmap_log, x_log)

def plot_run_spectral_stack(df, method_label, figures_dir, data="spectrum", *, method_id):
    """Stack the five initialization scales using the original figure layout."""
    groups = df.groupby(["config.task.init_scale"])
    scales = sorted((key[0] for key in groups.keys()), reverse=True)
    fig, axes = plt.subplots(
        len(scales), 1, figsize=(8, 1.2 * len(scales)), sharex=True,
        gridspec_kw={"hspace": 0.2},
    )
    axes = np.atleast_1d(axes)
    for i, scale in enumerate(scales):
        ax = axes[i]
        history = get_history(groups[(scale,)][0])
        evolution = history[data].T
        num_directions = evolution.shape[0]
        offset = int(0.05 * evolution.shape[1])
        target = np.tile(history["target_spectrum"].reshape(-1, 1), (1, offset))
        # Keep the original zero-step handling and interpolation convention.
        timesteps = history["timesteps"]
        timesteps[timesteps == 0] = 0.1
        evolution, timesteps = heatmap_log(evolution, timesteps)
        heatmap = np.clip(np.hstack([evolution, target]), 1e-4, 1.0)
        img = ax.imshow(
            heatmap, aspect="auto", cmap="magma",
            norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
            extent=[0, len(timesteps) + offset, 0, num_directions],
            interpolation="nearest",
        )
        ax.add_patch(patches.Rectangle(
            (len(timesteps), 0), offset, num_directions,
            linewidth=3, edgecolor="red", facecolor="none", zorder=10,
        ))
        powers = 10 ** np.arange(6)
        powers = powers[(powers >= timesteps.min()) & (powers <= timesteps.max())]
        indices = [int(np.argmin(np.abs(timesteps - value))) for value in powers]
        indices, powers = zip(*sorted(dict(zip(indices, powers)).items()))
        ax.set_xticks(indices)
        ax.set_xticklabels([rf"$10^{{{int(np.log10(value))}}}$" for value in powers], fontsize=15)
        exponent = int(np.log10(scale))
        label = rf"$\rho_0=$10{to_superscript(exponent)}"
        ax.set_ylabel(label, rotation=0, ha="left", va="center", fontsize=20, labelpad=100)
        if i < len(scales) - 1:
            ax.set_xticks([])
    axes[-1].set_xlabel("Training iterations", fontsize=30)
    title = "Spectral ratio" if data == "spectrum" else "Eigenvector alignment"
    axes[0].set_title(f"{title}: {method_label}", pad=20, fontsize=30)
    if data == "spectrum":
        axes[0].text(
            len(timesteps) + 0.5, num_directions * 1.1, "Target",
            color="black", fontweight="bold", ha="center", fontsize=15,
        )
    colorbar = fig.colorbar(img, ax=axes, aspect=20, pad=0.02)
    colorbar.set_label("Magnitude", size=30)
    colorbar.ax.tick_params(labelsize=20)
    save_figure(f"{data}_image_{method_id}", figures_dir)
