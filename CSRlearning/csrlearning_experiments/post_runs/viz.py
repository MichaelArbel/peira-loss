import matplotlib.pyplot as plt
import numpy as np
import os

from imgcat import imgcat
import io

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
from typing import Optional
superscripts = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "-": "⁻", "=": "⁼", 
}

def to_superscript(n):
    return "".join(superscripts[c] for c in str(n))

def _method_key(df):
    return "config.algorithm.name" 





def _is_successful_row(row):
    status_keys = [
        "info.status",
        "info.mlxp.status",
        "mlxp.status",
    ]
    success_keys = [
        "info.succeeded",
        "info.mlxp.succeeded",
        "mlxp.succeeded",
    ]
    for key in success_keys:
        value = _safe_get(row, key)
        if value is not None:
            return bool(value)
    for key in status_keys:
        value = _safe_get(row, key)
        if value is not None:
            return str(value).upper() in {"COMPLETE", "SUCCEEDED", "SUCCESS"}
    return True


def _row_id(row):
    return row['info.logger.log_id']

def _row_method(row):
    return row['config.algorithm.name']

def _sanitize_filename(name):
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)



def save_and_show(filename='latest_plot', figures_dir="figures"):
    # 1. Save to disk
    name=  os.path.join(figures_dir, filename)
    plt.savefig(name+'.png',dpi=300, bbox_inches='tight')
    #plt.savefig(name+'.pdf', bbox_inches='tight')
    
    # 2. Display in terminal
    imgcat(open(name+'.png', 'rb').read())

def add_arrows(s1, s2, axis, arrow_steps, n_points, color, fixed_pixel_length=0.01):
    for idx in arrow_steps:
        if idx + 1 < n_points:  # Ensure we have a "next" point to calculate direction
            # 1. Calculate direction vector for Singular Values
            p1 = axis.transData.transform((s1[idx], s2[idx]))
            p2 = axis.transData.transform((s1[idx+1], s2[idx+1]))

            # 2. Calculate the visual direction vector in pixels
            dp = p2 - p1
            norm_ds = np.linalg.norm(dp)


            # Normalization for Log/Symlog axes
            # ds1 = np.log10(s1[idx+1] + 1e-9) - np.log10(s1[idx] + 1e-9)
            # ds2 = np.log10(s2[idx+1] + 1e-9) - np.log10(s2[idx] + 1e-9)
            # norm_ds = np.sqrt(ds1**2 + ds2**2)
            if norm_ds > 1e-9: # Only plot if there's movement
                u_dp = dp / norm_ds

                # 4. Define how long you want the arrow to be ON SCREEN (e.g., 20 pixels)
                fixed_pixel_length = fixed_pixel_length
                p2_fixed = p1 + u_dp * fixed_pixel_length
                
                # 5. Convert this 'fixed length' point back into data units
                target_data = axis.transData.inverted().transform(p2_fixed)
                
                # 6. Use annotate with a simple style to avoid hexagons/blockiness
                axis.annotate('', 
                                 xy=target_data,      # Head of arrow
                                 xytext=(s1[idx], s2[idx]), # Tail of arrow
                                 arrowprops=dict(
                                     arrowstyle='->', # Clean line style (no hexagons)
                                     color=color, 
                                     lw=1.5, 
                                     mutation_scale=15 # Size of the arrow head
                                 ),
                                 zorder=20)


def plot_phase_portrait(df, figures_dir="figures"):
    # 1. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Use standard grouping
    method_key = _method_key(df)
    df_group = df.groupby([method_key])
    methods = [key[0] for key in df_group.keys()]
    colors = ['#1f77b4', '#ff7f0e'] 
    heatmaps = [plt.cm.Blues, plt.cm.Oranges]
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    method_color_map_heat = {method: heatmaps[i % len(heatmaps)] for i, method in enumerate(methods)}
    legend_handles = []
    #norm = mcolors.Normalize(vmin=0, vmax=df[0]['config.task.steps_per_epoch'])
    norm = mcolors.LogNorm(vmin=1, vmax=df[0]['config.task.total_steps'])
    
    for method in methods:
        method_df = df_group[(method,)]
        color = method_color_map[method]
        cmap_name = method_color_map_heat[method]
        # Lists to store trajectories for mean/std calculation
        all_a1, all_a2 = [], []
        all_s1, all_s2 = [], []
        all_grads = []
        all_iters = []

        for row in method_df:
            # Data extraction
            a1, a2 = np.array(row['metrics.align_1']), np.array(row['metrics.align_2'])
            s1, s2 = np.array(row['metrics.singval_1']), np.array(row['metrics.singval_2'])
            iters = np.array(row['metrics.iter'])
            grad_norm = np.array(row['metrics.running_grad_norm'])

            axes[0].plot(a1, a2, color=color, lw=1,alpha=1.)
            axes[1].plot(s1, s2, color=color, lw=1,alpha=1.)


                # --- Add Directional Arrows ---
            # Pick specific indices to place arrows (e.g., every 20% of the trajectory)
            # We use a subset of indices to avoid clutter
            n_points = len(s1)
            arrow_steps = [3, 10, 100]
            add_arrows(s1, s2, axes[1], arrow_steps,  n_points, color,fixed_pixel_length=0.01)
            add_arrows(a1, a2, axes[0], arrow_steps, n_points, color, fixed_pixel_length=0.01)

            # Plot individual trajectories with high transparency
            #line, = axes[0].plot(a1, a2, color=color, lw=1.)
            #axes[1].plot(s1, s2, color=color, lw=1.)

            line, = axes[2].plot(iters, grad_norm, color=color, lw=1,alpha=1.)

            # Store for averaging (assuming same lengths, or interpolate if needed)

            # Start and End Markers for individual runs
            axes[0].scatter(a1[0], a2[0], color='black', s=15, marker='o')
            axes[0].scatter(a1[-1], a2[-1], color=color, s=25, marker='x')
            axes[1].scatter(s1[0], s2[0], color='black', s=15, marker='o')
            axes[1].scatter(s1[-1], s2[-1], color=color, s=25, marker='x')

        # # Add this after you've added the collections to the axes
        # sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        # sm.set_array([]) # Required for older matplotlib versions
        # cbar = fig.colorbar(sm, ax=axes[0])
        # cbar.set_label('Iteration Value')

        target_spectrum = row['artifact.pickle.']['target_spectrum'].load()
        axes[1].scatter(target_spectrum[0], target_spectrum[1], color=color, s=300, marker='*', label='Target', edgecolors='black', zorder=10)
        if "SemiGradientSSL" in method:
            axes[1].scatter(target_spectrum[0],0, color='tab:red', s=200, marker=r'$\oslash$',label='Collapsed fixed point', edgecolors='black', zorder=10)


        legend_handles.append(line)

    # --- Formatting ---
    
    # 1. Alignment Trajectory
    axes[0].set_title("Subspace Alignment Trajectory", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Align 1 (Cosine)")
    axes[0].set_ylabel("Align 2 (Cosine)")
    # Ground truth reference
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].scatter(1, 1, color='gold', s=300, marker='*', label='Target', edgecolors='black', zorder=10)
    axes[0].set_aspect('equal') # Standardize aspect ratio
    axes[0].set_xlim(-0.05, 1.1); axes[0].set_ylim(-0.05, 1.1)

    axes[1].scatter(0, 0, color='tab:red', s=100, marker=r'$\oslash$',label='Collapsed fixed point', edgecolors='black', zorder=10)


    # 2. Singular Value Trajectory
    axes[1].set_title("Singular Value Trajectory", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Singular Value 1")
    axes[1].set_ylabel("Singular Value 2")
    #axes[1].set_xlim([0,10])
    lin_thresh = 0.0000001
    axes[1].set_xscale('symlog', linthresh=lin_thresh)
    axes[1].set_yscale('symlog', linthresh=lin_thresh)



    # --- Integration of your Theoretical Fixed Point ---
    # Replace these with the actual Corr and thresh values from your config if needed
    # val = 0.5 * (Corr + sqrt(max(0, Corr**2 - 4*thresh))) if thresh >= 0.25 else 0
    # Here we plot the targets as red crosses if you have them calculated:
    # axes[1].scatter(target_s1, target_s2, color='red', marker='+', s=150, zorder=12, label='Analytical FP')

    # 3. Convergence Rate
    axes[2].set_title("Convergence Rate", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Running Grad Norm")
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].set_ylim(bottom=1e-10)

    for ax in axes:
        ax.grid(True, which="both", linestyle=':', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Clean Legend
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
               ncol=len(methods), frameon=False, fontsize=12)

    plt.tight_layout()
    save_and_show('trajectory_comparison', figures_dir= figures_dir)


def get_history(row):
    import re

    # 1. Get all keys
    all_keys = row['artifact.pickle.'].keys()

    # 2. Extract and parse keys starting with 'alignments_'
    # We split by '_' and take the last part, converting it to an int
    alignments_indices = {
        int(k.split('_')[-1]): k 
        for k in all_keys if k.startswith('alignments_')
    }

    # 3. Sort by iteration number to ensure chronological order
    sorted_iterations = sorted(alignments_indices.keys())
    history = {'alignment': [],
                'spectrum': [],
                'timesteps': []}
    # Example usage:
    for i in sorted_iterations:
        key = alignments_indices[i]
        data = row['artifact.pickle.'][key].load()
        history['alignment'].append(data['alignment'])
        history['spectrum'].append(data['spectrum_ratio'])
        history['timesteps'].append(i)
        # print(f"Iteration {i}: {data}")

    history['alignment'] = np.array(history['alignment'])
    history['spectrum'] = np.array(history['spectrum'])
    history['timesteps'] = np.array(history['timesteps'])
    # Assign the original values to the first two slots
    history['target_spectrum'] = data['target_ratio']
    #history['target_spectrum'] = np.ones(history['spectrum'].shape[1])#data['target_ratio']

    return history


def plot_run_spectral_stack(
    row,
    figures_dir,
    data="spectrum",
    filename=None,
    vmin=None,
    vmax=None,
    percent_max_iter=1.0,
    percent_target=0.05,
):
    """
    Plot spectrum/alignment evolution for a single run, as a standalone figure.
    """
    history = get_history(row)
    if data not in history:
        raise KeyError(f"'{data}' is not available in history.")

    values = history[data]
    max_iter = max(1, int(values.shape[0] * percent_max_iter))
    values = values[:max_iter, :]
    evolution = values.T
    timesteps = history["timesteps"][:max_iter].copy()
    timesteps[timesteps == 0] = 1e-1

    offset = 0
    if data == "spectrum":
        target = history["target_spectrum"][: evolution.shape[0]].reshape(-1, 1)
        offset = max(1, int(percent_target * evolution.shape[1]))
        target_block = np.tile(target, (1, offset))
        evolution = np.hstack([evolution, target_block])

    if data == "spectrum":
        vmin = 1e-6 if vmin is None else vmin
        vmax = 10.0 if vmax is None else vmax
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        title = "Spectral Evolution"
        cbar_label = "Spectrum magnitude"
    else:
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        title = "Alignment Evolution"
        cbar_label = "Alignment magnitude"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    img = ax.imshow(
        np.clip(evolution, vmin, vmax),
        aspect="auto",
        cmap="magma",
        norm=norm,
        extent=[0, len(timesteps) + offset, 0, evolution.shape[0]],
        interpolation="nearest",
    )

    if data == "spectrum":
        rect = patches.Rectangle(
            (len(timesteps), 0),
            offset,
            evolution.shape[0],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            len(timesteps) + max(0.2, offset / 2),
            evolution.shape[0] * 1.02,
            "Target",
            color="black",
            fontweight="bold",
            ha="center",
            fontsize=11,
        )

    method = _row_method(row)
    ax.set_xlabel("Training iterations", fontsize=12)
    ax.set_ylabel("Index", fontsize=12)
    ax.set_title(f"{title}: {method}", fontsize=13)
    cbar = fig.colorbar(img, ax=ax, aspect=25, pad=0.02)
    cbar.set_label(cbar_label, size=11)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()

    #if filename is None:
    run_id = _row_id(row)
    filename +=  f"{_sanitize_filename(method)}_{_sanitize_filename(run_id)}"
    save_and_show(filename, figures_dir=figures_dir)
    plt.close(fig)


def plot_run_grad_norm(row, figures_dir, filename=None):
    """
    Plot gradient norm vs iterations for a single run.
    """
    grad_norm = np.array(_safe_get(row, "metrics.running_grad_norm", []))
    iters = np.array(_safe_get(row, "metrics.iter", []))
    if len(grad_norm) == 0 or len(iters) == 0:
        raise ValueError("Run is missing 'metrics.running_grad_norm' or 'metrics.iter'.")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(iters, grad_norm, color="#1f77b4", lw=1.2)
    ax.set_ylabel(r"$\|\dot{U}_t\|_2$", fontsize=12)
    ax.set_xlabel("Training iterations", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(bottom=1e-10, top=1e0)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    method = _row_method(row)
    ax.set_title(method, fontsize=12, pad=10)
    plt.tight_layout()

    if filename is None:
        run_id = _row_id(row)
        filename = f"grad_norm_{_sanitize_filename(method)}_{_sanitize_filename(run_id)}"
    save_and_show(filename, figures_dir=figures_dir)
    plt.close(fig)


def plot_method_standalone_figures(rows, figures_dir):
    """
    For each method, generate standalone figures for:
    - spectrum evolution
    - alignment evolution
    - gradient norm vs iterations
    Uses one representative run per method (first row in that group).
    """

    for row in rows:
        plot_run_spectral_stack(
            row,
            figures_dir=figures_dir,
            data="spectrum",
            filename=f"spectrum_image_",
        )
        plot_run_spectral_stack(
            row,
            figures_dir=figures_dir,
            data="alignment",
            filename=f"alignment_image_",
        )
        plot_run_grad_norm(
            row,
            figures_dir=figures_dir,
            filename=f"grad_norm_",
        )
