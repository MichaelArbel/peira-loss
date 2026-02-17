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


def get_history(row, step=10000):
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
        if np.mod(i,step)==0:
            key = alignments_indices[i]
            data = row['artifact.pickle.'][key].load()
            history['alignment'].append(data['alignment'])
            history['spectrum'].append(data['spectrum_ratio'])
            history['timesteps'].append(i)
        # print(f"Iteration {i}: {data}")

    history['alignment'] = np.array(history['alignment'])
    history['spectrum'] = np.array(history['spectrum'])
    history['timesteps'] = np.array(history['timesteps'])

    spectrum = np.zeros_like(history['alignment'])
    min_shape = min(history['alignment'].shape[1], history['spectrum'].shape[1])
    spectrum[:,:min_shape] = history['spectrum'][:,:min_shape]
    history['spectrum'] = spectrum
    # Assign the original values to the first two slots
    #history['target_spectrum'] = data['target_ratio']
    history['target_spectrum'] = np.ones(history['alignment'].shape[1])#data['target_ratio']
    history['target_spectrum'][4:] = 0.
    return history

def heatmap_log(full_heatmap, x):
    if np.any(x <= 0):
    # common pragmatic choice:
        x = x - x.min() + 1.0

    N, T = full_heatmap.shape

    # choose how many columns you want in the log-resampled image (often same as T)
    T_log = T

    # log-spaced x grid covering same range
    x_log = np.exp(np.linspace(np.log(x[0]), np.log(x[-1]), T_log))

    # resample each row using 1D linear interpolation
    heatmap_log = np.empty((N, T_log), dtype=float)
    for i in range(N):
        heatmap_log[i] = np.interp(x_log, x, full_heatmap[i])
    return heatmap_log, x_log



def plot_run_spectral_stack(
    df,
    method_label,
    figures_dir,
    data="spectrum",
    filename=None,
    vmin=1e-4, 
    vmax=1.,
    percent_max_iter=1.0,
    percent_target=0.05):
    """
    Plots a stacked heatmap of singular value evolution over time.
    Y-axis is partitioned by initialization scale.
    """
    # 1. Setup the grid (Rows = Methods, Cols = Scales)
    df_group  = df.groupby(['config.task.init_scale'])
    init_scales =  sorted(list(np.unique([key[0] for key in df_group.keys()])))
    num_scales = len(init_scales)
    heatmap_data = []
    scale_labels = []
    
    fig, axes = plt.subplots(num_scales, 1, figsize=(8, 1.2 * num_scales), 
                             sharex=True, gridspec_kw={'hspace': 0.2})
    # Sort scales from smallest to largest
    sorted_scales = sorted(init_scales, reverse=True)
        
    for i, scale in enumerate(sorted_scales):
        ax = axes[i]
        heatmap_data = []
        target_data = []
        # Safely extract data
        try:
            scale_df = df_group[(scale,)][0]
            history = get_history(scale_df)
            max_iter = int(history[data].shape[0]*percent_max_iter)
            # Get (Time, 30) -> Slice to (Time, num_top_sv) -> Transpose to (num_top_sv, Time)
            num_top_sv = history[data].shape[1]
            sv_evolution = history[data][:max_iter,:].T
            offset = 0
            #if data == 'spectrum':
            target_col = history['target_spectrum'][:num_top_sv].reshape(-1, 1)
            offset = int(percent_target*sv_evolution.shape[1])
                
            timesteps = history['timesteps'][:max_iter]
            timesteps[timesteps == 0] = 1e-1
            sv_evolution, timesteps =  heatmap_log(sv_evolution, timesteps)
            target_block = np.tile(target_col, (1, offset))
            sv_evolution = np.hstack([sv_evolution,target_block])



            target_data.append(target_block)
            heatmap_data.append(sv_evolution)
            exponent = int(np.log10(scale))
            if exponent==0:
                scale_label = rf"$\rho_0=$1"
            else:
                scale_label = rf"$\rho_0=$10{to_superscript(exponent)}"
            scale_labels.append(scale_label)
        except KeyError:
            print(f"Warning: No data for {method_label} at scale {scale}")
            continue

        if not heatmap_data:
            continue

        # Create the vertical stack
        full_heatmap = np.vstack(heatmap_data)
        target_heatmap = np.vstack(target_data)
        full_heatmap = np.clip(full_heatmap, vmin, vmax)
        target_heatmap = np.clip(target_heatmap, vmin, vmax)
        # Plotting
        # if data=='spectrum':

        #     norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        # else:
        norm = mcolors.Normalize(vmin=0., vmax=1.)
        img = ax.imshow(full_heatmap, aspect='auto', cmap='magma',
                                    norm=norm,
                                    extent=[0, len(timesteps) + offset, 0, num_top_sv],
                                    interpolation='nearest')
        rect = patches.Rectangle((len(timesteps), 0), offset, num_top_sv, 
                                     linewidth=3, edgecolor='red', facecolor='none', zorder=10)
        ax.add_patch(rect)
        k = 6
        pows = 10 ** np.arange(0, k)  # 1..1e5
        pows = pows[(pows >= timesteps.min()) & (pows <= timesteps.max())]

        # index of closest timestep to each power of 10
        idx = [int(np.argmin(np.abs(timesteps - v))) for v in pows]

        # (optional) drop duplicates if timesteps are coarse / repeated matches
        idx, pows = zip(*sorted(dict(zip(idx, pows)).items()))

        
        ax.set_xticks(idx)

        ax.set_xticklabels([rf"$10^{{{int(np.log10(v))}}}$" for v in pows], fontsize=15)

        # img = ax.imshow(full_heatmap, aspect='auto', cmap='magma',
        #                             norm=norm,
        #                             extent=[timesteps[0], timesteps[-1], 0, num_top_sv],
        #                             interpolation='nearest')
        
        # img2 = ax.imshow(target_heatmap, aspect='auto', cmap='magma',
        #                             norm=norm,
        #                             extent=[timesteps[-1], timesteps[-1]+ offset, 0, num_top_sv],
        #                             interpolation='nearest')

        # #if data=='spectrum':
        # rect = patches.Rectangle((timesteps[-1], 0), offset, num_top_sv, 
        #                              linewidth=3, edgecolor='red', facecolor='none', zorder=10)
        # ax.add_patch(rect)
        #print(lol)
        # 4. Labeling the Scale on the Left
        fontsize=30
        ax.set_ylabel(scale_label, rotation=0, ha='left', va='center', fontsize=20,labelpad=100)
        
        # Remove internal ticks to make it look like a single stack
        #ax.set_yticks([]) 
        if i < num_scales - 1:
            ax.set_xticks([])

        # # Labeling and Grid

    # Shared colorbar
    axes[-1].set_xlabel("Training iterations",fontsize=fontsize)
    if data=='spectrum':
        title = "Spectral ratio"
    else: 
        title = "Eigenvector alignment"

    axes[0].set_title(f"{title}: {method_label}", pad=20, fontsize=fontsize)
    
    # Add 'Target' text over the top box
    if data=='spectrum':
        axes[0].text(len(timesteps) + 0.5, num_top_sv * 1.1, "Target", 
                     color='black', fontweight='bold', ha='center', fontsize=15)

    cbar = fig.colorbar(img, ax=axes, aspect=20, pad=0.02)

    # Set the main side-label font size
    cbar.set_label(f'Magnitude', size=fontsize)

    # Set the font size of the numbers (ticks)
    cbar.ax.tick_params(labelsize=20)
    save_and_show(f'{data}_image_{method_label}', figures_dir= figures_dir)









def plot_run_grad_norm(df, method_label, figures_dir, filename=None):
    """
    Plot gradient norm vs iterations for a single run.
    """
    df_group = df.groupby(['config.task.init_scale'])
    init_scales =  sorted(list(np.unique([key[0] for key in df_group.keys()])))
    cmap = plt.get_cmap('magma') 
    norm = mcolors.LogNorm(vmin=min(init_scales), vmax=max(init_scales))


    fig, ax = plt.subplots(figsize=(4, 3))
    step = 100
    for j, scale in enumerate(init_scales): 
        row = df_group[(scale,)][0]
        grad_norm = np.array(row['metrics.grad_norm'])
        iters = np.array(row['metrics.iter'])
        color = cmap(norm(scale))
        ax.plot(iters[::step], grad_norm[::step], color=color, lw=1, alpha=1.)
    fontsize = 30
    ax.set_ylabel(r"$\|\dot{U}_t\|_2$", fontsize=fontsize)
    ax.set_xlabel("Training iterations", fontsize=fontsize)
    
                # 3. Add a Colorbar to show the scale progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Initialization scale $\rho_0$",fontsize=fontsize)


    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(bottom=1e-10, top=1e0)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_title(f"Vector field norm: {method_label}", fontsize=fontsize, pad=10)
    plt.tight_layout()

    filename  = f"grad_norm_{method_label}"
    save_and_show(filename, figures_dir=figures_dir)
    plt.close(fig)


def plot_method_standalone_figures(df, figures_dir):
    """
    For each method, generate standalone figures for:
    - spectrum evolution
    - alignment evolution
    - gradient norm vs iterations
    Uses one representative run per method (first row in that group).
    """
    method_key = _method_key(df)
    df_group = df.groupby([method_key])
    methods = list(np.unique([key[0] for key in df_group.keys()]))

    colors = ['#1f77b4', '#ff7f0e'] 
    heatmaps = [plt.cm.Blues, plt.cm.Oranges]
    #method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    cmaps = {method: heatmaps[i % len(heatmaps)] for i, method in enumerate(methods)}
    method_label = {
        "csrlearning.algorithms.CSEExactRegressionAlgorithm": "CSE",
        "csrlearning.algorithms.SSLExactSemiGrad": "SG-SSL",
        "csrlearning.algorithms.UnconstrainedCCAAlgorithm": "EY",
    }




    for key, group in df_group.items():
        method = method_label[key[0]]
        plot_run_spectral_stack(
            group,
            method,
            figures_dir=figures_dir,
            data="spectrum",
            filename=f"spectrum_image_",
        )
        plot_run_spectral_stack(
            group,
            method,
            figures_dir=figures_dir,
            data="alignment",
            filename=f"alignment_image_",
        )
        plot_run_grad_norm(
            group,
            method,
            figures_dir=figures_dir,
            filename=f"grad_norm_",
        )
