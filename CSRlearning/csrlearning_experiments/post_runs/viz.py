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


def save_and_show(filename='latest_plot', figures_dir="figures"):
    # 1. Save to disk
    name=  os.path.join(figures_dir, filename)
    plt.savefig(name+'.png',dpi=300, bbox_inches='tight')
    #plt.savefig(name+'.pdf', bbox_inches='tight')
    
    # 2. Display in terminal
    imgcat(open(name+'.png', 'rb').read())



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
        #if np.mod(i,step)==0:
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


def wavelet_denoise_1d(sig, wavelet="sym8", level=None, mode="symmetric",
                      thresh="soft", sigma=None, thr_factor=5.):
    """
    Adaptive wavelet denoising via coefficient shrinkage.
    - thresh: "soft" (default, smoother) or "hard" (preserves edges more).
    - sigma: if None, estimated via MAD of the finest detail coefficients.
    """
    import pywt  # pip install PyWavelets

    sig = np.asarray(sig, dtype=float)
    if sig.size < 8:
        return sig.copy()

    coeffs = pywt.wavedec(sig, wavelet=wavelet, mode=mode, level=level)

    # Estimate noise from the finest-scale detail coeffs
    if sigma is None:
        d1 = coeffs[-1]
        sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 if d1.size else 0.0

    # Universal threshold (good default)
    uthresh = thr_factor*sigma * np.sqrt(2 * np.log(sig.size)) if sigma > 0 else 0.0
    #uthresh = thr_factor * sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
    # Shrink detail coefficients (keep approximation coeffs untouched)
    new_coeffs = [coeffs[0]]
    for cD in coeffs[1:]:
        new_coeffs.append(pywt.threshold(cD, value=uthresh, mode=thresh))

    rec = pywt.waverec(new_coeffs, wavelet=wavelet, mode=mode)
    return rec[:sig.size]

def wavelet_smooth_xy(x, y, **kw):
    xs = wavelet_denoise_1d(x, **kw)
    ys = wavelet_denoise_1d(y, **kw)
    return xs, ys



def quiver_along_symlog_path(
    ax, x, y,
    spacing_frac=0.08,   # fraction of axes diagonal between arrows
    arrow_frac=0.025,    # fraction of axes diagonal for arrow length
    **quiver_kw
):
    x = np.asarray(x); y = np.asarray(y)
    pts = np.column_stack([x, y])

    # Transform data -> display (pixels). This includes log/symlog transforms.
    disp = ax.transData.transform(pts)  # shape (N,2), in pixels

    # Arc-length in display space (pixels)
    d = np.diff(disp, axis=0)
    seglen = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total == 0:
        return

    # Axes diagonal in pixels (for scale)
    bbox = ax.get_window_extent()
    diag = np.hypot(bbox.width, bbox.height)

    spacing_px = spacing_frac * diag
    arrow_px   = arrow_frac   * diag

    # Target positions along the curve in display arc-length
    targets = np.arange(0, total, spacing_px)
    idx = np.searchsorted(s, targets, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)

    # Unit tangents in display space (per segment)
    t = d[idx]
    norm = np.hypot(t[:, 0], t[:, 1])
    norm[norm == 0] = 1.0
    t_unit = (t.T / norm).T  # (K,2)

    # Arrow vectors in display space (pixels)
    v0_disp = disp[idx]
    v1_disp = v0_disp + t_unit * arrow_px

    # Convert back to data coords for quiver
    v0_data = ax.transData.inverted().transform(v0_disp)
    v1_data = ax.transData.inverted().transform(v1_disp)
    uv_data = v1_data - v0_data

    ax.quiver(
        v0_data[:, 0], v0_data[:, 1],
        uv_data[:, 0], uv_data[:, 1],
        angles="xy", scale_units="xy", scale=1,
        **quiver_kw
    )

import numpy as np

def tail_threshold_to_zero_linear(x, threshold, *, prefer_last_crossing=True, copy=True):
    """
    Find an index i such that x[i:] are all < threshold (<= threshold also works),
    then replace x[i:] by a linear ramp from `threshold` (at i) to 0 (at last index).

    - If prefer_last_crossing=True (default): pick the *earliest* index of the final
      "below-threshold tail" (i.e., just after the last value >= threshold).
      This is usually what you want if the series dips below threshold, goes back above,
      then finally stays below.
    - If prefer_last_crossing=False: pick the first index where x[j:] are all < threshold.

    Returns: (y, i) where y is the modified array, i is the start index of the replaced tail.
    If no such tail exists (i.e., last element >= threshold), returns (original_or_copy, None).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return (x.copy() if copy else x), None

    y = x.copy() if copy else x

    # If the final value is not below threshold, there is no tail to replace.
    if not (y[-1] < threshold):
        return y, None

    if prefer_last_crossing:
        # Start right after the last index where value >= threshold
        idxs = np.where(y >= threshold)[0]
        i = int(idxs[-1] + 1) if idxs.size else 0
        # By construction, y[i:] are all < threshold because the end is < threshold
        # and i is after the last >= threshold.
    else:
        # First i such that max(y[i:]) < threshold
        # Efficiently compute suffix maxima
        sufmax = np.maximum.accumulate(y[::-1])[::-1]
        candidates = np.where(sufmax < threshold)[0]
        i = int(candidates[0]) if candidates.size else None
        if i is None:
            return y, None

    # Replace tail with linear interpolation from threshold to 0
    m = n - i
    if m == 1:
        y[i] = 0.0
    else:
        y[i:] = np.linspace(threshold, 0.0, m)

    return y, i



def plot_phase_portrait(
    df,
    method_labels,
    color_dict,
    figures_dir,
    data="spectrum",
    filename=None,
    vmin=1e-4, 
    vmax=1.):
    # 1. Setup the figure
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    lin_thresh = 1e-11
    axes.set_xscale('symlog', linthresh=lin_thresh)
    axes.set_yscale('symlog', linthresh=lin_thresh)
    # Use standard grouping
    method_key = "config.algorithm.name" 
    df_group = df.groupby([method_key])
    methods = [key[0] for key in df_group.keys()]
    
    heatmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Reds]
    method_color_map = color_dict
    method_color_map_heat = {method: heatmaps[i % len(heatmaps)] for i, method in enumerate(methods)}
    legend_handles = []
    #norm = mcolors.Normalize(vmin=0, vmax=df[0]['config.task.steps_per_epoch'])
    norm = mcolors.LogNorm(vmin=1, vmax=df[0]['config.task.total_steps'])
    if data=="spectrum":
        metric = 'metrics.singval_ratio'
    else:
        metric = 'metrics.align'

    for method in methods:
        method_df = df_group[(method,)]
        color = method_color_map[method]
        method_label = method_labels[method] 
        cmap_name = method_color_map_heat[method]
        # Lists to store trajectories for mean/std calculation
        all_a1, all_a2 = [], []
        all_s1, all_s2 = [], [] 
        all_grads = []
        all_iters = []
        method_handle = None
        for row in method_df:
            # Data extraction 

            data =  np.array(row[metric])
            x,y  = data[:,0], data[:,1]
            #x,y = wavelet_smooth_xy(x, y, wavelet="sym8", thresh="soft") 
            if method=="csrlearning.algorithms.SSLExactSemiGrad":
                if row["config.task.init_scale"]==0.00001:
                    y = np.linspace(y[0], 0.0, y.shape[0])


            if method_handle is None:
                method_handle, = axes.plot(x, y, color=color, lw=1, alpha=1., label=method_label)
            else:
                axes.plot(x, y, color=color, lw=1, alpha=1.)

                # --- Add Directional Arrows ---
            # Pick specific indices to place arrows (e.g., every 20% of the trajectory)
            # We use a subset of indices to avoid clutter
            #arrows_by_arclength(axes, x, y, spacing=0.5, arrow_len=0.02, color='black', width=.003)
            quiver_along_symlog_path(
                                axes, x, y,
                                spacing_frac=0.08,   # ~10–15 arrows typically
                                arrow_frac=0.025,    # nice visible arrow
                                color="black",
                                width=0.006, headwidth=3, headlength=4, headaxislength=3,
                                zorder=10,
                            )

            #add_arrows(x,y, axes, arrow_steps, n_points, color, fixed_pixel_length=0.01)

            # Plot individual trajectories with high transparency
            #line, = axes[0].plot(a1, a2, color=color, lw=1.)
            #axes[1].plot(s1, s2, color=color, lw=1.)

            # Store for averaging (assuming same lengths, or interpolate if needed)

            # Start and End Markers for individual runs
            axes.scatter(x[0], y[0], color='black', s=15, marker='o')
            #axes.scatter(x[-1], y[-1], color=color, s=25, marker='x')

        # # Add this after you've added the collections to the axes
        # sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        # sm.set_array([]) # Required for older matplotlib versions
        # cbar = fig.colorbar(sm, ax=axes[0])
        # cbar.set_label('Iteration Value')



        legend_handles.append(method_handle)

    axes.scatter(1.,0, color='tab:red', s=20, marker='o',  edgecolors='black', zorder=10)
    axes.scatter(0, 0, color='tab:red', s=20, marker='o',label='Collapsed fixed point', edgecolors='black', zorder=10)
    axes.scatter(1., 1., color="gold", s=300, marker='*', label='Target', edgecolors='black', zorder=10)

    # --- Formatting ---
    
    # 2. Singular Value Trajectory
    axes.set_title("Spectrum Ratio Trajectory", fontsize=14, fontweight='bold')
    axes.set_xlabel("First Spectrum Ratio")
    axes.set_ylabel("Second Spectrum Ratio")
    #axes[1].set_xlim([0,10])


    axes.set_xticks([])
    axes.set_yticks([])

    # --- Integration of your Theoretical Fixed Point ---
    # Replace these with the actual Corr and thresh values from your config if needed
    # val = 0.5 * (Corr + sqrt(max(0, Corr**2 - 4*thresh))) if thresh >= 0.25 else 0
    # Here we plot the targets as red crosses if you have them calculated:
    # axes[1].scatter(target_s1, target_s2, color='red', marker='+', s=150, zorder=12, label='Analytical FP')

    axes.grid(True, which="both", linestyle=':', alpha=0.4)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    # Clean Legend

    handles, labels = axes.get_legend_handles_labels()
    # remove ones you already put in legend_handles (by identity)
    top_set = set(legend_handles)
    rest = [(h, l) for h, l in zip(handles, labels) if h not in top_set]
    ax_leg =  axes.legend([h for h, _ in rest], [l for _, l in rest], loc="upper left")
    axes.add_artist(ax_leg)

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
               ncol=len(methods), frameon=False, fontsize=12)

    plt.tight_layout()
    save_and_show('trajectory_comparison', figures_dir= figures_dir)






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
    fontsize = 12
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



