import matplotlib.pyplot as plt
import numpy as np
import os

from imgcat import imgcat
import io

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
def save_and_show(filename='latest_plot', figures_dir="figures"):
    # 1. Save to disk
    name=  os.path.join(figures_dir, filename)
    plt.savefig(name+'.png',dpi=300, bbox_inches='tight')
    plt.savefig(name+'.pdf', bbox_inches='tight')
    
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
    df_group = df.groupby(['config.learner.name'])
    methods = [key[0] for key in df_group.keys()]
    colors = ['#1f77b4', '#ff7f0e'] 
    heatmaps = [plt.cm.Blues, plt.cm.Oranges]
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    method_color_map_heat = {method: heatmaps[i % len(heatmaps)] for i, method in enumerate(methods)}
    legend_handles = []
    #norm = mcolors.Normalize(vmin=0, vmax=df[0]['config.task.steps_per_epoch'])
    norm = mcolors.LogNorm(vmin=1, vmax=df[0]['config.task.steps_per_epoch'])
    
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

            # # 1. Create segments for LineCollection: (x, y) to (x_next, y_next)
            # points = np.array([s1, s2]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # # 2. Pick a colormap: Darker colors at higher iterations
            # lc = LineCollection(segments, cmap=cmap_name, norm=norm, linewidth=1., alpha=.7)
            # #print(lol)
            # # 3. Set the values to map the color (iterations)
            # lc.set_array(iters[1:])
                
            # #print(lol)
            # # 4. Add to the Singular Value axis
            # axes[1].add_collection(lc)

            # # Optional: Repeat for the Alignment axis (axes[0])
            # points_a = np.array([a1, a2]).T.reshape(-1, 1, 2)
            # segments_a = np.concatenate([points_a[:-1], points_a[1:]], axis=1)
            # lc_a = LineCollection(segments_a, cmap=cmap_name,norm=norm, linewidth=1., alpha=.7)
            # lc_a.set_array(iters[1:])
            # axes[0].add_collection(lc_a)


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
        if method=="algorithms.SemiGradSSL":
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
                'spectrum': []}
    # Example usage:
    for i in sorted_iterations:
        key = alignments_indices[i]
        data = row['artifact.pickle.'][key].load()
        history['alignment'].append(data['alignment'])
        history['spectrum'].append(data['spectrum'])
        # print(f"Iteration {i}: {data}")

    history['alignment'] = np.array(history['alignment'])
    history['spectrum'] = np.array(history['spectrum'])

    return history


def plot_comparative_spider(res_sce, res_base, iters_to_plot):
    """
    Overlays SCEgrad and Baseline on the same radar chart.
    """
    # 1. Setup indices based on k
    # Assuming both have same k, we pull from the first available iteration
    key_sample = [k for k in res_sce.keys() if "alignments_" in k][0]
    k = len(res_sce[key_sample].load()['alignment'])
    
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # 2. Define Color Palettes
    # SCEgrad = Blues/Greens (Cool) | Baseline = Reds/Oranges (Warm)
    colors_sce = plt.cm.viridis(np.linspace(0.4, 0.9, len(iters_to_plot)))
    colors_base = plt.cm.magma(np.linspace(0.4, 0.7, len(iters_to_plot)))

    for idx, t in enumerate(iters_to_plot):
        # Extract alignment for SCEgrad
        key_sce = f"alignments_{t}"
        val_sce = res_sce[key_sce].load()['alignment'].tolist()
        val_sce += val_sce[:1]
        
        # Extract alignment for Baseline
        key_base = f"alignments_{t}"
        val_base = res_base[key_base].load()['alignment'].tolist()
        val_base += val_base[:1]

        # Plot SCEgrad (Solid lines)
        ax.plot(angles, val_sce, color=colors_sce[idx], linewidth=2.5, 
                label=f"SCEgrad Iter {t}" if idx == len(iters_to_plot)-1 else "")
        ax.fill(angles, val_sce, color=colors_sce[idx], alpha=0.05)

        # Plot Baseline (Dashed lines)
        ax.plot(angles, val_base, color=colors_base[idx], linewidth=2, linestyle='--',
                label=f"Baseline Iter {t}" if idx == len(iters_to_plot)-1 else "")

    # 3. Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Eig {i+1}" for i in range(k)], fontsize=10)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.title("Spectral Escape: SCEgrad vs Baseline\n(Solid = SCEgrad, Dashed = Baseline)", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.show()

# Usage:
# iters = [100, 1000, 5000] # Pick a few key milestones
# plot_comparative_spider(row_sce['artifact.pickle.'], row_base['artifact.pickle.'], iters)




import matplotlib.pyplot as plt
import numpy as np

def plot_weighted_spider(res_sce, iter_t):
    # 1. Calculate weights: (s_i - lambda)_+
    weights = res_sce["target_spectrum"].load().numpy()

    
    # We only care about spokes where weight > 0
    active_indices = np.where(weights > 0)[0]
    active_weights = weights[active_indices]
    
    k_active = len(active_indices)
    angles = np.linspace(0, 2 * np.pi, k_active, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 2. Get data for the specific iteration
    
    
    def get_weighted_vals(res, iteration):
        key = f"alignments_{iteration}"
        #print(lol)
        #if key not in res: return [0]*(k_active+1)
        # Weight the alignment by the expected singular value
        alignments = res[key].load()['alignment'][active_indices]
        #print(lol)
        weighted = (alignments * active_weights).tolist()
        return weighted + [weighted[0]]

    for iteration in iter_t:
        sce_vals = get_weighted_vals(res_sce,iteration)
        #base_vals = get_weighted_vals(res_base)
        
        # 3. Plotting
        # Theoretical Max (The full (s-lambda)_+ support)
        target = active_weights.tolist() + [active_weights[0]]
        ax.plot(angles, target, color='gray', linestyle='--', alpha=0.5, label="Theoretical Target")
        ax.fill(angles, target, color='gray', alpha=0.05)

        # SCEgrad
        ax.plot(angles, sce_vals, color='teal', lw=3, label="SCEgrad (Weighted)")
        ax.fill(angles, sce_vals, color='teal', alpha=0.2)

        # Baseline
        #ax.plot(angles, base_vals, color='crimson', lw=2, linestyle='-', label="Baseline (Weighted)")

    # 4. Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Eig {i+1}\n(w={w:.2f})" for i, w in zip(active_indices, active_weights)])
    
    # The max radial limit is now the largest (s_i - lambda)
    ax.set_ylim(0, max(active_weights) * 1.1)
    
    plt.title(f"Energy-Weighted Spectral Alignment (Iter {iter_t})", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


def plot_spider_plots(df, figures_dir="figures"):
    # 1. Setup the figure
    #fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Use standard grouping
    df_group = df.groupby(['config.learner.name', 'config.task.init_scale'])
    methods = list(np.unique([key[0] for key in df_group.keys()]))
    scales =  list(np.unique([key[1] for key in df_group.keys()]))
    colors = ['#1f77b4', '#ff7f0e'] 
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    legend_handles = []
    iters = [0, 1000, 5000, 9900]
    for scale in scales:
        for method in methods:
            #method_df = [df_group[(method,scale)][0] for method in methods]
            plot_weighted_spider(df_group[(method,scale)][0]['artifact.pickle.'], iters)
            save_and_show(f'spider_scale_{scale}_{methods}', figures_dir= figures_dir)








