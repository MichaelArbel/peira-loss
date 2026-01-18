import matplotlib.pyplot as plt
import numpy as np
import os

from imgcat import imgcat
import io

def plot_phase_portrait(trajectories):
    plt.figure(figsize=(8, 6))
    for path in trajectories:
        s1, s2 = zip(*path)
        plt.plot(s1, s2, alpha=0.6)
        # Add an arrow to show direction of "flow"
        plt.arrow(s1[-2], s2[-2], s1[-1]-s1[-2], s2[-1]-s2[-2], 
                  shape='full', lw=0, length_includes_head=True, head_width=0.01)
    
    plt.xlabel('$\sigma_1$ (Leading Singular Value)')
    plt.ylabel('$\sigma_2$ (Second Singular Value)')
    plt.title('Spectral Dynamics: Escape from Zero')
    plt.grid(True)
    plt.show()



def plot_combined_analysis(df):
    # 1. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    df_group = df.groupby(['config.learner.name'])
    methods = [key[0] for key in df_group.keys()]
    colors = ['#1f77b4', '#ff7f0e'] 
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    legend_handles = []

    for i, method in enumerate(methods):
        method_df = df_group[(method,)]
        color = method_color_map[method]
        
        # Legend proxy
        line_handle = plt.Line2D([0], [0], color=color, lw=2, label=f"Method: {method}")
        legend_handles.append(line_handle)

        for row in method_df:
            # Data extraction
            a1, a2 = np.array(row['metrics.align_1']), np.array(row['metrics.align_2'])
            s1, s2 = np.array(row['metrics.singval_1']), np.array(row['metrics.singval_2'])
            iters = np.array(row['metrics.iter'])
            grad_norm = np.array(row['metrics.running_grad_norm'])

            # --- Subplot 1: Alignment ---
            axes[0].plot(a1, a2, color=color, alpha=0.3, lw=1)
            # Start (o) and End (x)
            axes[0].scatter(a1[0], a2[0], color=color, s=20, marker='o', alpha=0.5) 
            axes[0].scatter(a1[-1], a2[-1], color=color, s=50, marker='x', lw=1.5)

            # --- Subplot 2: Singular Values ---
            axes[1].plot(s1, s2, color=color, alpha=0.3, lw=1)
            # Start (o) and End (x)
            axes[1].scatter(s1[0], s2[0], color=color, s=20, marker='o', alpha=0.5)
            axes[1].scatter(s1[-1], s2[-1], color=color, s=50, marker='x', lw=1.5)

            # --- Subplot 3: Gradient Norm ---
            axes[2].plot(iters, grad_norm, color=color, alpha=0.4, lw=1.5)
            # End (x) - useful to see final precision relative to floor
            axes[2].scatter(iters[-1], grad_norm[-1], color=color, s=40, marker='x', zorder=5)

    # --- Formatting ---
    axes[0].set_title("Subspace Alignment Trajectory")
    axes[0].set_xlabel("Align 1 (Cosine)")
    axes[0].set_ylabel("Align 2 (Cosine)")
    axes[0].scatter(1, 1, color='gold', s=150, marker='*', label='Target', edgecolors='black', zorder=10)

    axes[1].set_title("Singular Value Trajectory")
    axes[1].set_xlabel("Singular Value 1")
    axes[1].set_ylabel("Singular Value 2")

    axes[2].set_title("Convergence Rate")
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Running Grad Norm")
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')


    axes[2].set_ylim(bottom=1e-10)

    for ax in axes:
        ax.grid(True, which="both", linestyle=':', alpha=0.6)

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(methods), frameon=True)

    plt.tight_layout()
    save_and_show('trajectory_comparison')

