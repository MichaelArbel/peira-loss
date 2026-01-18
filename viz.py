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
    
    plt.xlabel('$sigma_1$ (Leading Singular Value)')
    plt.ylabel('$sigma_2$ (Second Singular Value)')
    plt.title('Spectral Dynamics: Escape from Zero')
    plt.grid(True)
    plt.show()



# def plot_combined_analysis(df):
#     # 1. Setup the figure
#     fig, axes = plt.subplots(1, 3, figsize=(20, 6))
#     df_group = df.groupby(['config.learner.name'])
#     methods = [key[0] for key in df_group.keys()]
#     colors = ['#1f77b4', '#ff7f0e'] 
#     method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
#     legend_handles = []

#     for i, method in enumerate(methods):
#         method_df = df_group[(method,)]
#         color = method_color_map[method]
        
#         # Legend proxy
#         line_handle = plt.Line2D([0], [0], color=color, lw=2, label=f"Method: {method}")
#         legend_handles.append(line_handle)

#         for row in method_df:
#             # Data extraction
#             a1, a2 = np.array(row['metrics.align_1']), np.array(row['metrics.align_2'])
#             s1, s2 = np.array(row['metrics.singval_1']), np.array(row['metrics.singval_2'])
#             iters = np.array(row['metrics.iter'])
#             grad_norm = np.array(row['metrics.running_grad_norm'])

#             # --- Subplot 1: Alignment ---
#             axes[0].plot(a1, a2, color=color, alpha=0.3, lw=1)
#             # Start (o) and End (x)
#             axes[0].scatter(a1[0], a2[0], color=color, s=20, marker='o', alpha=0.5) 
#             axes[0].scatter(a1[-1], a2[-1], color=color, s=50, marker='x', lw=1.5)

#             # --- Subplot 2: Singular Values ---
#             axes[1].plot(s1, s2, color=color, alpha=0.3, lw=1)
#             # Start (o) and End (x)
#             axes[1].scatter(s1[0], s2[0], color=color, s=20, marker='o', alpha=0.5)
#             axes[1].scatter(s1[-1], s2[-1], color=color, s=50, marker='x', lw=1.5)

#             # --- Subplot 3: Gradient Norm ---
#             axes[2].plot(iters, grad_norm, color=color, alpha=0.4, lw=1.5)
#             # End (x) - useful to see final precision relative to floor
#             axes[2].scatter(iters[-1], grad_norm[-1], color=color, s=40, marker='x', zorder=5)

#     # --- Formatting ---
#     axes[0].set_title("Subspace Alignment Trajectory")
#     axes[0].set_xlabel("Align 1 (Cosine)")
#     axes[0].set_ylabel("Align 2 (Cosine)")
#     axes[0].scatter(1, 1, color='gold', s=150, marker='*', label='Target', edgecolors='black', zorder=10)

#     axes[1].set_title("Singular Value Trajectory")
#     axes[1].set_xlabel("Singular Value 1")
#     axes[1].set_ylabel("Singular Value 2")

#     axes[2].set_title("Convergence Rate")
#     axes[2].set_xlabel("Iterations")
#     axes[2].set_ylabel("Running Grad Norm")
#     axes[2].set_yscale('log')
#     axes[2].set_xscale('log')


#     axes[2].set_ylim(bottom=1e-10)

#     for ax in axes:
#         ax.grid(True, which="both", linestyle=':', alpha=0.6)

#     fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
#                ncol=len(methods), frameon=True)

#     plt.tight_layout()
#     save_and_show('trajectory_comparison')


def save_and_show(filename='latest_plot', figures_dir="figures"):
    # 1. Save to disk
    name=  os.path.join(figures_dir, filename)
    plt.savefig(name+'.png',dpi=300, bbox_inches='tight')
    plt.savefig(name+'.pdf', bbox_inches='tight')
    
    # 2. Display in terminal
    imgcat(open(name+'.png', 'rb').read())


def plot_combined_analysis(df, figures_dir="figures"):
    # 1. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Use standard grouping
    df_group = df.groupby(['config.learner.name'])
    methods = [key[0] for key in df_group.keys()]
    colors = ['#1f77b4', '#ff7f0e'] 
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    legend_handles = []

    for method in methods:
        method_df = df_group[(method,)]
        color = method_color_map[method]
        
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

            # Plot individual trajectories with high transparency
            line, = axes[0].plot(a1, a2, color=color, lw=1.)
            axes[1].plot(s1, s2, color=color, lw=1.)
            axes[2].plot(iters, grad_norm, color=color, lw=1)

            # Store for averaging (assuming same lengths, or interpolate if needed)

            # Start and End Markers for individual runs
            axes[0].scatter(a1[0], a2[0], color=color, s=15, marker='o')
            #axes[0].scatter(a1[-1], a2[-1], color=color, s=40, marker='x', alpha=0.6)
            axes[1].scatter(s1[0], s2[0], color=color, s=15, marker='o')
            #axes[1].scatter(s1[-1], s2[-1], color=color, s=40, marker='x', alpha=0.6)


        target_spectrum = row['artifact.pickle.']['target_spectrum'].load()
        axes[1].scatter(target_spectrum[0], target_spectrum[1], color=color, s=300, marker='*', label='Target', edgecolors='black', zorder=10)


        legend_handles.append(line)
        print(lol)

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




    # 2. Singular Value Trajectory
    axes[1].set_title("Singular Value Trajectory", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Singular Value 1")
    axes[1].set_ylabel("Singular Value 2")
    

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
    save_and_show('trajectory_comparison.pdf', figures_dir= figures_dir)










def plot_combined_analysis(df, figures_dir="figures"):
    # 1. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Use standard grouping
    df_group = df.groupby(['config.learner.name'])
    methods = [key[0] for key in df_group.keys()]
    colors = ['#1f77b4', '#ff7f0e'] 
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    legend_handles = []

    for method in methods:
        method_df = df_group[(method,)]
        color = method_color_map[method]
        
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

            # Plot individual trajectories with high transparency
            line, = axes[0].plot(a1, a2, color=color, lw=1.)
            axes[1].plot(s1, s2, color=color, lw=1.)
            axes[2].plot(iters, grad_norm, color=color, lw=1)

            # Store for averaging (assuming same lengths, or interpolate if needed)

            # Start and End Markers for individual runs
            axes[0].scatter(a1[0], a2[0], color=color, s=15, marker='o')
            #axes[0].scatter(a1[-1], a2[-1], color=color, s=40, marker='x', alpha=0.6)
            axes[1].scatter(s1[0], s2[0], color=color, s=15, marker='o')
            #axes[1].scatter(s1[-1], s2[-1], color=color, s=40, marker='x', alpha=0.6)


        target_spectrum = row['artifact.pickle.']['target_spectrum'].load()
        axes[1].scatter(target_spectrum[0], target_spectrum[1], color=color, s=300, marker='*', label='Target', edgecolors='black', zorder=10)


        legend_handles.append(line)
        print(lol)

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




    # 2. Singular Value Trajectory
    axes[1].set_title("Singular Value Trajectory", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Singular Value 1")
    axes[1].set_ylabel("Singular Value 2")
    

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



