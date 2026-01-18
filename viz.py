import matplotlib.pyplot as plt

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