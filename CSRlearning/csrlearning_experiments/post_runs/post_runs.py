import os

import mlxp

from viz import (
        plot_run_spectral_stack, 
        plot_run_grad_norm,
        plot_phase_portrait,
        )

# Path to the directory containing experiment runs.
PATH = "../../../data/outputs/deterministic_2/"

# Where figures are saved.
FIGURES_DIR = "../../../figures"

# Keep only this initialization scale.
INIT_SCALE = "0.000001"

# For ssl_exact only, keep this optimizer learning rate.
SSL_EXACT_LR = "10"


method_label = {
    "csrlearning.algorithms.CSEExactRegressionAlgorithm": "CSE",
    "csrlearning.algorithms.SSLExactSemiGrad": "SG-SSL",
    "csrlearning.algorithms.UnconstrainedCCAAlgorithm": "EY",
}


color_dict = {
    "csrlearning.algorithms.CSEExactRegressionAlgorithm": '#ff0000',
    "csrlearning.algorithms.SSLExactSemiGrad": '#1f77b4',
    "csrlearning.algorithms.UnconstrainedCCAAlgorithm": '#ff7f0e',
   
}
methods_present = set(["csrlearning.algorithms.CSEExactRegressionAlgorithm",
    "csrlearning.algorithms.SSLExactSemiGrad",
    "csrlearning.algorithms.UnconstrainedCCAAlgorithm"])
color_dict = {
        method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(methods_present)
    }



def plot_method_standalone_figures(df, figures_dir):
    """
    For each method, generate standalone figures for:
    - spectrum evolution
    - alignment evolution
    - gradient norm vs iterations
    Uses one representative run per method (first row in that group).
    """
    method_key = "config.algorithm.name"
    df_group = df.groupby([method_key])
    methods = list(np.unique([key[0] for key in df_group.keys()]))

    colors = ['#1f77b4', '#ff7f0e'] 
    heatmaps = [plt.cm.Blues, plt.cm.Oranges]
    #method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    cmaps = {method: heatmaps[i % len(heatmaps)] for i, method in enumerate(methods)}



    

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


def main():
    # Start with complete runs from mlxp metadata.
    query = "info.status=='COMPLETE'"
    query_ssl  = "config.algorithm.name=='csrlearning.algorithms.SSLExactSemiGrad' & config.optimizer.kwargs.lr==10"
    query_non_ssl = "config.algorithm.name!='csrlearning.algorithms.SSLExactSemiGrad'"
    total_query = "("+query + ") & ( (" +  query_ssl + ") | (" + query_non_ssl+") )"
    results = mlxp.Reader(PATH, refresh=False).filter(query_string=query)

    plot_method_standalone_figures(results, figures_dir=FIGURES_DIR)



def main_phase_portrait():
    query = "info.status=='COMPLETE' & config.task.init_scale in [0.000001, 0.00001, 0.01]"
    query_ssl  = "config.algorithm.name=='csrlearning.algorithms.SSLExactSemiGrad' & config.optimizer.kwargs.lr==10."
    query_non_ssl = "config.algorithm.name!='csrlearning.algorithms.SSLExactSemiGrad'"
    total_query = "("+query + ") & ( (" +  query_ssl + ") | (" + query_non_ssl+") )"
    results = mlxp.Reader(PATH, refresh=False).filter(query_string=total_query)


    plot_phase_portrait(results, method_label, color_dict, figures_dir=FIGURES_DIR)



if __name__ == "__main__":
    main_phase_portrait()
