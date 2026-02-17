import os

import mlxp

from viz import plot_method_standalone_figures


# Path to the directory containing experiment runs.
PATH = "../../../data/outputs/deterministic_2/"

# Where figures are saved.
FIGURES_DIR = "../../../figures"

# Keep only this initialization scale.
INIT_SCALE = "0.000001"

# For ssl_exact only, keep this optimizer learning rate.
SSL_EXACT_LR = "10"


def main():
    # Start with complete runs from mlxp metadata.
    query = "info.status=='COMPLETE'"
    query_ssl  = "config.algorithm.name=='csrlearning.algorithms.SSLExactSemiGrad' & config.optimizer.kwargs.lr==10"
    query_non_ssl = "config.algorithm.name!='csrlearning.algorithms.SSLExactSemiGrad'"
    total_query = "("+query + ") & ( (" +  query_ssl + ") | (" + query_non_ssl+") )"
    results = mlxp.Reader(PATH, refresh=False).filter(query_string=query)



    # Enforce requested filtering:
    # - single init scale
    # - ssl_exact only with lr=10
    # - ssl_exact only successful/not failed (when success field is present)
    
    plot_method_standalone_figures(results, figures_dir=FIGURES_DIR)


if __name__ == "__main__":
    main()
