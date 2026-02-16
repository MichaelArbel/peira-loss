import os

import mlxp

from viz import filter_runs, plot_method_standalone_figures


# Path to the directory containing experiment runs.
PATH = os.environ.get("CSR_RESULTS_PATH", "data/outputs/deterministic_9")

# Where figures are saved.
FIGURES_DIR = os.environ.get("CSR_FIGURES_DIR", "../figures")

# Keep only this initialization scale.
INIT_SCALE = float(os.environ.get("CSR_INIT_SCALE", "1e-4"))

# For ssl_exact only, keep this optimizer learning rate.
SSL_EXACT_LR = float(os.environ.get("CSR_SSL_EXACT_LR", "10"))


def main():
    # Start with complete runs from mlxp metadata.
    query = "info.status=='COMPLETE'"
    results = mlxp.Reader(PATH).filter(query_string=query)

    # Enforce requested filtering:
    # - single init scale
    # - ssl_exact only with lr=10
    # - ssl_exact only successful/not failed (when success field is present)
    filtered_rows = filter_runs(
        results,
        init_scale=INIT_SCALE,
        ssl_exact_lr=SSL_EXACT_LR,
        require_ssl_exact_success=True,
    )

    if len(filtered_rows) == 0:
        raise RuntimeError(
            f"No runs matched filters: init_scale={INIT_SCALE}, ssl_exact_lr={SSL_EXACT_LR}"
        )

    plot_method_standalone_figures(filtered_rows, figures_dir=FIGURES_DIR)


if __name__ == "__main__":
    main()
