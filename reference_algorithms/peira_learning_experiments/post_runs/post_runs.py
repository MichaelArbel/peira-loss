"""Generate the four PEIRA and Self-distillation heatmaps from completed toy runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlxp

if __package__:
    from .viz import plot_run_spectral_stack
else:
    from viz import plot_run_spectral_stack

BASE_DIR = Path(__file__).resolve().parents[3]
METHODS = {
    "peira": ("peira_learning.algorithms.ExactPEIRA", "PEIRA", 0.1, 0.4),
    "self_distillation": ("peira_learning.algorithms.ExactSelfDistillation", "Self-distillation", 10.0, 1e-5),
}
SCALES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=BASE_DIR / "data/outputs/deterministic_2")
    parser.add_argument("--figures-dir", type=Path, default=BASE_DIR / "figures")
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"Run directory does not exist: {args.runs_dir}")
    reader = mlxp.Reader(str(args.runs_dir), refresh=True)
    groups = []
    for method_id, (algorithm, label, lr, reg) in METHODS.items():
        query = (
            "(info.status=='COMPLETE') & (config.seed==0)"
            " & (config.task.total_steps==400000)"
            f" & (config.algorithm.name=='{algorithm}')"
            f" & (config.optimizer.kwargs.lr=={lr})"
            f" & (config.algorithm.kwargs.reg=={reg:.8f})"
            " & (config.task.init_scale in [0.000001, 0.00001, 0.0001, 0.001, 0.01])"
        )
        results = reader.filter(query_string=query)
        by_scale = results.groupby(["config.task.init_scale"])
        for scale in SCALES:
            group = by_scale[(scale,)] if (scale,) in by_scale.keys() else None
            if group is None or len(group) != 1:
                parser.error(f"Expected exactly one completed {label} run at scale {scale}; use a directory with one sweep.")
        groups.append((method_id, label, results))
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for method_id, label, results in groups:
        for metric in ("spectrum", "alignment"):
            plot_run_spectral_stack(results, label, str(args.figures_dir), data=metric, method_id=method_id)
            plt.close("all")


if __name__ == "__main__":
    main()
