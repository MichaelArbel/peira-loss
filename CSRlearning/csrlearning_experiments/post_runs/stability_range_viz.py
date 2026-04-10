import os

import matplotlib.pyplot as plt
import mlxp
import numpy as np
from matplotlib.lines import Line2D

from viz import save_and_show


def _safe_get(row, key, default=None):
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _is_close(a, b, rtol=1e-12, atol=1e-12):
    try:
        return np.isclose(float(a), float(b), rtol=rtol, atol=atol)
    except (TypeError, ValueError):
        return False


def _row_method(row):
    method = _safe_get(row, "config.algorithm.name")
    if method is None:
        method = _safe_get(row, "config.algorithm")
    if method is None:
        method = _safe_get(row, "config.learner.name")
    if method is None:
        method = _safe_get(row, "config.learner.kwargs.algorithm_name")
    if method is None:
        method = "unknown"
    return str(method)


def _method_label(method):
    mapping =  {
    "csrlearning.algorithms.CSEExactRegressionAlgorithm": "CSE",
    "csrlearning.algorithms.SSLExactSemiGrad": "SG-SSL",
    "csrlearning.algorithms.UnconstrainedCCAAlgorithm": "EY"
    }
    return mapping.get(method, str(method))




def _is_successful_row(row):
    success_keys = [
        "info.succeeded",
        "info.mlxp.succeeded",
        "mlxp.succeeded",
    ]
    status_keys = [
        "info.status",
        "info.mlxp.status",
        "mlxp.status",
    ]
    for key in success_keys:
        val = _safe_get(row, key)
        if val is not None:
            return bool(val)
    for key in status_keys:
        val = _safe_get(row, key)
        if val is not None:
            return str(val).upper() in {"COMPLETE", "SUCCEEDED", "SUCCESS"}
    return True


def _extract_min_grad_norm(row):
    grad_norm = _safe_get(row, "metrics.grad_norm", None)
    if grad_norm is None or len(grad_norm) == 0:
        return None
    grad_norm = np.asarray(grad_norm, dtype=float)
    grad_norm = grad_norm[np.isfinite(grad_norm)]
    if grad_norm.size == 0:
        return None
    return float(np.min(grad_norm))


def _extract_lr(row):
    lr = _safe_get(row, "config.optimizer.kwargs.lr")
    if lr is None:
        lr = _safe_get(row, "config.optimizer.lr")
    if lr is None:
        return None
    try:
        return float(lr)
    except (TypeError, ValueError):
        return None


def _extract_init_scale(row):
    scale = _safe_get(row, "config.task.init_scale")
    if scale is None:
        return None
    try:
        return float(scale)
    except (TypeError, ValueError):
        return None


def _as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def _median_min_grad_of_group(group_rows):
    vals = []
    for row in group_rows:
        val = _extract_min_grad_norm(row)
        if val is not None:
            vals.append(val)
    if len(vals) == 0:
        return None
    return float(np.median(np.asarray(vals, dtype=float)))


def plot_stability_range_min_grad_norm(
    results,
    figures_dir="../figures",
    filename="stability_range_min_grad_norm",
):
    """
    Plot x=learning rate, y=min running gradient norm over optimization history.
    - Color encodes method.
    - Line style encodes initialization scale.
    """
    grouped = results.groupby(
        ["config.algorithm.name", "config.task.init_scale", "config.optimizer.kwargs.lr"]
    )
    aggregated = {}
    methods_present = set()
    init_scales = list(set(results[:]["config.task.init_scale"]))

    for key in grouped.keys():
        method, scale, lr = _as_tuple(key)
        min_grad = _median_min_grad_of_group(grouped[key])
        if min_grad is None:
            continue
        aggregated[(method, scale, lr)] = min_grad
        methods_present.add(method)

    if len(aggregated) == 0:
        raise RuntimeError("No grouped data for requested methods/init_scales.")

    methods_present = sorted(methods_present)
    method_to_color = {
        method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(methods_present)
    }
    scale_to_style = {float(init_scales[0]): "-", float(init_scales[1]): "--"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods_present:
        for scale in init_scales:
            points = [
                (lr, y)
                for (m, s, lr), y in aggregated.items()
                if (m == method) and _is_close(s, float(scale))
            ]
            if len(points) == 0:
                continue
            points = sorted(points, key=lambda p: p[0])
            x = np.array([p[0] for p in points], dtype=float)
            y = np.array([p[1] for p in points], dtype=float)

            style = scale_to_style.get(float(scale), "-")
            color = method_to_color[method]
            ax.plot(
                x,
                y,
                linestyle=style,
                marker="o",
                color=color,
                linewidth=2,
                markersize=4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Minimum running gradient norm (history)")
    ax.set_title("Stability Range: Min Gradient Norm vs Learning Rate")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    method_handles = [
        Line2D([0], [0], color=method_to_color[m], lw=2, linestyle="-", label=_method_label(m))
        for m in methods_present
    ]
    scale_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=scale_to_style[float(s)], label=rf"$\rho_0={s:g}$")
        for s in init_scales
    ]

    legend_methods = ax.legend(
        handles=method_handles,
        title="Method",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(legend_methods)
    ax.legend(
        handles=scale_handles,
        title="Init scale",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.74),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    save_and_show(filename=filename, figures_dir=figures_dir)


def main():
    path = "../../../data/outputs/stability_range_2/"
    figures_dir = "../../../figures"


    rows = mlxp.Reader(path,refresh=False).filter()

    plot_stability_range_min_grad_norm(
        rows,
        figures_dir=figures_dir,
        filename="stability_range_min_grad_norm",
    )


if __name__ == "__main__":
    main()
