import os
from collections import defaultdict

import matplotlib.pyplot as plt
import mlxp
import numpy as np

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
    canon = _canonical_method(method)
    mapping = {"ssl_exact": "SSLExact", "cse_exact": "CSEExact", "ey_cca": "EyCCA"}
    return mapping.get(canon, str(method))


def _canonical_method(method):
    m = str(method).lower()
    if ("ssl_exact" in m) or ("sslexactsemigrad" in m) or ("semigradientssl" in m):
        return "ssl_exact"
    if ("cse_exact" in m) or ("canonicalsubspaceembeddingexact" in m):
        return "cse_exact"
    if ("ey_cca" in m) or ("unconstrainedcca" in m):
        return "ey_cca"
    return str(method)


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
    grad_norm = _safe_get(row, "metrics.running_grad_norm", None)
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


def _filter_rows(rows, init_scales=(0.01, 10.0), methods=("ssl_exact", "cse_exact", "ey_cca")):
    allowed_methods = set(methods) if methods is not None else None
    filtered = []
    for row in rows:
        method = _canonical_method(_row_method(row))
        scale = _extract_init_scale(row)
        lr = _extract_lr(row)
        min_grad = _extract_min_grad_norm(row)
        if scale is None or lr is None or min_grad is None:
            continue
        if not any(_is_close(scale, s) for s in init_scales):
            continue
        if (allowed_methods is not None) and (method not in allowed_methods):
            continue
        filtered.append(row)
    return filtered


def plot_stability_range_min_grad_norm(
    rows,
    figures_dir="../figures",
    init_scales=(0.01, 10.0),
    methods=("ssl_exact", "cse_exact", "ey_cca"),
    filename="stability_range_min_grad_norm",
):
    """
    Plot x=learning rate, y=min running gradient norm over optimization history.
    - Color encodes method.
    - Line style encodes initialization scale.
    """
    filtered_rows = _filter_rows(rows, init_scales=init_scales, methods=methods)
    if len(filtered_rows) == 0:
        raise RuntimeError("No rows found for requested methods/init_scales with valid metrics.")

    grouped = defaultdict(list)
    for row in filtered_rows:
        method = _canonical_method(_row_method(row))
        scale = _extract_init_scale(row)
        lr = _extract_lr(row)
        min_grad = _extract_min_grad_norm(row)
        grouped[(method, scale)].append((lr, min_grad))

    # If duplicates exist for a (method, scale, lr), use median.
    compressed = defaultdict(lambda: defaultdict(list))
    for (method, scale), points in grouped.items():
        for lr, min_grad in points:
            compressed[(method, scale)][lr].append(min_grad)

    methods_present = sorted({k[0] for k in compressed.keys()})
    method_to_color = {
        method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(methods_present)
    }
    scale_to_style = {float(init_scales[0]): "-", float(init_scales[1]): "--"}

    plt.figure(figsize=(8, 5))
    for method in methods_present:
        for scale in init_scales:
            key = (method, float(scale))
            if key not in compressed:
                continue
            lr_to_vals = compressed[key]
            x = np.array(sorted(lr_to_vals.keys()), dtype=float)
            y = np.array([np.median(lr_to_vals[lr]) for lr in x], dtype=float)

            style = scale_to_style.get(float(scale), "-")
            color = method_to_color[method]
            label = f"{_method_label(method)} | init={scale:g}"
            plt.plot(
                x,
                y,
                linestyle=style,
                marker="o",
                color=color,
                linewidth=2,
                markersize=4,
                label=label,
            )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Minimum running gradient norm (history)")
    plt.title("Stability Range: Min Gradient Norm vs Learning Rate")
    plt.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_and_show(filename=filename, figures_dir=figures_dir)


def main():
    path = os.environ.get("CSR_RESULTS_PATH", "data/outputs/deterministic")
    figures_dir = os.environ.get("CSR_FIGURES_DIR", "../figures")

    rows = mlxp.Reader(path).filter()

    plot_stability_range_min_grad_norm(
        rows,
        figures_dir=figures_dir,
        init_scales=(0.01, 10.0),
        methods=("ssl_exact", "cse_exact", "ey_cca"),
        filename="stability_range_min_grad_norm",
    )


if __name__ == "__main__":
    main()
