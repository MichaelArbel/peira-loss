import argparse
import math
from dataclasses import dataclass
from typing import Dict, List
from types import SimpleNamespace

import torch

from evaluation import compute_alignment
from generator import LinearCCADataGenerator
from factory import instantiate


@dataclass
class AlgorithmTestConfig:
    algorithm_name: str
    state_kwargs: dict
    regressor_spec: dict


def set_all_seeds(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_finite_tensors(tensors: List[torch.Tensor]) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return False
    return True


def compute_alignment_summary(model, data_gen):
    W = model.encoder_u.weight.t() @ model.encoder_u.weight
    alignment = compute_alignment(W, data_gen.info["P"])
    return alignment


def run_algorithm_test(
    cfg: AlgorithmTestConfig,
    iterations: int,
    print_interval: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, float]:
    set_all_seeds(seed)
    d1=20
    d2=20
    data_gen = LinearCCADataGenerator(
        d1=d1,
        d2=d2,
        k=10,
        sigma=0.1,
        batch_size=512,
        device=device,
        dtype=dtype,
        deterministic=True,
        sign_flip=False,
    )

    latent_dim = cfg.state_kwargs["latent_dim"]
    encoder_u = torch.nn.Linear(20, latent_dim, bias=False)
    if cfg.state_kwargs.get("tied_encoders", False):
        encoder_v = encoder_u
    else:
        encoder_v = torch.nn.Linear(20, latent_dim, bias=False)

    lr_u = cfg.state_kwargs.get("lr_u", 0.001)
    lr_v = cfg.state_kwargs.get("lr_v", lr_u)
    optimizer_ctor = lambda params, lr: torch.optim.SGD(params, lr=lr)
    optimizers = [optimizer_ctor(encoder_u.parameters(), lr_u)]
    if encoder_v is not encoder_u:
        optimizers.append(optimizer_ctor(encoder_v.parameters(), lr_v))
    t_max = cfg.state_kwargs.get("t_max", iterations)
    scheduler_ctor = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=0.000001
    )
    schedulers = [scheduler_ctor(opt) for opt in optimizers]

    model = SimpleNamespace(
        encoder_u=encoder_u.to(device),
        encoder_v=encoder_v.to(device),
        optimizers=optimizers,
        schedulers=schedulers,
        k=latent_dim,
        eta=cfg.state_kwargs.get("eta", 0.1),
        reg_c=cfg.state_kwargs.get("reg_c", 1.0),
        n_opt_pred=cfg.state_kwargs.get("n_opt_pred", 1),
        t_max=t_max,
        counter=0,
        first_update=True,
        Q=torch.eye(latent_dim, device=device, dtype=dtype),
        S=torch.zeros(latent_dim, latent_dim, device=device, dtype=dtype),
        F=torch.zeros(latent_dim, latent_dim, device=device, dtype=dtype),
    )

    algorithm_class = instantiate(cfg.algorithm_name)
    algorithm = algorithm_class()
    algorithm.configure_model(model, {"regressor": cfg.regressor_spec})

    if dtype == torch.float64:
        model.encoder_u = model.encoder_u.double()
        if model.encoder_v is not model.encoder_u:
            model.encoder_v = model.encoder_v.double()

    with torch.no_grad():
        for m in [model.encoder_u, model.encoder_v]:
            if isinstance(m, torch.nn.Linear):
                if cfg.state_kwargs.get("init_mode", "near_zero") == "near_zero":
                    torch.nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=cfg.state_kwargs.get("init_scale", 1e-4),
                    )

    last_metrics = None
    for step in range(iterations):
        metrics = algorithm.step(model, data_gen)
        last_metrics = metrics

        if (step % print_interval) == 0 or step == iterations - 1:
            spectrum, target_spectrum = algorithm.compute_spectrum(model, data_gen)
            alignment = compute_alignment_summary(model, data_gen)
            s1, s2 = spectrum[0].item(), spectrum[1].item()
            ts1, ts2 = target_spectrum[0].item(), target_spectrum[1].item()
            align1, align2 = alignment[0].item(), alignment[1].item()
            print(
                f"[{cfg.algorithm_name}] step={step} "
                f"s=({abs(s1-ts1):.4e},{abs(s2-ts2):.4e}) "
                f"align=({align1:.4f},{align2:.4f}) "
                f"loss={metrics.get('loss', math.nan):.4e}"
            )

        params = [p.data for p in model.encoder_u.parameters()]
        if model.encoder_v is not model.encoder_u:
            params += [p.data for p in model.encoder_v.parameters()]
        if not has_finite_tensors(params):
            raise AssertionError(f"Non-finite parameter encountered for {cfg.algorithm_name}")
        if not all(math.isfinite(v) for v in metrics.values()):
            raise AssertionError(f"Non-finite metric encountered for {cfg.algorithm_name}")

    spectrum, target_spectrum = algorithm.compute_spectrum(model, data_gen)
    alignment = compute_alignment_summary(model, data_gen)

    alignment_error = float(torch.mean(1.0 - alignment).item())
    spectrum_error = float(torch.mean(torch.abs(spectrum - target_spectrum)).item())

    return {
        "alignment_error": alignment_error,
        "spectrum_error": spectrum_error,
        "loss": float(last_metrics.get("loss", math.nan)) if last_metrics else math.nan,
    }


def default_algorithm_configs() -> List[AlgorithmTestConfig]:
    base_state = {
        "latent_dim": 30,
        "lr_u": 0.1,
        "lr_v": 0.1,
        "eta": 1.0,
        "reg_c": 0.1,
        "n_opt_pred": 1,
        "t_max": 20000,
        "tied_encoders": False,
        "init_mode": "near_zero",
        "init_scale": 1e-4,
    }

    regressor_spec = {
        "name": "core.linear_regressor.IterativeDiffRegressor",
        "kwargs": {"symmetric": True, "with_adjoint": True},
        "optimizer": {"name": "torch.optim.SGD", "kwargs": {"lr": 1.0}},
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {"T_max": 20000, "eta_min": 0.000001},
        },
    }

    return [
        AlgorithmTestConfig(
            algorithm_name="core.algorithms.CanonicalSubspaceEmbeddingExactAlgorithm",
            state_kwargs=dict(base_state),
            regressor_spec=regressor_spec,
        ),
        AlgorithmTestConfig(
            algorithm_name="core.algorithms.CanonicalSubspaceEmbeddingBilevelAlgorithm",
            state_kwargs=dict(base_state),
            regressor_spec=regressor_spec,
        ),
        AlgorithmTestConfig(
            algorithm_name="core.algorithms.CanonicalSubspaceEmbeddingBilevelJointAlgorithm",
            state_kwargs=dict(base_state),
            regressor_spec=regressor_spec,
        ),
        AlgorithmTestConfig(
            algorithm_name="core.algorithms.SemiGradientSSLAlgorithm",
            state_kwargs=dict(base_state),
            regressor_spec={
                "name": "core.linear_regressor.MomentumRegressor",
                "kwargs": {"symmetric": False, "with_adjoint": False},
                "scheduler": {
                    "name": "utils.CosineScheduler",
                    "kwargs": {"T_max": 20000, "m_start": 1.0, "m_end": 0.001},
                },
            },
        ),
        AlgorithmTestConfig(
            algorithm_name="core.algorithms.UnconstrainedCCAAlgorithm",
            state_kwargs=dict(base_state),
            regressor_spec=regressor_spec,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-truth recovery tests")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Run a single algorithm by full class path (e.g., core.algorithms.CanonicalSubspaceEmbeddingExactAlgorithm)",
    )
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    results = {}
    algorithm_configs = default_algorithm_configs()
    if args.algorithm is not None:
        algorithm_configs = [cfg for cfg in algorithm_configs if cfg.algorithm_name == args.algorithm]
        if not algorithm_configs:
            available = [cfg.algorithm_name for cfg in default_algorithm_configs()]
            raise ValueError(f"Unknown algorithm '{args.algorithm}'. Available: {available}")

    for cfg in algorithm_configs:
        results[cfg.algorithm_name] = run_algorithm_test(
            cfg,
            iterations=args.iterations,
            print_interval=args.print_interval,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
        )

    print("\nSummary")
    for name, metrics in results.items():
        print(
            f"{name}: alignment_error={metrics['alignment_error']:.4e}, "
            f"spectrum_error={metrics['spectrum_error']:.4e}, "
            f"loss={metrics['loss']:.4e}"
        )


if __name__ == "__main__":
    main()
