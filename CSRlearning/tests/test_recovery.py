import argparse
import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from evaluation import compute_alignment
from generator import LinearCCADataGenerator
from learners import CSELearningModel


@dataclass
class AlgorithmTestConfig:
    algorithm_name: str
    learner_kwargs: dict


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

    data_gen = LinearCCADataGenerator(
        d1=20,
        d2=20,
        k=10,
        sigma=0.1,
        batch_size=512,
        device=device,
        dtype=dtype,
        deterministic=True,
        sign_flip=False,
    )

    encoder_u = torch.nn.Linear(20, cfg.learner_kwargs["latent_dim"], bias=False)
    encoder_v = torch.nn.Linear(20, cfg.learner_kwargs["latent_dim"], bias=False)

    model = CSELearningModel(
        encoder_u=encoder_u,
        encoder_v=encoder_v,
        algorithm_name=cfg.algorithm_name,
        **cfg.learner_kwargs,
    ).to(device)

    if dtype == torch.float64:
        model = model.double()

    model.reset_parameters(mode=cfg.learner_kwargs.get("init_mode", "near_zero"),
                          scale=cfg.learner_kwargs.get("init_scale", 1e-4))

    last_metrics = None
    for step in range(iterations):
        metrics = model.forward_backward(data_gen)
        last_metrics = metrics

        if (step % print_interval) == 0 or step == iterations - 1:
            spectrum, target_spectrum = model.compute_spectrum(data_gen)
            alignment = compute_alignment_summary(model, data_gen)
            s1, s2 = spectrum[0].item(), spectrum[1].item()
            ts1, ts2 = target_spectrum[0].item(), target_spectrum[1].item()
            align1, align2 = alignment[0].item(), alignment[1].item()
            print(
                f"[{cfg.algorithm_name}] step={step} "
                f"s=({s1:.4e},{s2:.4e}) "
                f"t=({ts1:.4e},{ts2:.4e}) "
                f"align=({align1:.4f},{align2:.4f}) "
                f"loss={metrics.get('loss_CSE', math.nan):.4e}"
            )

        # Sanity: no NaNs or infs in model params or losses
        params = [p.data for p in model.parameters()]
        if not has_finite_tensors(params):
            raise AssertionError(f"Non-finite parameter encountered for {cfg.algorithm_name}")
        if not all(math.isfinite(v) for v in metrics.values()):
            raise AssertionError(f"Non-finite metric encountered for {cfg.algorithm_name}")

    spectrum, target_spectrum = model.compute_spectrum(data_gen)
    alignment = compute_alignment_summary(model, data_gen)

    alignment_error = float(torch.mean(1.0 - alignment).item())
    spectrum_error = float(torch.mean(torch.abs(spectrum - target_spectrum)).item())

    return {
        "alignment_error": alignment_error,
        "spectrum_error": spectrum_error,
        "loss_CSE": float(last_metrics.get("loss_CSE", math.nan)) if last_metrics else math.nan,
    }


def default_algorithm_configs() -> List[AlgorithmTestConfig]:
    base_kwargs = {
        "latent_dim": 30,
        "lr_u": 1.0,
        "lr_v": 0.1,
        "eta": 1.0,
        "reg_c": 0.1,
        "n_opt_v": 1,
        "n_opt_pred": 1,
        "n_accum_pred": 1,
        "update_mode_cold_start": False,
        "full_solve_v": True,
        "params_q": {
            "update_rule": "exact_inverse",
            "momentum": False,
            "cold_start": True,
            "kwargs": None,
        },
        "init_mode": "near_zero",
        "init_scale": 1e-4,
    }

    return [
        AlgorithmTestConfig(
            algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingExactAlgorithm",
            learner_kwargs=dict(base_kwargs),
        ),
        AlgorithmTestConfig(
            algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingBilevelAlgorithm",
            learner_kwargs=dict(base_kwargs),
        ),
        AlgorithmTestConfig(
            algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingBilevelJointAlgorithm",
            learner_kwargs=dict(base_kwargs),
        ),
        AlgorithmTestConfig(
            algorithm_name="algorithm_steps.SemiGradientSSLAlgorithm",
            learner_kwargs=dict(base_kwargs),
        ),
        AlgorithmTestConfig(
            algorithm_name="algorithm_steps.UnconstrainedCCAAlgorithm",
            learner_kwargs=dict(base_kwargs),
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
        help="Run a single algorithm by full class path (e.g., algorithm_steps.CanonicalSubspaceEmbeddingExactAlgorithm)",
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
            raise ValueError(f"Unknown algorithm '{args.algorithm}'. उपलब्ध: {available}")

    for cfg in algorithm_configs:
        results[cfg.algorithm_name] = run_algorithm_test(
            cfg,
            iterations=args.iterations,
            print_interval=args.print_interval,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
        )

    print("Summary")
    for name, metrics in results.items():
        print(
            f"{name}: alignment_error={metrics['alignment_error']:.4e}, "
            f"spectrum_error={metrics['spectrum_error']:.4e}, "
            f"loss_CSE={metrics['loss_CSE']:.4e}"
        )


if __name__ == "__main__":
    main()
