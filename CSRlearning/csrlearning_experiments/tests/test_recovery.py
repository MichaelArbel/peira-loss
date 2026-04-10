import argparse
import math
from dataclasses import dataclass
from typing import Dict, List
from types import SimpleNamespace

import torch

from csrlearning_experiments.evaluation.evaluation import compute_alignment, compute_algorithm_spectrum
from csrlearning_experiments.datasets.gaussian import LinearCCADataGenerator
from csrlearning_experiments.factory import instantiate
from csrlearning_experiments.training.pipeline import compute_covariance, encode
from csrlearning_experiments.training.optimizer_bundle import OptimizerBundle


@dataclass
class AlgorithmTestConfig:
    algorithm_name: str
    algo_kwargs: dict
    tied_encoders: bool = False


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


def compute_alignment_summary(encoders, data_gen, tied_encoders: bool):
    if tied_encoders:
        W = encoders[0].weight
        B = data_gen.info.cov_1
        U = data_gen.info.singular_vectors_1
    else:
        W = torch.cat([encoders[0].weight, encoders[1].weight], dim=0)
        B = torch.block_diag(data_gen.info.cov_1, data_gen.info.cov_2)
        U = torch.cat([data_gen.info.singular_vectors_1, data_gen.info.singular_vectors_2], dim=0)

    eigvals, eigvecs = torch.linalg.eigh(B)
    sqrt_B = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T
    M = sqrt_B @ W.t() @ W @ sqrt_B
    return compute_alignment(M, U)


def compute_cov_spectrum(encoders, data_gen, tied_encoders: bool):
    inter_cov, cov_lists = compute_covariance(
        lambda data: encode(data, encoders, False),
        data_gen,
    )
    intra_cov = cov_lists[0]
    if not tied_encoders:
        intra_cov = intra_cov + cov_lists[1]
    return torch.linalg.svdvals(intra_cov)

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

    latent_dim = cfg.algo_kwargs["latent_dim"]
    encoder_u = torch.nn.Linear(20, latent_dim, bias=False)
    if cfg.tied_encoders:
        encoder_v = encoder_u
    else:
        encoder_v = torch.nn.Linear(20, latent_dim, bias=False)

    optimizer_ctor = lambda params: torch.optim.SGD(params, lr=cfg.algo_kwargs.get("lr", 0.001))
    optimizer = optimizer_ctor(encoder_u.parameters())
    optimizers = [optimizer]
    if encoder_v is not encoder_u:
        optimizers.append(optimizer_ctor(encoder_v.parameters()))

    t_max = cfg.algo_kwargs.get("T_max", iterations)
    scheduler_ctor = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t_max, eta_min=0.000001
    )
    schedulers = [scheduler_ctor(opt) for opt in optimizers]

    bundle = OptimizerBundle()
    for opt, sch in zip(optimizers, schedulers):
        bundle.add(opt, sch, managed_by_algorithm=False)

    algorithm_class = instantiate(cfg.algorithm_name)
    algorithm = algorithm_class(**cfg.algo_kwargs)
    ext_opts, ext_scheds = algorithm.configure([encoder_u, encoder_v])
    for opt, sch in zip(ext_opts, ext_scheds):
        bundle.add(opt, sch, managed_by_algorithm=False)

    if dtype == torch.float64:
        encoder_u = encoder_u.double()
        if encoder_v is not encoder_u:
            encoder_v = encoder_v.double()

    with torch.no_grad():
        for m in [encoder_u, encoder_v]:
            if isinstance(m, torch.nn.Linear):
                if cfg.algo_kwargs.get("init_mode", "near_zero") == "near_zero":
                    torch.nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=cfg.algo_kwargs.get("init_scale", 1e-4),
                    )

    last_metrics = None
    for step in range(iterations):
        state = algorithm.get_internal_state()
        for _ in range(algorithm.inner_steps()):
            update_batch = next(data_gen)
            with torch.no_grad():
                update_encoded = encode(update_batch, [encoder_u, encoder_v], algorithm.stop_grad)
            state = algorithm.update_internal_state(update_encoded.x1, update_encoded.x2)

        bundle.zero_grad()
        data = next(data_gen)
        output = encode(data, [encoder_u, encoder_v], state.stop_grad)
        loss = algorithm(output.x1, output.x2, state)
        loss.backward()
        bundle.step()

        metrics = {"loss": float(loss.item())}
        last_metrics = metrics

        if (step % print_interval) == 0 or step == iterations - 1:
            cov_spectrum = compute_cov_spectrum([encoder_u, encoder_v], data_gen, cfg.tied_encoders)
            spectrum, target_spectrum = compute_algorithm_spectrum(
                algorithm, cov_spectrum, data_gen.info
            )
            alignment = compute_alignment_summary([encoder_u, encoder_v], data_gen, cfg.tied_encoders)
            s1, s2 = spectrum[0].item(), spectrum[1].item()
            ts1, ts2 = target_spectrum[0].item(), target_spectrum[1].item()
            align1, align2 = alignment[0].item(), alignment[1].item()
            print(
                f"[{cfg.algorithm_name}] step={step} "
                f"s=({abs(s1-ts1):.4e},{abs(s2-ts2):.4e}) "
                f"align=({align1:.4f},{align2:.4f}) "
                f"loss={metrics.get('loss', math.nan):.4e}"
            )

        params = [p.data for p in encoder_u.parameters()]
        if encoder_v is not encoder_u:
            params += [p.data for p in encoder_v.parameters()]
        if not has_finite_tensors(params):
            raise AssertionError(f"Non-finite parameter encountered for {cfg.algorithm_name}")
        if not all(math.isfinite(v) for v in metrics.values()):
            raise AssertionError(f"Non-finite metric encountered for {cfg.algorithm_name}")

    cov_spectrum = compute_cov_spectrum([encoder_u, encoder_v], data_gen, cfg.tied_encoders)
    spectrum, target_spectrum = compute_algorithm_spectrum(
        algorithm, cov_spectrum, data_gen.info
    )
    alignment = compute_alignment_summary([encoder_u, encoder_v], data_gen, cfg.tied_encoders)

    alignment_error = float(torch.mean(1.0 - alignment).item())
    spectrum_error = float(torch.mean(torch.abs(spectrum - target_spectrum)).item())

    return {
        "alignment_error": alignment_error,
        "spectrum_error": spectrum_error,
        "loss": float(last_metrics.get("loss", math.nan)) if last_metrics else math.nan,
    }


def default_algorithm_configs() -> List[AlgorithmTestConfig]:
    base_kwargs = {
        "latent_dim": 30,
        "reg": 0.1,
        "lr": 0.1,
        "T_max": 20000,
        "init_mode": "near_zero",
        "init_scale": 1e-4,
    }

    return [
        AlgorithmTestConfig(
            algorithm_name="csrlearning.algorithms.CSEExactRegressionAlgorithm",
            algo_kwargs={"reg": 0.1, "latent_dim": 30},
            tied_encoders=False,
        ),
        AlgorithmTestConfig(
            algorithm_name="csrlearning.algorithms.CSEBilevelStoGrad",
            algo_kwargs={**dict(base_kwargs), "n_iter": 1, "symmetric": True, "with_adjoint": True},
            tied_encoders=False,
        ),
        AlgorithmTestConfig(
            algorithm_name="csrlearning.algorithms.CSEBilevelStoGradJoint",
            algo_kwargs={"latent_dim": 30, "reg": 0.1, "symmetric": True, "with_adjoint": True, "lr": 0.1, "T_max": 20000},
            tied_encoders=False,
        ),
        AlgorithmTestConfig(
            algorithm_name="csrlearning.algorithms.UnconstrainedCCAAlgorithm",
            algo_kwargs={},
            tied_encoders=False,
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
        help="Run a single algorithm by full class path (e.g., csrlearning.algorithms.CSEBilevelStoGrad)",
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
