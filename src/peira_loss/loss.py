"""PEIRA loss (Arbel, Terver, Ponce, NeurIPS 2026, arXiv:2605.17671).

Extracted from ``eb_jepa.losses.anticollapse.PEIRALoss``. The ``state_dict`` keys
(``intra_cov``, ``inter_cov``, ``_extra_state``) are unchanged, so checkpoints of the
paper runs load as is.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from peira_loss.distributed import _total_batch_size, all_reduce


def _add_ridge(A: torch.Tensor, reg: float) -> torch.Tensor:
    """Return ``A + reg * I``."""
    return A + reg * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)


def _exact_inverse(A: torch.Tensor, max_jitter_tries: int = 4) -> torch.Tensor:
    """Inverse of an SPD matrix via Cholesky in fp64.

    The input and output are symmetrized. If Cholesky fails, it is retried with a
    diagonal jitter ``10**k * mean(diag(A)) * 1e-6``; the last resort is an
    eigendecomposition with clamped eigenvalues.
    """
    orig_dtype = A.dtype
    A64 = A.to(torch.float64)
    A64 = 0.5 * (A64 + A64.t())
    eye = torch.eye(A64.shape[-1], device=A64.device, dtype=A64.dtype)
    base_jitter = A64.diagonal().mean().clamp_min(1e-12) * 1e-6

    for k in range(max_jitter_tries):
        try:
            M = A64 if k == 0 else A64 + (10.0**k) * base_jitter * eye
            Q = torch.cholesky_inverse(torch.linalg.cholesky(M))
            return (0.5 * (Q + Q.t())).to(orig_dtype)
        except torch._C._LinAlgError:
            continue

    w, V = torch.linalg.eigh(A64)
    Q = (V * (1.0 / w.clamp_min(base_jitter))) @ V.t()
    return (0.5 * (Q + Q.t())).to(orig_dtype)


class PEIRALoss(nn.Module):
    """PEIRA: self-distillation with the closed-form ridge predictor.

    Keeps EMA estimates of the symmetrized second moments of the two views,
    ``Sigma = E[V U^T + U V^T]`` (``inter_cov``) and ``N = E[U U^T + V V^T]``
    (``intra_cov``), with an EMA rate ``eta`` cosine-annealed from ``lr`` to
    ``lr_min`` over ``T_max`` steps (Algorithm 1). The predictor is
    ``P = Sigma Q`` with ``Q = (N + reg I)^-1``, both stop-gradient, and the encoders
    receive the gradient of the PEIRA objective through the auxiliary loss
    ``L_aux`` plus the ridge ``0.5 reg E[||u||^2 + ||v||^2]``.

    Args:
        latent_dim: Dimension ``k`` of the representations.
        T_max: Number of training steps of the cosine schedule on ``eta``.
        reg: Ridge ``lambda``.
        lr: Initial EMA rate ``eta_init`` (weight of the current batch), in (0, 1].
        lr_min: Final EMA rate ``eta_min``, in (0, lr].
        predictor_cov: Which ``Sigma`` enters ``P``. ``"batch"`` uses the current
            batch's ``Sigma`` (the code that produced the paper's results);
            ``"ema"`` uses the EMA estimate, as written in Algorithm 1. ``Q``
            always uses the EMA ``N``.
    """

    def __init__(
        self,
        latent_dim: int,
        T_max: int,
        reg: float = 0.3,
        lr: float = 0.9,
        lr_min: float = 0.5,
        predictor_cov: Literal["batch", "ema"] = "batch",
    ):
        super().__init__()
        # The EMA update is a convex combination of PSD matrices only for lr <= 1.
        if not 0.0 < lr <= 1.0:
            raise ValueError(f"PEIRA `lr` must be in (0, 1]; got {lr}.")
        if not 0.0 < lr_min <= lr:
            raise ValueError(f"PEIRA `lr_min` must be in (0, lr]; got lr_min={lr_min}, lr={lr}.")
        if predictor_cov not in ("batch", "ema"):
            raise ValueError(f"PEIRA `predictor_cov` must be 'batch' or 'ema'; got {predictor_cov}.")
        self.reg = reg
        self.T_max = T_max
        self.lr = lr
        self.lr_min = lr_min
        self.predictor_cov = predictor_cov
        self._step = 0
        # Buffers, not parameters: DDP must not all-reduce them. Batch statistics are
        # all-reduced in forward, so every rank applies the same update.
        zeros = torch.zeros(latent_dim, latent_dim)
        self.register_buffer("intra_cov", zeros.clone())
        self.register_buffer("inter_cov", zeros.clone())

    @property
    def _current_lr(self) -> float:
        """Cosine-annealed EMA rate, as ``CosineAnnealingLR(T_max, eta_min=lr_min)``."""
        t = min(self._step, self.T_max)
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * t / self.T_max))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the PEIRA loss and update the EMA covariances.

        Args:
            z1: First-view representations [B, D].
            z2: Second-view representations [B, D].

        Returns:
            Dict with ``loss`` (gradient-bearing ``L_aux`` + ridge) and detached
            diagnostics ``peira_loss`` (objective ``-0.5 Tr(P) + ridge``), ``trace_P``,
            ``cse_loss``, ``reg_loss`` and ``peira_lr``.
        """
        # fp32 outside autocast: bf16 rounding on X^T X can break positive definiteness.
        with torch.amp.autocast("cuda", enabled=False):
            x = z1.detach().float()
            y = z2.detach().float()
            B_global = _total_batch_size(z1.shape[0])
            inter_cov = all_reduce(y.t() @ x, op="SUM")
            intra_cov = all_reduce(x.t() @ x, op="SUM") + all_reduce(y.t() @ y, op="SUM")
            inter_cov = (inter_cov + inter_cov.t()) / B_global
            intra_cov = intra_cov / B_global

            lr = self._current_lr
            self.intra_cov.sub_(lr * (self.intra_cov - intra_cov))
            self.inter_cov.sub_(lr * (self.inter_cov - inter_cov))

            Q = _exact_inverse(_add_ridge(self.intra_cov.float(), self.reg))
            sigma = inter_cov if self.predictor_cov == "batch" else self.inter_cov.float()
            P = sigma @ Q

        reg_loss = 0.5 * self.reg * (z1.square().sum(dim=1).mean() + z2.square().sum(dim=1).mean())
        cse_loss = 0.5 * torch.mean(torch.sum((z1 @ P.t() - z2) * (z1 @ Q.t()), dim=1)) + 0.5 * torch.mean(
            torch.sum((z2 @ P.t() - z1) * (z2 @ Q.t()), dim=1)
        )
        trace_P = torch.trace(P)
        self._step += 1
        return {
            "loss": cse_loss + reg_loss,
            "peira_loss": (-0.5 * trace_P + reg_loss).detach(),
            "trace_P": trace_P,
            "cse_loss": cse_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "peira_lr": torch.tensor(lr, device=z1.device),
        }

    def get_extra_state(self) -> dict:
        """Save the step counter that drives the ``eta`` schedule, for resuming."""
        return {"_step": self._step}

    def set_extra_state(self, state: dict) -> None:
        self._step = int(state["_step"])
