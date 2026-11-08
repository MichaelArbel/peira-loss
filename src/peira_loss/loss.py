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
    """PEIRA: Trace loss with the closed-form ridge predictor.

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
            batch's ``Sigma`` (the ImageNet-1K ablations of Appendix G.1);
            ``"ema"`` uses the EMA estimate, as in Algorithm 1 and Table 1. ``Q``
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
        with torch.autocast(device_type=z1.device.type, enabled=False):
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

class SelfDistillationLoss(nn.Module):
    """SelfDistillationLoss: self-distillation with the closed-form ridge predictor.

    Keeps EMA estimates of the symmetrized second moments of the two views,
    ``Sigma = E[V U^T + U V^T]`` (``inter_cov``) and ``N = E[U U^T + V V^T]``
    (``intra_cov``), with an EMA rate ``eta`` cosine-annealed from ``lr`` to
    ``lr_min`` over ``T_max`` steps (Algorithm 1). The predictor is
    ``P = Sigma Q`` with ``Q = (N + reg I)^-1``, both stop-gradient. The first
    batch initializes both buffers directly; later batches use EMA updates.
    The loss sums both prediction directions with detached targets, plus
    ``0.5 reg E[||u||^2 + ||v||^2]``.

    Args:
        latent_dim: Dimension ``k`` of the representations.
        T_max: Number of training steps of the cosine schedule on ``eta``.
        reg: Ridge ``lambda``.
        lr: Initial EMA rate ``eta_init`` (weight of the current batch), in (0, 1].
        lr_min: Final EMA rate ``eta_min``, in (0, lr].
        predictor_cov: Which ``Sigma`` enters ``P``. ``"ema"`` (default) uses
            the EMA estimate; ``"batch"`` uses the current batch's ``Sigma``.
            ``Q`` always uses the EMA ``N``.
    """

    def __init__(
        self,
        latent_dim: int,
        T_max: int,
        reg: float = 0.3,
        lr: float = 0.9,
        lr_min: float = 0.5,
        predictor_cov: Literal["batch", "ema"] = "ema",
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
        """Compute symmetric self-distillation and update the EMA covariances.

        Args:
            z1: First-view representations [B, D].
            z2: Second-view representations [B, D].

        Returns:
            Dict with ``loss`` (symmetric prediction loss plus feature ridge)
            and ``ssl_lr`` (the scheduled EMA rate).
        """
        # fp32 outside autocast: bf16 rounding on X^T X can break positive definiteness.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            x = z1.detach().float()
            y = z2.detach().float()
            B_global = _total_batch_size(z1.shape[0])
            inter_cov = all_reduce(y.t() @ x, op="SUM")
            intra_cov = all_reduce(x.t() @ x, op="SUM") + all_reduce(y.t() @ y, op="SUM")
            inter_cov = (inter_cov + inter_cov.t()) / B_global
            intra_cov = intra_cov / B_global

            lr = self._current_lr
            if self._step == 0:
                self.intra_cov.copy_(intra_cov)
                self.inter_cov.copy_(inter_cov)
            else:
                self.intra_cov.sub_(lr * (self.intra_cov - intra_cov))
                self.inter_cov.sub_(lr * (self.inter_cov - inter_cov))

            M = self.intra_cov.double()
            M = M + self.reg * torch.eye(
                M.shape[0], device=M.device, dtype=M.dtype
            )
            S = inter_cov.double() if self.predictor_cov == "batch" else self.inter_cov.double()
            P = torch.linalg.solve(M, S.T).T.float()


        distill_loss = self.compute_self_distillation_loss(z1, z2, P.detach(), self.reg)


        reg_loss = 0.5 * self.reg * (z1.square().sum(dim=1).mean() + z2.square().sum(dim=1).mean())

        self._step += 1
        return {
            "loss": distill_loss+reg_loss,
            "ssl_lr": torch.tensor(lr, device=z1.device),
        }

    def compute_self_distillation_loss(self, X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float) -> torch.Tensor:
        data_term = 0.5 * torch.mean(torch.sum((X @ P.T - Y.detach()) ** 2, dim=1))
        data_term = data_term + 0.5 * torch.mean(torch.sum((Y @ P.T - X.detach()) ** 2, dim=1))
        return data_term

    def get_extra_state(self) -> dict:
        """Save the step counter that drives the ``eta`` schedule, for resuming."""
        return {"_step": self._step}

    def set_extra_state(self, state: dict) -> None:
        self._step = int(state["_step"])

class JointSelfDistillationLoss(nn.Module):
    """Self-distillation with a zero-initialized, implicitly updated predictor.

    Forward uses the current predictor and detached targets. After encoder
    backward/step, call ``step_predictor`` exactly once for the cached batch.
    It solves P_new (I + eta H) = P_old + eta C in fp64, where H is the
    batch second moment plus ridge. No SGD optimizer should own the predictor.
    """

    def __init__(
        self, latent_dim: int, reg: float = 1e-4, symmetric: bool = False,
        lr: float = 0.01, lr_min: float | None = None, T_max: int = 1000,
    ):
        super().__init__()
        lr_min = lr if lr_min is None else lr_min
        if reg < 0 or not 0 < lr_min <= lr or T_max <= 0:
            raise ValueError("Require reg >= 0, 0 < lr_min <= lr, and T_max > 0")
        self.reg = reg
        self.symmetric = symmetric
        self.lr, self.lr_min, self.T_max = lr, lr_min, T_max
        self._step = 0
        self._predictor_inputs = None
        self.register_buffer("predictor", torch.zeros(latent_dim, latent_dim))

    @property
    def _current_lr(self) -> float:
        t = min(self._step, self.T_max)
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (
            1 + math.cos(math.pi * t / self.T_max)
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=z1.device.type, enabled=False):
            x, y = z1.float(), z2.float()
            prediction_loss = 0.5 * (x @ self.predictor.t() - y.detach()).square().sum(1).mean()
            feature_reg = 0.5 * self.reg * x.square().sum(1).mean()
            if self.symmetric:
                prediction_loss = prediction_loss + 0.5 * (
                    y @ self.predictor.t() - x.detach()
                ).square().sum(1).mean()
                feature_reg = feature_reg + 0.5 * self.reg * y.square().sum(1).mean()
            predictor_reg = 0.5 * self.reg * self.predictor.square().sum()
            loss = prediction_loss + feature_reg + predictor_reg
        self._predictor_inputs = (x.detach(), y.detach())
        return {
            "loss": loss,
            "prediction_loss": prediction_loss.detach(),
            "feature_reg": feature_reg.detach(),
            "predictor_reg": predictor_reg.detach(),
            "predictor_lr": torch.tensor(self._current_lr, device=z1.device),
        }

    @torch.no_grad()
    def step_predictor(self) -> None:
        """Implicit regression step on the cached, globally aggregated batch."""
        if self._predictor_inputs is None:
            raise RuntimeError("Call forward and finish backward before step_predictor")
        x, y = self._predictor_inputs
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, y = x.double(), y.double()
            B = _total_batch_size(x.shape[0])
            A = all_reduce(x.t() @ x, op="SUM") / B
            C = all_reduce(y.t() @ x, op="SUM") / B
            if self.symmetric:
                A = A + all_reduce(y.t() @ y, op="SUM") / B
                C = C + C.t()
            A = 0.5 * (A + A.t())
            eta = self._current_lr
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            M = eye + eta * (A + self.reg * eye)
            rhs = self.predictor.double() + eta * C
            updated = torch.linalg.solve(M, rhs.t()).t().to(self.predictor.dtype)
            if not torch.isfinite(updated).all():
                raise FloatingPointError("Non-finite implicit predictor update")
            self.predictor.copy_(updated)
        self._predictor_inputs = None
        self._step += 1

    def get_extra_state(self) -> dict:
        return {"_step": self._step}

    def set_extra_state(self, state: dict) -> None:
        self._step = int(state["_step"])
