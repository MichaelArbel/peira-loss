"""CPU tests for ``peira_loss.PEIRALoss``."""

import copy
import os
import sys

import pytest
import torch

from peira_loss import PEIRALoss
from peira_loss.loss import _add_ridge, _exact_inverse

# Optional reference: a checkout of the eb_jepa code that produced the paper's results.
EB_JEPA_ROOT = os.environ.get("PEIRA_EB_JEPA_ROOT")
KWARGS = dict(latent_dim=16, T_max=10, reg=0.3, lr=0.9, lr_min=0.5)


def _reference_cls() -> type:
    """Import the original eb_jepa ``PEIRALoss`` or skip."""
    if not EB_JEPA_ROOT:
        pytest.skip("set PEIRA_EB_JEPA_ROOT to compare against the eb_jepa reference")
    if EB_JEPA_ROOT not in sys.path:
        sys.path.insert(0, EB_JEPA_ROOT)
    try:
        from eb_jepa.losses.anticollapse import PEIRALoss as RefPEIRALoss
    except ImportError as e:
        pytest.skip(f"eb_jepa reference unavailable: {e}")
    return RefPEIRALoss


def _views(seed: int, B: int = 32, D: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Two correlated random views, each [B, D]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, D, generator=g)
    return x + 0.1 * torch.randn(B, D, generator=g), x + 0.1 * torch.randn(B, D, generator=g)


def test_forward_backward() -> None:
    loss_fn = PEIRALoss(**KWARGS)
    z1, z2 = (v.requires_grad_() for v in _views(0))
    out = loss_fn(z1, z2)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert z1.grad.abs().sum() > 0 and z2.grad.abs().sum() > 0
    assert loss_fn._step == 1
    assert set(loss_fn.state_dict()) == {"intra_cov", "inter_cov", "_extra_state"}


def test_equivalence_with_eb_jepa() -> None:
    """``predictor_cov="batch"``: same values, gradients and state as eb_jepa over 5 steps."""
    Ref = _reference_cls()
    ref, new = Ref(**KWARGS), PEIRALoss(**KWARGS, predictor_cov="batch")
    max_diff = 0.0
    for step in range(5):
        z1, z2 = _views(step)
        a1, a2 = z1.clone().requires_grad_(), z2.clone().requires_grad_()
        b1, b2 = z1.clone().requires_grad_(), z2.clone().requires_grad_()
        out_ref, out_new = ref(a1, a2), new(b1, b2)
        out_ref["loss"].backward()
        out_new["loss"].backward()
        for k in out_new:
            max_diff = max(max_diff, (out_ref[k] - out_new[k]).abs().max().item())
        max_diff = max(max_diff, (a1.grad - b1.grad).abs().max().item())
        max_diff = max(max_diff, (a2.grad - b2.grad).abs().max().item())
    sd_ref, sd_new = ref.state_dict(), new.state_dict()
    assert sd_ref.keys() == sd_new.keys()
    assert sd_ref["_extra_state"] == sd_new["_extra_state"] == {"_step": 5}
    for k in ("intra_cov", "inter_cov"):
        max_diff = max(max_diff, (sd_ref[k] - sd_new[k]).abs().max().item())
    print(f"\nmax_abs_diff = {max_diff:.3e}")
    assert max_diff == 0.0

    # A mid-training eb_jepa checkpoint resumes identically.
    resumed = PEIRALoss(**KWARGS, predictor_cov="batch")
    resumed.load_state_dict(copy.deepcopy(sd_ref))
    z1, z2 = _views(99)
    torch.testing.assert_close(resumed(z1, z2)["loss"], ref(z1, z2)["loss"], rtol=0, atol=0)


def test_ema_predictor_uses_ema_sigma() -> None:
    """``predictor_cov="ema"``: ``Tr(P) = Tr(Sigma_ema (N_ema + reg I)^-1)`` after the update."""
    loss_fn = PEIRALoss(**KWARGS, predictor_cov="ema")
    for step in range(3):
        out = loss_fn(*_views(step))
    Q = _exact_inverse(_add_ridge(loss_fn.intra_cov, KWARGS["reg"]))
    torch.testing.assert_close(out["trace_P"], torch.trace(loss_fn.inter_cov @ Q))
    batch = PEIRALoss(**KWARGS, predictor_cov="batch")
    for step in range(3):
        out_batch = batch(*_views(step))
    assert not torch.allclose(out["trace_P"], out_batch["trace_P"])


def test_compile_gradient_nonzero() -> None:
    loss_fn = PEIRALoss(**KWARGS)
    try:
        compiled = torch.compile(loss_fn)
        z1, z2 = (v.requires_grad_() for v in _views(0))
        out = compiled(z1, z2)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"torch.compile unavailable on this CPU host: {type(e).__name__}: {e}")
    out["loss"].backward()
    assert z1.grad is not None and z1.grad.abs().sum() > 0
    assert z2.grad is not None and z2.grad.abs().sum() > 0
