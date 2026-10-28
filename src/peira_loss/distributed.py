"""Distributed helpers for PEIRA (no-ops when ``torch.distributed`` is not initialized).

PEIRA assumes plain data parallelism: every rank holds a shard of the same global
batch, and the batch second moments are summed across ranks with ``all_reduce``
so all ranks apply the same covariance-buffer update.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


@torch._dynamo.disable
def all_reduce(x: torch.Tensor, op: str) -> torch.Tensor:
    """In-place all-reduce across ranks; identity when not distributed.

    ``@torch._dynamo.disable`` is required: under ``torch.compile`` the in-place
    ``dist.all_reduce`` is functionalized into an op without an autograd formula,
    which silently detaches the result and removes the gradient of every loss
    built on top of it.

    Args:
        x: Tensor to reduce, any shape (e.g. a [D, D] second-moment matrix).
        op: Name of a ``torch.distributed.ReduceOp`` member, e.g. ``"SUM"``.

    Returns:
        The reduced tensor (same object as ``x``), same shape.
    """
    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    return x


def _total_batch_size(local_n: int) -> int:
    """Return the global batch size ``local_n * world_size`` (``local_n`` if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return local_n * dist.get_world_size()
    return local_n

