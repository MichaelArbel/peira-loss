from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import torch


@dataclass(frozen=False)
class AlgorithmState:
    pred: Any = None
    adjoint: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)
    stop_grad: bool = False


class DataSource(Protocol):
    """
    Protocol for infinite data sources. Implementations must allow calling
    __next__() indefinitely without raising StopIteration.
    """
    info: Optional[Any]

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        ...

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        ...
