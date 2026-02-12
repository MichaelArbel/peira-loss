from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class EncodedBatch:
    views: Tuple[torch.Tensor, ...]

    @property
    def x1(self) -> torch.Tensor:
        return self.views[0]

    @property
    def x2(self) -> torch.Tensor:
        return self.views[1]


@dataclass
class AlgorithmState:
    pred: Any = None
    adjoint: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)
    stop_grad: bool = False


@dataclass(frozen=True)
class DataInfo:
    singular_vectors_1: torch.Tensor
    singular_vectors_2: torch.Tensor
    cov_1: torch.Tensor
    cov_2: torch.Tensor
    intra_spectrum_1: torch.Tensor
    intra_spectrum_2: torch.Tensor
    inter_spectrum: torch.Tensor
    spectrum: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self.to_dict().items()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DataSource(Protocol):
    """
    Protocol for infinite data sources. Implementations must allow calling
    __next__() indefinitely without raising StopIteration.
    """
    info: Optional[DataInfo]

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        ...

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        ...


@dataclass(frozen=True)
class SpectralMetrics:
    spectrum: torch.Tensor
    target_spectrum: torch.Tensor
    local_alignment: torch.Tensor
    alignment: torch.Tensor
