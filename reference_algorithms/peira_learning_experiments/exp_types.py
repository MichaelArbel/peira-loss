from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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


@dataclass(frozen=True)
class DataInfo:
    singular_vectors_1: torch.Tensor
    singular_vectors_2: torch.Tensor
    cov_1: torch.Tensor
    cov_2: torch.Tensor
    intra_spectrum_1: torch.Tensor
    inter_spectrum: torch.Tensor
    spectrum: torch.Tensor

@dataclass(frozen=True)
class SpectralMetrics:
    spectrum_ratio: torch.Tensor
    alignment: torch.Tensor
    target_ratio: torch.Tensor
