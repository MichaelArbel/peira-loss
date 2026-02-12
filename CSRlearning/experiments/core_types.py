from __future__ import annotations


from typing import Any, Dict, Optional, Protocol, Tuple

import torch





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
