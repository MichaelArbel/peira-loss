from typing import Optional, Tuple

import torch

from experiments.exp_types import DataInfo
from csrlearning.core_types import DataSource


class DatasetDataSource:
    """
    Wraps a finite dataloader to provide an infinite __next__() stream by
    restarting the iterator on StopIteration.
    """
    def __init__(self, dataloader, info: Optional[DataInfo] = None):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.info = info

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        return self.__next__()
