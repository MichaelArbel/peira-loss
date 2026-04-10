from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class OptimizerEntry:
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None


@dataclass
class OptimizerBundle:
    entries: List[OptimizerEntry] = field(default_factory=list)

    def add(self, optimizer: Optional[Any] = None, scheduler: Optional[Any] = None) -> None:
        self.entries.append(
            OptimizerEntry(
                optimizer=optimizer,
                scheduler=scheduler,
            )
        )

    def zero_grad(self) -> None:
        for entry in self.entries:
            if entry.optimizer:
                entry.optimizer.zero_grad()

    def step(self) -> None:
        for entry in self.entries:
            if entry.optimizer:
                entry.optimizer.step()
           
    def schedule(self)-> None:
        for entry in self.entries:
            if entry.scheduler is not None:
                entry.scheduler.step()
