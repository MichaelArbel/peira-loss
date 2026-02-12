from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class OptimizerEntry:
    optimizer: Any
    scheduler: Optional[Any] = None
    managed_by_algorithm: bool = False


@dataclass
class OptimizerBundle:
    entries: List[OptimizerEntry] = field(default_factory=list)

    def add(self, optimizer: Any, scheduler: Optional[Any] = None, managed_by_algorithm: bool = False) -> None:
        self.entries.append(
            OptimizerEntry(
                optimizer=optimizer,
                scheduler=scheduler,
                managed_by_algorithm=managed_by_algorithm,
            )
        )

    def zero_grad(self) -> None:
        for entry in self.entries:
            if not entry.managed_by_algorithm:
                entry.optimizer.zero_grad()

    def step(self) -> None:
        for entry in self.entries:
            if not entry.managed_by_algorithm:
                entry.optimizer.step()
                if entry.scheduler is not None:
                    entry.scheduler.step()
