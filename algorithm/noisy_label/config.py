from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by every compared classifier."""

    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    tlr: float = 1e-3
    reg: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    min_delta: float = 1e-4
    validation_fraction: float = 0.2
    num_workers: int = 0
    dropout: float = 0.25
    device: str = "auto"
    deterministic: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.patience <= 0 or self.min_delta < 0:
            raise ValueError("invalid early-stopping settings")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

