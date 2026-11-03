from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


LossFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True)
class LossSpec:
    function: LossFunction
    requires_transition: bool


_REGISTRY: dict[str, LossSpec] = {}


def register_loss(name: str, requires_transition: bool = False):
    """Decorator used by Members 2 and 3 to plug in a robust objective."""

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("loss name cannot be empty")

    def decorator(function: LossFunction) -> LossFunction:
        if normalized_name in _REGISTRY:
            raise ValueError(f"loss already registered: {normalized_name}")
        _REGISTRY[normalized_name] = LossSpec(function, requires_transition)
        return function

    return decorator


@register_loss("ce")
def cross_entropy_baseline(
    logits: torch.Tensor,
    noisy_targets: torch.Tensor,
    transition_matrix: torch.Tensor | None,
) -> torch.Tensor:
    del transition_matrix
    return F.cross_entropy(logits, noisy_targets)


def available_losses() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_loss(name: str) -> LossSpec:
    normalized_name = name.strip().lower()
    try:
        return _REGISTRY[normalized_name]
    except KeyError as error:
        raise ValueError(
            f"unknown method '{name}'. Registered methods: {', '.join(available_losses())}"
        ) from error

