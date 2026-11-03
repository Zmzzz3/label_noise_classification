"""Shared framework for noisy-label classification experiments."""

from .config import TrainConfig


def train_model(*args, **kwargs):
    """Lazily import PyTorch so lightweight result tools remain usable without it."""

    from .experiment import train_model as implementation

    return implementation(*args, **kwargs)


__all__ = ["TrainConfig", "train_model"]

