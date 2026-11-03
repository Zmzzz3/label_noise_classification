from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


KNOWN_TRANSITIONS: dict[str, np.ndarray] = {
    "FashionMNIST0.3": np.array(
        [[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]], dtype=np.float32
    ),
    "FashionMNIST0.6": np.array(
        [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]], dtype=np.float32
    ),
}


def validate_transition(matrix: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"transition matrix must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("transition matrix must contain finite non-negative values")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=atol):
        raise ValueError("each transition-matrix row must sum to one")
    return matrix


def resolve_transition(
    value: str | Path | np.ndarray | None,
    dataset_name: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve a known or file-based T[i,j]=P(noisy=j | clean=i) matrix."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        matrix = value
    else:
        value = str(value)
        if value.lower() == "auto":
            if dataset_name not in KNOWN_TRANSITIONS:
                raise ValueError(
                    f"no provided transition matrix for {dataset_name}; pass Member 4's .npy or JSON file"
                )
            matrix = KNOWN_TRANSITIONS[dataset_name]
        else:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() == ".npy":
                matrix = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == ".json":
                matrix = np.asarray(json.loads(path.read_text()), dtype=np.float32)
            else:
                raise ValueError("transition matrix must be a .npy or .json file")
    return torch.as_tensor(validate_transition(matrix), dtype=torch.float32, device=device)

