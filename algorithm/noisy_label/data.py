from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_KEYS = {"Xtr", "Str", "Xts", "Yts"}


@dataclass(frozen=True)
class DatasetArrays:
    train_images: np.ndarray
    noisy_train_labels: np.ndarray
    test_images: np.ndarray
    clean_test_labels: np.ndarray
    dataset_name: str
    in_channels: int


def _validate_images(images: np.ndarray, name: str) -> int:
    if images.ndim == 2 and images.shape[1] == 28 * 28:
        return 1
    if images.ndim == 3 and images.shape[1:] == (28, 28):
        return 1
    if images.ndim == 4 and images.shape[1:] == (32, 32, 3):
        return 3
    if images.ndim == 4 and images.shape[1:] == (3, 32, 32):
        return 3
    raise ValueError(f"unsupported {name} image shape: {images.shape}")


def _validate_labels(labels: np.ndarray, expected_size: int, name: str) -> None:
    if labels.ndim != 1 or len(labels) != expected_size:
        raise ValueError(f"{name} must have shape ({expected_size},), got {labels.shape}")
    observed = set(np.unique(labels).tolist())
    if not observed.issubset({0, 1, 2}):
        raise ValueError(f"{name} contains labels outside {{0, 1, 2}}: {sorted(observed)}")


def load_npz(path: str | Path) -> DatasetArrays:
    """Load and validate one assignment dataset without changing its arrays."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")
        x_train = archive["Xtr"]
        s_train = archive["Str"].astype(np.int64, copy=False)
        x_test = archive["Xts"]
        y_test = archive["Yts"].astype(np.int64, copy=False)

    train_channels = _validate_images(x_train, "training")
    test_channels = _validate_images(x_test, "test")
    if train_channels != test_channels:
        raise ValueError("training and test images have different channel counts")
    _validate_labels(s_train, len(x_train), "Str")
    _validate_labels(y_test, len(x_test), "Yts")

    return DatasetArrays(
        train_images=x_train,
        noisy_train_labels=s_train,
        test_images=x_test,
        clean_test_labels=y_test,
        dataset_name=path.stem,
        in_channels=train_channels,
    )


def random_split_indices(
    n_examples: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fresh disjoint random 80/20 split for one run."""

    if n_examples < 2:
        raise ValueError("at least two examples are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_examples)
    n_validation = int(round(n_examples * validation_fraction))
    n_validation = min(max(n_validation, 1), n_examples - 1)
    validation_indices = permutation[:n_validation]
    training_indices = permutation[n_validation:]
    return training_indices, validation_indices


def to_chw_float(image: np.ndarray) -> torch.Tensor:
    """Convert a FashionMNIST or CIFAR image to float32 CHW in [0, 1]."""

    if image.ndim == 1 and image.size == 28 * 28:
        image = image.reshape(1, 28, 28)
    elif image.ndim == 2 and image.shape == (28, 28):
        image = image[None, :, :]
    elif image.ndim == 3 and image.shape == (32, 32, 3):
        image = np.moveaxis(image, -1, 0)
    elif image.ndim == 3 and image.shape == (3, 32, 32):
        pass
    else:
        raise ValueError(f"unsupported individual image shape: {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image)).to(torch.float32).div_(255.0)


class ImageArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        if indices is None:
            indices = np.arange(len(images), dtype=np.int64)
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[position])
        image = to_chw_float(self.images[index])
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_loaders(
    arrays: DatasetArrays,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        ImageArrayDataset(arrays.train_images, arrays.noisy_train_labels, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        ImageArrayDataset(
            arrays.train_images, arrays.noisy_train_labels, validation_indices
        ),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        ImageArrayDataset(arrays.test_images, arrays.clean_test_labels),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader

