from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from .config import TrainConfig
from .data import load_npz, make_loaders, random_split_indices
from .losses import get_loss
from .model import SmallCNN
from .transitions import resolve_transition
from .volminloss import VolMinLoss

def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    loader,
    loss_function,
    transition_matrix: torch.Tensor | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    # for VolMinLoss
    if isinstance(loss_function, nn.Module):
        loss_function.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels, transition_matrix)
            if not torch.isfinite(loss):
                raise FloatingPointError("encountered a non-finite loss")
            if is_training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += batch_size
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def _evaluate_clean_test(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.size(0)
    return 100.0 * correct / total


def train_model(
    dataset_path: str | Path,
    method: str,
    transition_matrix: str | Path | np.ndarray | None,
    seed: int,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Train one run with a fresh random split and return auditable metrics."""

    config = config or TrainConfig()
    config.validate()
    seed_everything(seed, config.deterministic)
    arrays = load_npz(dataset_path)
    train_indices, validation_indices = random_split_indices(
        len(arrays.train_images), config.validation_fraction, seed
    )
    train_loader, validation_loader, test_loader = make_loaders(
        arrays,
        train_indices,
        validation_indices,
        config.batch_size,
        seed,
        config.num_workers,
    )
    device = choose_device(config.device)
    model = SmallCNN(arrays.in_channels, dropout=config.dropout).to(device)
    
    # VolMinLoss needs a stateful loss function that allows updates to T
    loss_spec = get_loss(method)
    loss_fn = loss_spec.function
    if isinstance(loss_fn, type) and issubclass(loss_fn, nn.Module):
        loss_fn = loss_fn(reg=config.reg).to(device)

    requires_transition = loss_spec.requires_transition
    transition = resolve_transition(transition_matrix, arrays.dataset_name, device)
    if requires_transition and transition is None:
        raise ValueError(f"method '{method}' requires a transition matrix")

    groups = [{"params": model.parameters()}]
    if isinstance(loss_fn, nn.Module):
        # This sets the decay to 0 for T in VolMinLoss
        groups.append({"params": loss_fn.parameters(), "weight_decay": 0.0, "lr": config.tlr}) # type: ignore[arg-type]
    optimizer = AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = float("nan")
    best_T = None
    epochs_without_improvement = 0
    started_at = time.perf_counter()
    epochs_completed = 0

    for epoch in range(1, config.epochs + 1):
        _run_epoch(
            model,
            train_loader,
            loss_fn,
            transition,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = _run_epoch(
            model,
            validation_loader,
            loss_fn,
            transition,
            device,
            optimizer=None,
        )
        epochs_completed = epoch
        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if isinstance(loss_fn, nn.Module):
                best_T = loss_fn.T
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_T is not None: print(best_T)

    if best_state is None:
        raise RuntimeError("training finished without a valid checkpoint")
    model.load_state_dict(best_state)
    test_accuracy = _evaluate_clean_test(model, test_loader, device)
    elapsed = time.perf_counter() - started_at
    return {
        "dataset": arrays.dataset_name,
        "method": method.lower(),
        "seed": seed,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(arrays.test_images),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "validation_loss": best_validation_loss,
        "validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "training_seconds": elapsed,
        "device": str(device),
        "best_T" : best_T 
    }

