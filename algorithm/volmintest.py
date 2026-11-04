from __future__ import annotations

import sys
from pathlib import Path

from noisy_label.config import TrainConfig
from noisy_label.experiment import train_model
from noisy_label.results import summarize_runs, write_runs, write_summary

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def exp(tlr, reg, dataset, output, lr=1e-3) -> int:

    config = TrainConfig(
        learning_rate=lr,
        tlr=tlr,
        reg=reg,
    )

    runs_path = output / "runs.csv"
    summary_path = output / "summary.json"
    rows = []

    for seed in SEEDS:
        metrics = train_model(
            dataset_path=dataset,
            method="volmin",
            transition_matrix=None,
            seed=seed,
            config=config,
        )
        rows.append(metrics)
        rows.sort(key=lambda row: int(row["seed"]))
        write_runs(runs_path, rows)
        write_summary(summary_path, summarize_runs(rows), config.to_dict())
        print(
            f"seed={seed} test_accuracy={metrics['test_accuracy']:.3f}% "
            f"best_epoch={metrics['best_epoch']}",
            flush=True,
        )

    if not rows:
        print("No runs available to summarize", file=sys.stderr)
        return 1
    summary = summarize_runs(rows)
    print(
        f"{summary['dataset']} / {summary['method']}: "
        f"{summary['test_accuracy_mean']:.3f} +/- "
        f"{summary['test_accuracy_sample_std']:.3f}% "
        f"over {summary['number_of_runs']} run(s)"
    )
    return 0

def lr():
    lr = [0.001, 0.01, 0.1]
    reg = 0.0001

    for l in lr:
        print(l, reg)
        path = Path("data/FashionMNIST0.3.npz")
        output = Path("results") / "mnist0.3" / f"volmin_lr{l}"
        exp(l, reg, path, output)

        path = Path("data/FashionMNIST0.6.npz")
        output = Path("results") / "mnist0.6" / f"volmin_lr{l}"
        exp(l, reg, path, output)

        path = Path("data/CIFAR.npz")
        output = Path("results") / "cifar" / f"volmin_lr{l}"
        exp(l, reg, path, output)

def mnist03():
    path = Path("data/FashionMNIST0.3.npz")
    lr = 0.1
    reg = [0.0001, 0.0005, 0.001]

    for r in reg:
        print(lr, r)
        output = Path("results") / "mnist0.3" / f"volmin_reg{r}"
        exp(lr, r, path, output)

def mnist06():
    path = Path("data/FashionMNIST0.6.npz")
    lr = 0.1
    reg = [0.0001, 0.0005, 0.001]

    for r in reg:
        print(lr, r)
        output = Path("results") / "mnist0.6" / f"volmin_reg{r}"
        exp(lr, r, path, output)

def cifar():
    path = Path("data/CIFAR.npz")
    lr = 0.1
    reg = [0.0001, 0.0005, 0.001]

    for r in reg:
        print(lr, r)
        output = Path("results") / "cifar" / f"volmin_reg{r}"
        exp(lr, r, path, output)

lr()
mnist03()
mnist06()
cifar()