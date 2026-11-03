from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from noisy_label.config import TrainConfig
from noisy_label.data import load_npz
from noisy_label.experiment import train_model
from noisy_label.results import load_runs, summarize_runs, write_runs, write_summary

SEEDS = [0]


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


def mnist06():
    path = Path("data/FashionMNIST0.6.npz")
    tlr = [0.015, 0.02, 0.025]
    reg = [0.003, 0.0035, 0.004]

    for t in tlr:
        for r in reg:
            print(t, r)
            output = Path("results") / "mnist0.6" / f"volmin_tlr{tlr}_reg{reg}"
            exp(t, r, path, output)

mnist06()