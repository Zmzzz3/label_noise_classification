from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


RUN_FIELDS = [
    "dataset",
    "method",
    "seed",
    "train_size",
    "validation_size",
    "test_size",
    "best_epoch",
    "epochs_completed",
    "validation_loss",
    "validation_accuracy",
    "test_accuracy",
    "training_seconds",
    "device",
    "best_T"
]


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in (
            "seed",
            "train_size",
            "validation_size",
            "test_size",
            "best_epoch",
            "epochs_completed",
        ):
            row[key] = int(row[key])
        for key in (
            "validation_loss",
            "validation_accuracy",
            "test_accuracy",
            "training_seconds",
        ):
            row[key] = float(row[key])
    return rows


def write_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero runs")
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    validation_accuracies = [float(row["validation_accuracy"]) for row in rows]
    return {
        "dataset": rows[0]["dataset"],
        "method": rows[0]["method"],
        "number_of_runs": len(rows),
        "seeds": sorted(int(row["seed"]) for row in rows),
        "test_accuracy_mean": mean(test_accuracies),
        "test_accuracy_sample_std": stdev(test_accuracies) if len(rows) > 1 else 0.0,
        "validation_accuracy_mean": mean(validation_accuracies),
        "validation_accuracy_sample_std": (
            stdev(validation_accuracies) if len(rows) > 1 else 0.0
        ),
        "total_training_seconds": sum(float(row["training_seconds"]) for row in rows),
    }


def write_summary(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "training_configuration": config}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

