from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from noisy_label.config import TrainConfig
from noisy_label.data import load_npz
from noisy_label.experiment import train_model
from noisy_label.results import load_runs, summarize_runs, write_runs, write_summary

 
def parse_seeds(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed range must use start:stop")
        start, stop = map(int, parts)
        seeds = list(range(start, stop))
    else:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated noisy-label classification experiments."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--method-module",
        action="append",
        default=[],
        help="Importable module that registers a team member's loss (repeatable).",
    )
    parser.add_argument(
        "--transition",
        default=None,
        help="'auto' for a provided FashionMNIST T, or a CIFAR .npy/.json path.",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:10"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tlr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for module_name in args.method_module:
        importlib.import_module(module_name)

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        tlr=args.tlr,
        reg=args.reg,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        dropout=args.dropout,
        device=args.device,
        deterministic=not args.non_deterministic,
    )
    arrays = load_npz(args.dataset)
    output_directory = args.results_dir / arrays.dataset_name / args.method.lower()
    runs_path = output_directory / "runs.csv"
    summary_path = output_directory / "summary.json"
    rows = [] if args.no_resume else load_runs(runs_path)
    completed_seeds = {int(row["seed"]) for row in rows}

    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        print(
            f"Running dataset={arrays.dataset_name} method={args.method} seed={seed}",
            flush=True,
        )
        metrics = train_model(
            dataset_path=args.dataset,
            method=args.method,
            transition_matrix=args.transition,
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


if __name__ == "__main__":
    raise SystemExit(main())

