from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the assignment NPZ files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz files found under {args.data_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            print(path.name)
            for key in ("Xtr", "Str", "Xts", "Yts"):
                value = archive[key]
                detail = f"shape={value.shape}, dtype={value.dtype}"
                if key in {"Str", "Yts"}:
                    labels, counts = np.unique(value, return_counts=True)
                    detail += f", counts={dict(zip(labels.tolist(), counts.tolist()))}"
                print(f"  {key}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

