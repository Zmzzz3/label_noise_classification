# Member 1: common experimental framework

This directory contains the shared data pipeline, CNN backbone, training loop,
repeated 80/20 split protocol, early stopping, and result aggregation for
COMP5328 Assignment 2.

## Layout

```text
member1/
├── algorithm/
│   ├── noisy_label/          # shared Python package
│   ├── inspect_datasets.py   # dependency-light data sanity check
│   └── run_experiments.py    # main experiment entry point
├── data/                     # keep empty in the submitted zip
├── report/                   # Member 1 report draft and template
├── results/                  # generated CSV/JSON files
├── tests/
└── requirements.txt
```

The assignment datasets are not included in the submission bundle. Before
running, copy the three `.npz` files into `data/`.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Check the data

```bash
python algorithm/inspect_datasets.py --data-dir data
```

## Run the cross-entropy baseline

One quick run:

```bash
python algorithm/run_experiments.py \
  --dataset data/FashionMNIST0.3.npz \
  --method ce \
  --seeds 0
```

The required ten independently re-split runs:

```bash
python algorithm/run_experiments.py \
  --dataset data/FashionMNIST0.3.npz \
  --method ce \
  --seeds 0:10
```

Repeat for `FashionMNIST0.6.npz` and `CIFAR.npz`. Outputs are written under
`results/<dataset>/<method>/` and include every run plus a mean/sample-standard-
deviation summary. Existing completed seeds are skipped unless `--no-resume` is
passed.

## Connect the other members' methods

Loss implementations register with the shared interface:

```python
from noisy_label.losses import register_loss

@register_loss("forward", requires_transition=True)
def forward_loss(logits, noisy_targets, transition_matrix):
    ...
```

Save the implementation in an importable module and load it from the CLI:

```bash
PYTHONPATH=algorithm python algorithm/run_experiments.py \
  --dataset data/FashionMNIST0.3.npz \
  --method forward \
  --method-module team_methods.forward \
  --transition auto \
  --seeds 0:10
```

For CIFAR, Member 4 should save the estimated row-stochastic matrix as either
`.npy` or JSON and pass its path to `--transition`.

## Fair-comparison rule

Use the same backbone and training arguments for all methods. The default
configuration uses AdamW, batch size 128, at most 100 epochs, and early stopping
after 12 epochs without an improvement in noisy-validation loss. The clean test
set is evaluated only once after restoring the best validation checkpoint.

