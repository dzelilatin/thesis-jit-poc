"""Embed the validation split of the Camel PoC (mining is cached, embedding takes about 10 minutes).

The validation split is used by src/tuning_study.py to choose thresholds, balancing strategies and
hyperparameters, so that the test split stays untouched until the final evaluation.

Usage (from the repository root, with the virtual environment active):
    python scripts/embed_validation.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import chronological_split, load_camel_commits
from embeddings import attach_embeddings, pick_device


def main() -> None:
    _, val, _ = chronological_split(load_camel_commits())
    device = pick_device()
    print(f"validation split: {len(val)} commits, {int(val['buggy'].sum())} defect-inducing, device {device}")
    _, stats = attach_embeddings(val, "val", device)
    print(stats)


if __name__ == "__main__":
    main()
