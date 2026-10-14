#!/usr/bin/env python3
"""Run the full PoC pipeline on Apache Camel."""

import os

# Avoid OpenMP/threading crashes on macOS (XGBoost / sklearn)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import EMBEDDING_SAMPLE_SIZE, RESULTS_DIR
from data_loader import chronological_split, load_camel_commits
from embeddings import attach_embeddings
from train_eval import run_experiments


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading Apache Camel commits...")
    df = load_camel_commits()
    print(f"   {len(df)} commits, {df['buggy'].mean():.1%} buggy")

    print("2. Chronological split 70/10/20...")
    train, val, test = chronological_split(df)

    print("3. Extracting CodeBERT embeddings...")
    # Use same commits for fair comparison; sample speeds up first run
    n = EMBEDDING_SAMPLE_SIZE
    if n:
        print(f"   (PoC sample: {n} commits per split — set EMBEDDING_SAMPLE_SIZE=None for full run)")
        train = attach_embeddings(train, sample_size=min(n, len(train)))
        val = attach_embeddings(val, sample_size=min(n // 5, len(val)))
        test = attach_embeddings(test, sample_size=min(n // 5, len(test)))
    else:
        train = attach_embeddings(train)
        val = attach_embeddings(val)
        test = attach_embeddings(test)

    print("4. Training & evaluating 9 models...")
    results = run_experiments(train, val, test)

    out = RESULTS_DIR / "poc_results.csv"
    results.to_csv(out, index=False)
    print(f"\nDone. Results saved to {out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
