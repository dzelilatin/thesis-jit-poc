#!/usr/bin/env python3
"""Run the PoC pipeline on Apache Camel (ApacheJIT), following the paper pipeline.

    python run_poc.py --smoke                                  # pipeline check, a few minutes
    python run_poc.py                                          # all train / test commits
    python run_poc.py --train-sample 3000 --test-sample 1500   # stratified subsample

Outputs (results/):
    poc_results.csv          mean over seeds per (classifier, config) with 95% bootstrap CIs
    poc_results_by_seed.csv  one row per seed, with confusion matrix counts
    poc_predictions.csv      every test prediction (probability and label)
    poc_run_info.json        split statistics, extraction timing and device
"""

import os

# Avoid OpenMP/threading crashes on macOS (XGBoost / sklearn)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# PyDriller forks git subprocesses after the tokenizer was used; silence the fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from data_loader import chronological_split, describe_split, load_camel_commits
from embeddings import attach_embeddings, pick_device
from train_eval import bootstrap_ci, run_experiments, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sample", type=int, default=config.TRAIN_SAMPLE)
    ap.add_argument("--test-sample", type=int, default=config.TEST_SAMPLE)
    ap.add_argument("--smoke", action="store_true", help="tiny run into results/smoke (not for reporting)")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    args = ap.parse_args()

    tag, out_dir = "", config.RESULTS_DIR
    if args.smoke:
        args.train_sample, args.test_sample = 200, 100
        tag, out_dir = "_smoke", config.RESULTS_DIR / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    info: dict = {"device": str(device), "smoke": args.smoke}

    print("1. Loading Apache Camel commits...")
    df = load_camel_commits()
    print(f"   {len(df)} commits, {df['buggy'].mean():.1%} buggy")

    print("2. Chronological split 60/20/20 (validation is reserved for tuning, not used here)...")
    train, val, test = chronological_split(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        line = describe_split(name, part)
        print("   " + line)
        info[f"split_{name}"] = line

    print("3. Mining diffs and extracting CodeBERT embeddings...")
    train, train_stats = attach_embeddings(train, "train", device, args.train_sample, tag)
    test, test_stats = attach_embeddings(test, "test", device, args.test_sample, tag)
    info["extraction_train"], info["extraction_test"] = train_stats, test_stats
    info["train_used"] = {"n": len(train), "buggy": int(train["buggy"].sum())}
    info["test_used"] = {"n": len(test), "buggy": int(test["buggy"].sum())}

    print("4. Training and evaluating (3 classifiers x 3 configurations x 3 seeds)...")
    by_seed, preds = run_experiments(train, test)

    print("5. Bootstrap confidence intervals...")
    summary = summarize(by_seed).merge(bootstrap_ci(preds), on=["classifier", "config"])

    summary.to_csv(out_dir / "poc_results.csv", index=False)
    by_seed.to_csv(out_dir / "poc_results_by_seed.csv", index=False)
    preds.to_csv(out_dir / "poc_predictions.csv", index=False)
    (out_dir / "poc_run_info.json").write_text(json.dumps(info, indent=2))

    show = ["classifier", "config", "f1", "auc_roc", "g_mean", "precision", "recall", "mcc"]
    print(f"\nDone. Results saved to {out_dir}")
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
