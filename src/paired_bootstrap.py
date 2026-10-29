"""Paired bootstrap of metric differences between feature configurations.

For every classifier the same resample of test commits is evaluated under both
configurations (and for every seed; the metric is averaged over seeds, as in the
point estimates of poc_results.csv). The difference is reported with a 95%
percentile interval and the share of resamples in which the first configuration
is not better, a one-sided bootstrap p-value (no multiple-comparison correction).

Comparisons: hybrid - metrics_only, embeddings_only - metrics_only, hybrid - embeddings_only.

Usage:
    python src/paired_bootstrap.py
Output: results/poc_paired_bootstrap.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CLASSIFIERS, RESULTS_DIR
from train_eval import all_metrics

COMPARISONS = [
    ("hybrid", "metrics_only"),
    ("embeddings_only", "metrics_only"),
    ("hybrid", "embeddings_only"),
]
METRICS = ["f1", "auc_roc", "mcc"]


def main() -> None:
    path = RESULTS_DIR / "poc_predictions.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Run run_poc.py first.")
    preds = pd.read_csv(path)

    data = {}
    for (clf, cfg), g in preds.groupby(["classifier", "config"], sort=False):
        seeds = sorted(g["seed"].unique())
        probs = np.vstack([g[g["seed"] == s]["prob"].values for s in seeds])
        pds = np.vstack([g[g["seed"] == s]["pred"].values for s in seeds])
        y = g[g["seed"] == seeds[0]]["y_true"].values
        data[(clf, cfg)] = (y, probs, pds)

    n = len(next(iter(data.values()))[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = {(c, a, b, m): [] for c in CLASSIFIERS for a, b in COMPARISONS for m in METRICS}

    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        cell = {}
        for key, (y, probs, pds) in data.items():
            per_seed = [all_metrics(y[idx], pb[idx], prb[idx]) for pb, prb in zip(pds, probs)]
            cell[key] = {m: np.mean([d[m] for d in per_seed]) for m in METRICS}
        for c in CLASSIFIERS:
            for a, b in COMPARISONS:
                for m in METRICS:
                    diffs[(c, a, b, m)].append(cell[(c, a)][m] - cell[(c, b)][m])

    rows = []
    for (c, a, b, m), vals in diffs.items():
        vals = np.array(vals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        point = (
            preds[(preds.classifier == c) & (preds.config == a)]
            .groupby("seed")
            .apply(lambda g: all_metrics(g["y_true"].values, g["pred"].values, g["prob"].values)[m])
            .mean()
            - preds[(preds.classifier == c) & (preds.config == b)]
            .groupby("seed")
            .apply(lambda g: all_metrics(g["y_true"].values, g["pred"].values, g["prob"].values)[m])
            .mean()
        )
        rows.append({
            "classifier": c, "comparison": f"{a} - {b}", "metric": m,
            "difference": round(float(point), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "p_one_sided": round(float((vals <= 0).mean()), 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "poc_paired_bootstrap.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n-> {RESULTS_DIR / 'poc_paired_bootstrap.csv'}")


if __name__ == "__main__":
    main()
