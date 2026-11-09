"""Official ApacheJIT split, Camel commits only: balanced training set, natural test sets.

The ApacheJIT release ships three files (data/apachejit/dataset/):
    apachejit_train.csv       balanced by undersampling (all projects, commits up to 2016)
    apachejit_test_large.csv  the last three years, natural class distribution
    apachejit_test_small.csv  a subset of test_large

This script keeps the Camel rows of each file, so it answers the question "what changes if we follow the
official protocol (a BALANCED training set, a natural test set) instead of our own chronological 60/20/20
split?". Embeddings come from the PoC cache (the three Camel splits together cover all 22,700 commits), so no
new extraction is needed. Training: same classifiers and settings as the PoC (class weights from the training
set, threshold 0.5, seeds 42/123/456). Besides the natural test set, precision and F1 are also reported on test
subsets with a higher share of defect-inducing commits (26.5% = ApacheJIT overall, 50% = balanced); these use the
SAME predictions and only change the class mix of the evaluation.

Output: results/official_split_results.csv

Usage:
    python src/official_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DATA_DIR, LABEL_COL, RESULTS_DIR, SEEDS
from data_loader import chronological_split, load_camel_commits
from embeddings import attach_embeddings
from train_eval import all_metrics, make_classifier, raw_features

DATASET_DIR = DATA_DIR / "apachejit" / "dataset"
TEST_FILES = {"test_large": "apachejit_test_large.csv", "test_small": "apachejit_test_small.csv"}
CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
CONFIGS = ("metrics_only", "hybrid")
PREVALENCES = (0.265, 0.5)
RESAMPLES = 200


def camel_ids(filename):
    x = pd.read_csv(DATASET_DIR / filename)
    return x[x["project"].str.lower() == "apache/camel"]["commit_id"].tolist()


def balanced_sets(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {}
    for pi in PREVALENCES:
        k = int(round(len(pos) * (1 - pi) / pi))
        out[pi] = [np.concatenate([pos, rng.choice(neg, k, replace=False)]) for _ in range(RESAMPLES)]
    return out


def prf(y, pred):
    tp, fp, fn = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    cpu = torch.device("cpu")   # cache hits only; the encoder is never loaded
    parts = [attach_embeddings(part, name, cpu)[0] for name, part in zip(("train", "val", "test"), chronological_split(load_camel_commits()))]
    emb = pd.concat(parts).drop_duplicates("commit_id").set_index("commit_id")
    assert len(emb) == 22700, len(emb)

    train = emb.loc[camel_ids("apachejit_train.csv")].reset_index()
    y_train = train[LABEL_COL].astype(int).values
    pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"official train (Camel): {len(train)} commits, {y_train.mean():.1%} defect-inducing, pos_weight {pos_weight:.2f}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for tname, fname in TEST_FILES.items():
        test = emb.loc[camel_ids(fname)].reset_index()
        y = test[LABEL_COL].astype(int).values
        print(f"{tname}: {len(test)} commits, {int(y.sum())} defect-inducing ({y.mean():.1%})")
        bal = balanced_sets(y, rng)
        boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP_RESAMPLES)]
        for clf in CLASSIFIERS:
            for config in CONFIGS:
                scaler = StandardScaler().fit(raw_features(train, config))
                X_tr, X_te = scaler.transform(raw_features(train, config)), scaler.transform(raw_features(test, config))
                probs = []
                for seed in SEEDS:
                    model = make_classifier(clf, seed, pos_weight)
                    model.fit(X_tr, y_train)
                    probs.append(model.predict_proba(X_te)[:, 1])
                per_seed = [all_metrics(y, (p >= 0.5).astype(int), p) for p in probs]
                m = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
                m["avg_precision"] = float(np.mean([average_precision_score(y, p) for p in probs]))

                row = {"test_set": tname, "classifier": clf, "config": config, "train_commits": len(train),
                       "train_prevalence": round(float(y_train.mean()), 4), "test_commits": len(y),
                       "test_buggy": int(y.sum()), "test_prevalence": round(float(y.mean()), 4),
                       **{k: round(v, 4) for k, v in m.items()}}
                for pi, sets in bal.items():
                    ps, fs = zip(*[np.mean([prf(y[i], p[i] >= 0.5) for p in probs], axis=0) for i in sets])
                    row[f"precision_{pi * 100:g}pct"], row[f"f1_{pi * 100:g}pct"] = round(float(np.mean(ps)), 4), round(float(np.mean(fs)), 4)
                # bootstrap interval of F1 / AUC / AP on the natural test set (seed-mean metrics)
                f1s, aucs, aps = [], [], []
                for idx in boot_idx:
                    f1s.append(np.mean([prf(y[idx], p[idx] >= 0.5)[1] for p in probs]))
                    aucs.append(np.mean([roc_auc_score(y[idx], p[idx]) for p in probs]))
                    aps.append(np.mean([average_precision_score(y[idx], p[idx]) for p in probs]))
                for name, vals in (("f1", f1s), ("auc_roc", aucs), ("avg_precision", aps)):
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    row[f"{name}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                rows.append(row)
                print(f"  {tname} {clf:20s} {config:12s} F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} AUC {m['auc_roc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "official_split_results.csv", index=False)
    print(f"\n-> {RESULTS_DIR / 'official_split_results.csv'}")


if __name__ == "__main__":
    main()
