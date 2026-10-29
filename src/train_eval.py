"""Train classifiers, evaluate on the held-out test set, bootstrap confidence intervals.

Mirrors the paper pipeline (jit-llm-embeddings, src/train.py): the same three feature
configurations, z-score standardization fitted on the training split only (for the
embeddings too), three seeds, a fixed 0.5 decision threshold, and the same metric set
(F1, AUC-ROC, G-Mean, Precision, Recall, MCC). In addition the confusion matrix and
every test prediction are kept, so any metric can be recomputed later.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CLASSIFIERS, COMMIT_COL, CONFIGS, LABEL_COL,
    METRIC_COLS, SEEDS, THRESHOLD,
)

METRIC_NAMES = ["f1", "auc_roc", "g_mean", "precision", "recall", "mcc"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    f1_den = 2 * tp + fp + fn
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "f1": 2 * tp / f1_den if f1_den else 0.0,
        "g_mean": float(np.sqrt(tpr * tnr)),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tpr,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
    }


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = metrics_from_counts(*confusion(y_true, y_pred))
    out["auc_roc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Features and models
# ---------------------------------------------------------------------------

def raw_features(df: pd.DataFrame, config: str) -> np.ndarray:
    metrics = df[METRIC_COLS].values.astype(float)
    if config == "metrics_only":
        return metrics
    emb = np.vstack(df["embedding"].values).astype(float)
    if config == "embeddings_only":
        return emb
    return np.hstack([metrics, emb])  # hybrid: metrics first, as in the paper


def build_features(train: pd.DataFrame, test: pd.DataFrame, config: str):
    """Standardize with training-set statistics, for all three configurations."""
    scaler = StandardScaler().fit(raw_features(train, config))
    return scaler.transform(raw_features(train, config)), scaler.transform(raw_features(test, config))


def make_classifier(name: str, seed: int, pos_weight: float):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1
        )
    return XGBClassifier(
        scale_pos_weight=pos_weight, eval_metric="logloss", verbosity=0,
        random_state=seed, n_jobs=1, tree_method="hist",
    )


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiments(train: pd.DataFrame, test: pd.DataFrame):
    """Returns (per-seed metric rows, per-commit predictions)."""
    y_train = train[LABEL_COL].astype(int).values
    y_test = test[LABEL_COL].astype(int).values
    n_pos = max(int((y_train == 1).sum()), 1)
    pos_weight = float((y_train == 0).sum()) / n_pos
    print(f"  train: {len(y_train)} commits, {n_pos} buggy, pos_weight={pos_weight:.2f}; "
          f"test: {len(y_test)} commits, {int(y_test.sum())} buggy")

    features = {cfg: build_features(train, test, cfg) for cfg in CONFIGS}
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []

    for clf_name in CLASSIFIERS:
        for config in CONFIGS:
            x_train, x_test = features[config]
            for seed in SEEDS:
                t0 = time.time()
                clf = make_classifier(clf_name, seed, pos_weight)
                clf.fit(x_train, y_train)
                prob = clf.predict_proba(x_test)[:, 1]
                pred = (prob >= THRESHOLD).astype(int)
                seconds = time.time() - t0

                tn, fp, fn, tp = confusion(y_test, pred)
                rows.append({
                    "classifier": clf_name, "config": config, "seed": seed,
                    **{k: round(v, 4) for k, v in all_metrics(y_test, pred, prob).items()},
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "wall_clock_sec": round(seconds, 2),
                    "train_size": len(y_train), "test_size": len(y_test),
                    "test_buggy": int(y_test.sum()),
                })
                preds.append(pd.DataFrame({
                    "classifier": clf_name, "config": config, "seed": seed,
                    COMMIT_COL: test[COMMIT_COL].values, "y_true": y_test,
                    "prob": prob.round(5), "pred": pred,
                }))
            last = rows[-1]
            print(f"  {clf_name:20s} {config:16s} F1={last['f1']:.3f} AUC={last['auc_roc']:.3f} "
                  f"P={last['precision']:.3f} R={last['recall']:.3f} (seed {last['seed']})")

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (classifier, config); confusion counts are seed means too."""
    keep = METRIC_NAMES + ["tn", "fp", "fn", "tp", "wall_clock_sec"]
    out = by_seed.groupby(["classifier", "config"], sort=False)[keep].mean().reset_index()
    for col in ("train_size", "test_size", "test_buggy"):
        out[col] = int(by_seed[col].iloc[0])
    return out.round(4)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(preds: pd.DataFrame, n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """95% percentile intervals over resampled test commits.

    One resample of the test commits is evaluated for every seed and the metric is
    averaged over seeds, so the interval refers to the same quantity as the seed-mean
    point estimate in the results table.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for (clf, cfg), g in preds.groupby(["classifier", "config"], sort=False):
        seeds = sorted(g["seed"].unique())
        probs = np.vstack([g[g["seed"] == s]["prob"].values for s in seeds])
        pds = np.vstack([g[g["seed"] == s]["pred"].values for s in seeds])
        y = g[g["seed"] == seeds[0]]["y_true"].values
        groups[(clf, cfg)] = (y, probs, pds)

    n = len(next(iter(groups.values()))[0])
    samples = {k: {m: [] for m in METRIC_NAMES} for k in groups}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        for key, (y, probs, pds) in groups.items():
            yb = y[idx]
            per_seed = [all_metrics(yb, pb[idx], prb[idx]) for pb, prb in zip(pds, probs)]
            for m in METRIC_NAMES:
                with warnings.catch_warnings():  # AUC is NaN if a resample has no positives
                    warnings.simplefilter("ignore", RuntimeWarning)
                    samples[key][m].append(np.nanmean([d[m] for d in per_seed]))

    rows = []
    for (clf, cfg), metric_samples in samples.items():
        row = {"classifier": clf, "config": cfg}
        for m, vals in metric_samples.items():
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = round(float(lo), 4), round(float(hi), 4)
        rows.append(row)
    return pd.DataFrame(rows)
