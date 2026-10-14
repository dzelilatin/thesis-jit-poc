"""Train classifiers and evaluate on test set."""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import CLASSIFIERS, CONFIGS, METRIC_COLS, RANDOM_SEED


def g_mean(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) else 0
    tnr = tn / (tn + fp) if (tn + fp) else 0
    return float(np.sqrt(tpr * tnr))


def build_features(df: pd.DataFrame, config: str, scaler: StandardScaler | None = None):
    metrics = df[METRIC_COLS].values.astype(float)
    if config == "metrics_only":
        if scaler is None:
            scaler = StandardScaler().fit(metrics)
        return scaler.transform(metrics), scaler

    emb = np.vstack(df["embedding"].values)
    if config == "embeddings_only":
        return emb, None

    if scaler is None:
        scaler = StandardScaler().fit(metrics)
    m = scaler.transform(metrics)
    return np.hstack([m, emb]), scaler


def make_classifier(name: str, scale_pos_weight: float):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1
        )
    return XGBClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
    )


def _safe_auc(y_true, prob) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def run_experiments(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y_train = train["buggy"].astype(int).values
    y_val = val["buggy"].astype(int).values
    y_test = test["buggy"].astype(int).values
    neg, pos = (y_train == 0).sum(), max((y_train == 1).sum(), 1)
    scale_pos_weight = neg / pos

    rows: list[dict[str, Any]] = []

    for clf_name in CLASSIFIERS:
        for config in CONFIGS:
            t0 = time.time()
            x_train, scaler = build_features(train, config)
            x_val, _ = build_features(val, config, scaler)
            x_test, _ = build_features(test, config, scaler)

            clf = make_classifier(clf_name, scale_pos_weight)
            clf.fit(x_train, y_train)

            # simple val pick for threshold-free metrics; use default 0.5
            prob = clf.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            rows.append(
                {
                    "classifier": clf_name,
                    "config": config,
                    "f1": round(f1_score(y_test, pred, zero_division=0), 4),
                    "auc_roc": round(_safe_auc(y_test, prob), 4),
                    "g_mean": round(g_mean(y_test, pred), 4),
                    "wall_clock_sec": round(time.time() - t0, 2),
                    "train_size": len(train),
                    "test_size": len(test),
                }
            )
            print(f"  {clf_name} + {config}: F1={rows[-1]['f1']}, AUC={rows[-1]['auc_roc']}")

    return pd.DataFrame(rows)
