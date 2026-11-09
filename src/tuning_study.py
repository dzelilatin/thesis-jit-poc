"""Can the approach (threshold, class balancing, recency, tuning) improve on the PoC baseline?

Scenario study on the cached CodeBERT embeddings of the Camel PoC. Everything is selected on the
VALIDATION split (10.1% defect-inducing); the TEST split (3.5%) is only used to report results.

For every classifier (logistic_regression, xgboost) and configuration (metrics_only, hybrid):

    S0  baseline of the PoC (class weights, default hyperparameters, threshold 0.5)
    S1  baseline model, threshold chosen on validation (maximum F1, or maximum F2 = recall-oriented)
    S2  class balancing of the training data: none, random undersampling (1:1), SMOTE (minority = 50%)
    S3  Optuna tuning of the hyperparameters (including class weight), objective = validation average precision
    S4  training on the most recent 50% / 25% of the training commits only (prevalence drift)
    S5  S3 on the recency window with the best validation average precision

Each fitted model is reported under three threshold rules: 0.5, validation-F1, validation-F2.
The model is trained on the training split only (no refit on train+validation) with seed 42.

Outputs
    results/tuning_study.csv            one row per scenario x classifier x configuration x threshold rule
    results/tuning_study_params.json    tuned hyperparameters
    results/tuning_study_paired.csv     paired bootstrap differences on the test set

Usage:
    python src/tuning_study.py
    python src/tuning_study.py --trials 10 --classifiers xgboost --configs hybrid     # quicker
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, LABEL_COL, RESULTS_DIR
from data_loader import chronological_split, load_camel_commits
from embeddings import attach_embeddings
from train_eval import raw_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
WINDOWS = (1.0, 0.5, 0.25)
RULES = ("0.5", "val_F1", "val_F2")
BALANCED_PREVALENCES = (0.265, 0.5)   # ApacheJIT overall, balanced
BALANCED_RESAMPLES = 200
BAL_SETS: dict = {}                    # prevalence -> list of test-set index arrays (filled in run())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_splits():
    train, val, test = chronological_split(load_camel_commits())
    cpu = torch.device("cpu")   # embeddings come from the cache; the model is never loaded
    train, _ = attach_embeddings(train, "train", cpu)
    val, _ = attach_embeddings(val, "val", cpu)
    test, _ = attach_embeddings(test, "test", cpu)
    return train, val, test


def features(train, val, test, config, window=1.0):
    """Scaled features; the scaler is fitted on the (windowed) training commits only."""
    tr = train.iloc[int(len(train) * (1 - window)):]
    scaler = StandardScaler().fit(raw_features(tr, config))
    return (scaler.transform(raw_features(tr, config)), tr[LABEL_COL].astype(int).values,
            scaler.transform(raw_features(val, config)), scaler.transform(raw_features(test, config)))


# ---------------------------------------------------------------------------
# Models, thresholds, metrics
# ---------------------------------------------------------------------------

def make_model(clf, params, y):
    neg_pos = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    if clf == "logistic_regression":
        return LogisticRegression(max_iter=2000, C=params.get("C", 1.0),
                                  class_weight=params.get("class_weight", "balanced"), random_state=SEED)
    rest = {k: v for k, v in params.items() if k != "scale_pos_weight"}
    rest.setdefault("n_estimators", 200)
    return XGBClassifier(scale_pos_weight=params.get("scale_pos_weight", neg_pos), eval_metric="logloss",
                         verbosity=0, random_state=SEED, n_jobs=1, tree_method="hist", **rest)


UNWEIGHTED = {"logistic_regression": {"class_weight": None}, "xgboost": {"scale_pos_weight": 1.0}}


def fit_predict(clf, params, X, y, targets):
    model = make_model(clf, params, y)
    model.fit(X, y)
    return [model.predict_proba(t)[:, 1] for t in targets]


def pick_threshold(y, p, beta):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-12)
    return float(thr[int(np.argmax(f))])


def evaluate(y, p, thr):
    pred = p >= thr
    pos = y == 1
    tp, fp, fn = int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    order = np.argsort(-p, kind="stable")
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "flagged_share": float(pred.mean()), "auc_roc": float(roc_auc_score(y, p)),
           "avg_precision": float(average_precision_score(y, p))}
    for k in (10, 20):
        out[f"recall_top{k}"] = float(y[order[: int(round(len(y) * k / 100))]].sum() / pos.sum())
    return out


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def make_balanced_sets(y):
    """Test subsets with all defect-inducing commits and enough random clean commits for the target prevalence."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    sets = {}
    for pi in BALANCED_PREVALENCES:
        k = int(round(len(pos) * (1 - pi) / pi))
        sets[pi] = [np.concatenate([pos, rng.choice(neg, k, replace=False)]) for _ in range(BALANCED_RESAMPLES)]
    return sets


def balanced_eval(y, p, thr):
    """Precision / F1 of the SAME predictions on test subsets with a higher share of defect-inducing commits."""
    pred, out = p >= thr, {}
    for pi, sets in BAL_SETS.items():
        precs, f1s = [], []
        for idx in sets:
            yy, pp = y[idx] == 1, pred[idx]
            tp, fp, fn = int((pp & yy).sum()), int((pp & ~yy).sum()), int((~pp & yy).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            precs.append(prec)
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out[f"precision_{pi * 100:g}pct"], out[f"f1_{pi * 100:g}pct"] = float(np.mean(precs)), float(np.mean(f1s))
    return out


def rows_for(scenario, clf, config, p_val, p_test, y_val, y_test, store, extra=""):
    thresholds = {"0.5": 0.5, "val_F1": pick_threshold(y_val, p_val, 1.0), "val_F2": pick_threshold(y_val, p_val, 2.0)}
    val_ap = float(average_precision_score(y_val, p_val))
    rows = []
    for rule in RULES:
        m = evaluate(y_test, p_test, thresholds[rule])
        m.update(balanced_eval(y_test, p_test, thresholds[rule]))
        rows.append({"scenario": scenario, "detail": extra, "classifier": clf, "config": config, "rule": rule,
                     "threshold": round(thresholds[rule], 4), "val_avg_precision": round(val_ap, 4),
                     **{k: round(v, 4) for k, v in m.items()}})
        store[(scenario, clf, config, rule)] = (p_test, thresholds[rule])
    return rows


def tune(clf, X, y, X_val, y_val, trials):
    neg_pos = float((y == 0).sum()) / max(int((y == 1).sum()), 1)

    def objective(trial):
        if clf == "logistic_regression":
            params = {"C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                      "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])}
        else:
            params = {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
                      "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                      "max_depth": trial.suggest_int("max_depth", 3, 8),
                      "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                      "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                      "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 2 * neg_pos, log=True)}
        (p,) = fit_predict(clf, params, X, y, [X_val])
        return average_precision_score(y_val, p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_value


def run(args):
    train, val, test = load_splits()
    y_val, y_test = val[LABEL_COL].astype(int).values, test[LABEL_COL].astype(int).values
    print(f"train {len(train)} | val {len(val)} ({y_val.mean():.1%} buggy) | test {len(test)} ({y_test.mean():.1%} buggy)", flush=True)

    BAL_SETS.update(make_balanced_sets(y_test))
    rows, store, tuned = [], {}, {}
    for clf in args.classifiers:
        for config in args.configs:
            t0 = time.time()
            tag = f"{clf}/{config}"
            X, y, Xv, Xt = features(train, val, test, config)
            base = {"n_estimators": 200} if clf == "xgboost" else {}

            # S0 / S1: baseline model, three threshold rules
            pv, pt = fit_predict(clf, base, X, y, [Xv, Xt])
            rows += rows_for("S0/S1 baseline", clf, config, pv, pt, y_val, y_test, store)
            base_val_ap = average_precision_score(y_val, pv)

            # S2: balancing of the training data
            for name in ("none", "under", "smote"):
                Xb, yb = X, y
                if name == "under":
                    Xb, yb = RandomUnderSampler(sampling_strategy=1.0, random_state=SEED).fit_resample(X, y)
                if name == "smote":
                    Xb, yb = SMOTE(sampling_strategy=0.5, random_state=SEED).fit_resample(X, y)
                params = {**base, **UNWEIGHTED[clf]}
                pv, pt = fit_predict(clf, params, Xb, yb, [Xv, Xt])
                rows += rows_for(f"S2 balancing: {name}", clf, config, pv, pt, y_val, y_test, store)

            # S4: recency windows (baseline model)
            window_ap = {1.0: base_val_ap}
            for w in WINDOWS[1:]:
                Xw, yw, Xvw, Xtw = features(train, val, test, config, w)
                pv, pt = fit_predict(clf, base, Xw, yw, [Xvw, Xtw])
                window_ap[w] = average_precision_score(y_val, pv)
                rows += rows_for(f"S4 recency: last {int(w * 100)}%", clf, config, pv, pt, y_val, y_test, store)

            # S3: tuning on the full training period
            best, val_ap = tune(clf, X, y, Xv, y_val, args.trials)
            pv, pt = fit_predict(clf, best, X, y, [Xv, Xt])
            rows += rows_for("S3 tuned", clf, config, pv, pt, y_val, y_test, store, extra=f"{args.trials} trials")
            tuned[f"{tag} S3"] = {**best, "val_avg_precision": round(val_ap, 4)}

            # S5: tuning on the recency window with the best validation average precision
            w_best = max(window_ap, key=window_ap.get)
            if w_best == 1.0:
                rows += [{**r, "scenario": "S5 tuned + best window", "detail": "window 100% (= S3)"}
                         for r in rows[-3:]]
                store.update({("S5 tuned + best window", clf, config, rule): store[("S3 tuned", clf, config, rule)] for rule in RULES})
            else:
                Xw, yw, Xvw, Xtw = features(train, val, test, config, w_best)
                best5, val_ap5 = tune(clf, Xw, yw, Xvw, y_val, args.trials)
                pv, pt = fit_predict(clf, best5, Xw, yw, [Xvw, Xtw])
                rows += rows_for("S5 tuned + best window", clf, config, pv, pt, y_val, y_test, store,
                                 extra=f"window {int(w_best * 100)}%")
                tuned[f"{tag} S5"] = {**best5, "window": w_best, "val_avg_precision": round(val_ap5, 4)}
            print(f"[{tag}] done in {time.time() - t0:.0f}s (best window {int(w_best * 100)}%)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "tuning_study.csv", index=False)
    (RESULTS_DIR / "tuning_study_params.json").write_text(json.dumps(tuned, indent=2, default=float))
    paired(store, y_test, args)
    summarize(out)


# ---------------------------------------------------------------------------
# Paired bootstrap on the test set
# ---------------------------------------------------------------------------

def paired(store, y_test, args):
    comparisons = []
    for clf in args.classifiers:
        if "hybrid" in args.configs:
            comparisons.append((clf, "tuned hybrid vs PoC baseline (0.5)",
                                ("S5 tuned + best window", clf, "hybrid", "val_F1"), ("S0/S1 baseline", clf, "hybrid", "0.5")))
        if "hybrid" in args.configs and "metrics_only" in args.configs:
            comparisons.append((clf, "hybrid vs metrics, both tuned",
                                ("S5 tuned + best window", clf, "hybrid", "val_F1"), ("S5 tuned + best window", clf, "metrics_only", "val_F1")))
            comparisons.append((clf, "hybrid vs metrics, both untuned (val threshold)",
                                ("S0/S1 baseline", clf, "hybrid", "val_F1"), ("S0/S1 baseline", clf, "metrics_only", "val_F1")))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for clf, label, ka, kb in comparisons:
        (pa, ta), (pb, tb) = store[ka], store[kb]
        diffs = {m: [] for m in ("f1", "recall", "avg_precision", "recall_top20")}
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = rng.integers(0, len(y_test), len(y_test))
            ma, mb = evaluate(y_test[idx], pa[idx], ta), evaluate(y_test[idx], pb[idx], tb)
            for m in diffs:
                diffs[m].append(ma[m] - mb[m])
        pt_a, pt_b = evaluate(y_test, pa, ta), evaluate(y_test, pb, tb)
        for m, vals in diffs.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({"classifier": clf, "comparison": label, "metric": m,
                         "difference": round(pt_a[m] - pt_b[m], 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuning_study_paired.csv", index=False)


def summarize(out):
    cols = ["scenario", "classifier", "config", "rule", "precision", "recall", "f1", "avg_precision", "recall_top20"]
    for rule in ("0.5", "val_F1", "val_F2"):
        print(f"\n=== threshold rule: {rule} ===")
        print(out[out["rule"] == rule][cols].drop(columns="rule").to_string(index=False))
    print(f"\n-> {RESULTS_DIR / 'tuning_study.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--classifiers", nargs="+", default=["logistic_regression", "xgboost"])
    ap.add_argument("--configs", nargs="+", default=["metrics_only", "hybrid"])
    run(ap.parse_args())
