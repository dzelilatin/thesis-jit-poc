"""Why are precision and F1 low while AUC-ROC is high? Diagnostics from saved test predictions.

Uses results/poc_predictions.csv only (no retraining). Everything is computed per seed and then
averaged over the three seeds, like the headline table.

Outputs
    results/poc_diagnostics.csv           one row per (classifier, configuration)
    results/poc_prevalence_projection.csv precision / F1 of the SAME predictions at other prevalences

Columns of poc_diagnostics.csv
    prevalence            share of defect-inducing commits in the test period
    auc_roc, avg_precision  threshold-free ranking quality (average precision is the PR-curve area;
                          a random ranking scores `prevalence`)
    tpr, fpr              at the fixed 0.5 threshold; these two do not depend on prevalence
    precision, recall, f1 at 0.5
    best_f1_*             best threshold chosen ON THE TEST SET: an upper bound for diagnosis only,
                          not a reportable result (a reportable threshold must come from validation data)
    recall_top5/10/20     share of all defect-inducing commits found among the top-k% highest-scored commits
    lift_top5/10/20       precision among the top-k% divided by prevalence (1.0 = no better than random)

Projection: precision(pi) = tpr*pi / (tpr*pi + fpr*(1-pi)); recall does not change. The projection is
checked against an empirical resampling of the test negatives to the target prevalence.

Usage:
    python src/diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import BOOTSTRAP_SEED, RESULTS_DIR

TOP_K = (5, 10, 20)
PREVALENCES = (0.035, 0.10, 0.181, 0.265, 0.50)   # test period, mixed, Camel train period, ApacheJIT overall, balanced
RESAMPLES = 300


def counts(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    path = RESULTS_DIR / "poc_predictions.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Run run_poc.py first.")
    preds = pd.read_csv(path)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    thresholds = np.round(np.arange(0.02, 0.99, 0.01), 2)

    rows, proj_rows = [], []
    for (clf, cfg), g in preds.groupby(["classifier", "config"], sort=False):
        seeds = sorted(g["seed"].unique())
        y = g[g["seed"] == seeds[0]]["y_true"].values
        probs = [g[g["seed"] == s]["prob"].values for s in seeds]
        n, n_pos = len(y), int(y.sum())
        prev = n_pos / n

        auc = np.mean([roc_auc_score(y, p) for p in probs])
        ap = np.mean([average_precision_score(y, p) for p in probs])

        # fixed threshold 0.5
        c = np.mean([counts(y, (p >= 0.5).astype(int)) for p in probs], axis=0)   # tp, fp, fn, tn
        tp, fp, fn, tn = c
        p05, r05, f05 = prf(tp, fp, fn)
        tpr, fpr = tp / (tp + fn), fp / (fp + tn)

        # best threshold chosen on the test set (diagnostic upper bound)
        f_by_t = []
        for t in thresholds:
            cs = np.mean([counts(y, (p >= t).astype(int)) for p in probs], axis=0)
            f_by_t.append(prf(cs[0], cs[1], cs[2]))
        best = int(np.argmax([f for _, _, f in f_by_t]))
        bp, br, bf = f_by_t[best]

        # top-k% flagged commits
        topk = {}
        for k in TOP_K:
            m = int(round(n * k / 100))
            rec, prec = [], []
            for p in probs:
                order = np.argsort(-p, kind="stable")[:m]
                hit = y[order].sum()
                rec.append(hit / n_pos)
                prec.append(hit / m)
            topk[k] = (float(np.mean(rec)), float(np.mean(prec)) / prev)

        row = {
            "classifier": clf, "config": cfg, "test_commits": n, "test_buggy": n_pos,
            "prevalence": round(prev, 4), "auc_roc": round(auc, 4), "avg_precision": round(ap, 4),
            "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "precision": round(p05, 4), "recall": round(r05, 4), "f1": round(f05, 4),
            "flagged_share": round((tp + fp) / n, 4),
            "best_f1_threshold": float(thresholds[best]), "best_f1_precision": round(bp, 4),
            "best_f1_recall": round(br, 4), "best_f1": round(bf, 4),
        }
        for k in TOP_K:
            row[f"recall_top{k}"], row[f"lift_top{k}"] = round(topk[k][0], 4), round(topk[k][1], 2)
        rows.append(row)

        # same predictions, other prevalences
        for pi in PREVALENCES:
            p_pi = tpr * pi / (tpr * pi + fpr * (1 - pi))
            f_pi = 2 * p_pi * tpr / (p_pi + tpr)
            rec = {"classifier": clf, "config": cfg, "prevalence": pi,
                   "precision_formula": round(p_pi, 4), "recall": round(tpr, 4), "f1_formula": round(f_pi, 4)}
            if pi > prev:   # empirical check: keep all positives, subsample the negatives
                n_neg_keep = int(round(n_pos * (1 - pi) / pi))
                neg_idx, pos_idx = np.where(y == 0)[0], np.where(y == 1)[0]
                ps, fs = [], []
                for _ in range(RESAMPLES):
                    idx = np.concatenate([pos_idx, rng.choice(neg_idx, n_neg_keep, replace=False)])
                    per = [prf(*counts(y[idx], (p[idx] >= 0.5).astype(int))[:3]) for p in probs]
                    ps.append(np.mean([a for a, _, _ in per]))
                    fs.append(np.mean([f for _, _, f in per]))
                rec["precision_resampled"], rec["f1_resampled"] = round(float(np.mean(ps)), 4), round(float(np.mean(fs)), 4)
            proj_rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "poc_diagnostics.csv", index=False)
    pd.DataFrame(proj_rows).to_csv(RESULTS_DIR / "poc_prevalence_projection.csv", index=False)

    show = ["classifier", "config", "auc_roc", "avg_precision", "prevalence", "precision", "recall", "f1",
            "best_f1_threshold", "best_f1", "recall_top10", "lift_top10", "recall_top20", "lift_top20"]
    print(out[show].to_string(index=False))
    print(f"\n-> {RESULTS_DIR / 'poc_diagnostics.csv'}\n-> {RESULTS_DIR / 'poc_prevalence_projection.csv'}")


if __name__ == "__main__":
    main()
