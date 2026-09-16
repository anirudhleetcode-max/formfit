"""Metrics used for every form-classifier run.

Conventions
- The model outputs p = P(clean rep).
- For detection metrics the POSITIVE class is "faulty" (that is what the coach must catch), so
  F1 / precision / recall / PR-AUC are reported for faulty reps, with score = 1 - p.
- Calibration (Brier, ECE, reliability table) is reported for p = P(clean).
- Thresholds are chosen on the validation split only, then applied unchanged to test.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def reliability(y_clean: np.ndarray, p_clean: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error (equal-width bins) and the per-bin table."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_clean, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p_clean[m].mean()), float(y_clean[m].mean())
        ece += m.mean() * abs(conf - acc)
        table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                      "mean_predicted": round(conf, 3), "observed_clean_rate": round(acc, 3)})
    return round(float(ece), 4), table


def pick_threshold(y_clean: np.ndarray, p_clean: np.ndarray) -> float:
    """Threshold on P(clean) that maximises F1 of the faulty class (validation split only)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(1 - y_clean, (p_clean < t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return round(best_t, 2)


def pick_abstain_margin(y_clean: np.ndarray, p_clean: np.ndarray, threshold: float,
                        target_acc: float = 0.95) -> float:
    """Smallest confidence level c such that predictions with max(p, 1-p) >= c reach target accuracy
    on the validation split. Predictions below c are reported as 'uncertain'."""
    pred = (p_clean >= threshold).astype(int)
    conf = np.maximum(p_clean, 1 - p_clean)
    for c in np.linspace(0.5, 0.99, 50):
        m = conf >= c
        if m.sum() >= 20 and (pred[m] == y_clean[m]).mean() >= target_acc:
            return round(float(c), 2)
    return 0.99


def classification_report(y_clean: np.ndarray, p_clean: np.ndarray | None, pred_clean: np.ndarray,
                          abstain_conf: float | None = None) -> dict:
    y_fault, pred_fault = 1 - y_clean, 1 - pred_clean
    cm = confusion_matrix(y_fault, pred_fault, labels=[0, 1])
    out = {
        "n": int(len(y_clean)),
        "accuracy": round(float((pred_clean == y_clean).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_clean, pred_clean)), 4),
        "faulty": {
            "precision": round(float(precision_score(y_fault, pred_fault, zero_division=0)), 4),
            "recall": round(float(recall_score(y_fault, pred_fault, zero_division=0)), 4),
            "f1": round(float(f1_score(y_fault, pred_fault, zero_division=0)), 4),
            "support": int(y_fault.sum()),
        },
        "clean": {
            "precision": round(float(precision_score(y_clean, pred_clean, zero_division=0)), 4),
            "recall": round(float(recall_score(y_clean, pred_clean, zero_division=0)), 4),
            "f1": round(float(f1_score(y_clean, pred_clean, zero_division=0)), 4),
            "support": int(y_clean.sum()),
        },
        # rows = true (clean, faulty), cols = predicted (clean, faulty)
        "confusion_matrix": {"labels": ["clean", "faulty"], "matrix": cm.tolist()},
    }
    if p_clean is not None and len(np.unique(y_clean)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_fault, 1 - p_clean)), 4)
        out["pr_auc_faulty"] = round(float(average_precision_score(y_fault, 1 - p_clean)), 4)
        out["brier"] = round(float(brier_score_loss(y_clean, p_clean)), 4)
        out["ece"], out["reliability"] = reliability(y_clean, p_clean)
        if abstain_conf is not None:
            conf = np.maximum(p_clean, 1 - p_clean)
            m = conf >= abstain_conf
            out["selective"] = {
                "min_confidence": abstain_conf,
                "coverage": round(float(m.mean()), 4),
                "accuracy_on_covered": round(float((pred_clean[m] == y_clean[m]).mean()), 4) if m.any() else None,
            }
    return out
