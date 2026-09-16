"""Train the per-exercise form-quality classifiers on simulated reps.

    python -m ml.train            # writes ml/artifacts/form_model.joblib and ml/artifacts/metrics.json

For each exercise we compare three models on a held-out test set generated with a *different*
random seed: a rule baseline (same thresholds the browser uses), logistic regression and a
gradient-boosted tree ensemble. The best model by ROC-AUC on a validation split is saved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import rules
from .features import EXERCISE_FEATURES, EXERCISES, FEATURES
from .simulate import generate

ART = Path(__file__).resolve().parent / "artifacts"


def _metrics(y, pred, proba):
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1_good": round(float(f1_score(y, pred)), 4),
        "f1_faulty": round(float(f1_score(1 - y, 1 - pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if proba is not None else None,
    }


def train(n_train: int, n_test: int, seed: int = 7) -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    bundle = {"features": list(FEATURES), "columns": {}, "models": {}, "version": time.strftime("%Y%m%d")}
    report = {"n_train": n_train, "n_test": n_test, "exercises": {}}
    for i, ex in enumerate(EXERCISES):
        X, y = generate(ex, n_train, seed=seed + i)
        Xte, yte = generate(ex, n_test, seed=10_000 + seed + i)   # independent test draw
        cols = [FEATURES.index(k) for k in EXERCISE_FEATURES[ex]]
        Xall, Xte_full = X, Xte
        X, Xte = X[:, cols], Xte[:, cols]
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        candidates = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "random_forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=3,
                                                    n_jobs=1, random_state=0),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                            learning_rate=0.1, random_state=0),
        }
        # balanced sample weights: classes are weighted equally, so a score of 50 means "undecided"
        # instead of reflecting how rare clean reps happen to be in the simulator
        def weights(yy):
            return compute_sample_weight("balanced", yy)

        def fit(m, XX, yy):
            key = m.steps[-1][0] if hasattr(m, "steps") else None
            return m.fit(XX, yy, **({f"{key}__sample_weight": weights(yy)} if key else {"sample_weight": weights(yy)}))

        val_auc = {}
        for name, m in candidates.items():
            fit(m, Xtr, ytr)
            val_auc[name] = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        best = max(val_auc, key=val_auc.get)
        model = fit(candidates[best], X, y)  # refit on train+val

        rule_pred = np.array([
            int(not rules.evaluate(ex, dict(zip(FEATURES, row)))[1]) for row in Xte_full
        ])
        rule_score = np.array([rules.evaluate(ex, dict(zip(FEATURES, row)))[0] for row in Xte_full]) / 100
        del Xall
        ex_rep = {
            "good_rate_test": round(float(yte.mean()), 3),
            "selected": best,
            "features": list(EXERCISE_FEATURES[ex]),
            "val_auc": {k: round(float(v), 4) for k, v in val_auc.items()},
            "rule_baseline": _metrics(yte, rule_pred, rule_score),
            "model": _metrics(yte, model.predict(Xte), model.predict_proba(Xte)[:, 1]),
        }
        if hasattr(model, "feature_importances_"):
            imp = sorted(zip(EXERCISE_FEATURES[ex], model.feature_importances_), key=lambda t: -t[1])[:5]
            ex_rep["top_features"] = {k: round(float(v), 3) for k, v in imp}
        report["exercises"][ex] = ex_rep
        bundle["models"][ex] = model
        bundle["columns"][ex] = list(EXERCISE_FEATURES[ex])
        print(f"{ex:8s} best={best:18s} model={ex_rep['model']}  rules={ex_rep['rule_baseline']}")

    joblib.dump(bundle, ART / "form_model.joblib", compress=3)
    (ART / "metrics.json").write_text(json.dumps(report, indent=2))
    print("saved", ART / "form_model.joblib", f"{(ART / 'form_model.joblib').stat().st_size / 1e6:.2f} MB")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    a = ap.parse_args()
    train(a.n_train, a.n_test)
