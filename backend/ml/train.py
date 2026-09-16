"""Train and evaluate the per-exercise form-quality classifiers on SYNTHETIC reps.

    python -m ml.train                         # default config, writes the served artifact
    python -m ml.train --out-dir <dir>         # also writes a full run record there
    (python -m experiments.run --config experiments/configs/form_classifier_synthetic.json does both)

Splits: train / validation / test are three INDEPENDENT simulator draws (different seeds), so no
rep appears in two splits. Model selection, the decision threshold and the abstention margin are
chosen on validation only; test is touched once at the end.

Compared on the same splits:
  majority      always predicts the majority class of the training split
  rules         the browser's rule engine (Python mirror), rule score / 100 as the probability
  logreg        standardised logistic regression
  random_forest
  gradient_boosting
The best learned model by validation ROC-AUC is saved as the served model.

Served artifact: ml/artifacts/form_model.joblib + form_model.card.json (model card) + metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import rules
from .evaluate import classification_report, pick_abstain_margin, pick_threshold
from .features import EXERCISE_FEATURES, EXERCISES, FEATURES, FEATURES_VERSION
from .runinfo import env_info
from .simulate import SIMULATOR_VERSION, dataset_hash, generate

ART = Path(__file__).resolve().parent / "artifacts"

DEFAULT_CONFIG = {
    "name": "form_classifier_synthetic",
    "task": "form_classifier",
    "seed": 7,
    "n_train": 6000,
    "n_val": 2000,
    "n_test": 2000,
    "exercises": list(EXERCISES),
    "selective_target_accuracy": 0.95,
    "ood_quantiles": [0.005, 0.995],
    "ood_margin": 0.10,
    "models": {
        "logreg": {"C": 1.0, "max_iter": 1000},
        "random_forest": {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 3},
        "gradient_boosting": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1},
    },
    "errors_per_side": 15,
}


def _candidates(cfg: dict) -> dict:
    m = cfg["models"]
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(**m["logreg"])),
        "random_forest": RandomForestClassifier(**m["random_forest"], n_jobs=1, random_state=0),
        "gradient_boosting": GradientBoostingClassifier(**m["gradient_boosting"], random_state=0),
    }


def _rule_proba(ex: str, X_full: np.ndarray) -> np.ndarray:
    return np.array([rules.evaluate(ex, dict(zip(FEATURES, row)))[0] for row in X_full]) / 100.0


def _ood_bounds(X: np.ndarray, cols: list[str], q: list[float], margin: float) -> dict:
    lo, hi = np.quantile(X, q[0], axis=0), np.quantile(X, q[1], axis=0)
    span = hi - lo
    return {c: [round(float(a - margin * s), 3), round(float(b + margin * s), 3)] for c, a, b, s in zip(cols, lo, hi, span)}


def _fault_recall(y_faults: list[list[str]], pred_clean: np.ndarray) -> dict:
    """Recall per true fault type: share of reps with that fault that were predicted faulty."""
    out: dict[str, dict] = {}
    for faults, pc in zip(y_faults, pred_clean):
        for f in faults:
            d = out.setdefault(f, {"n": 0, "caught": 0})
            d["n"] += 1
            d["caught"] += int(pc == 0)
    return {f: {**d, "recall": round(d["caught"] / d["n"], 3)} for f, d in sorted(out.items())}


def _dump_errors(path: Path, ex: str, cols: list[str], X: np.ndarray, y: np.ndarray,
                 faults: list[list[str]], p: np.ndarray, thr: float, k: int) -> None:
    pred = (p >= thr).astype(int)
    fn = np.where((y == 0) & (pred == 1))[0]          # faulty rep passed as clean (missed fault)
    fp = np.where((y == 1) & (pred == 0))[0]          # clean rep flagged as faulty (false alarm)
    fn = fn[np.argsort(-p[fn])][:k]
    fp = fp[np.argsort(p[fp])][:k]
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["exercise", "error", "p_clean", "true_faults", *FEATURES])
        for kind, idx in (("missed_fault", fn), ("false_alarm", fp)):
            for i in idx:
                w.writerow([ex, kind, round(float(p[i]), 3), "|".join(faults[i]) or "-", *[round(float(v), 3) for v in X[i]]])


def train(cfg: dict | None = None, out_dir: Path | None = None, save_model: bool = True) -> dict:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    seed = cfg["seed"]
    t0 = time.time()
    bundle = {"features": list(FEATURES), "columns": {}, "models": {}, "thresholds": {},
              "abstain_conf": {}, "ood_bounds": {}, "version": None}
    report = {"config": cfg, "env": env_info(), "synthetic": True, "exercises": {}}
    datasets = {}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        err_path = out_dir / "errors.csv"
        err_path.unlink(missing_ok=True)

    for i, ex in enumerate(cfg["exercises"]):
        Xtr_f, ytr = generate(ex, cfg["n_train"], seed=seed + i)
        Xva_f, yva = generate(ex, cfg["n_val"], seed=5_000 + seed + i)
        Xte_f, yte, fte = generate(ex, cfg["n_test"], seed=10_000 + seed + i, with_faults=True)
        datasets[ex] = {"train": dataset_hash(Xtr_f, ytr), "val": dataset_hash(Xva_f, yva),
                        "test": dataset_hash(Xte_f, yte),
                        "clean_rate": {"train": round(float(ytr.mean()), 3), "val": round(float(yva.mean()), 3),
                                       "test": round(float(yte.mean()), 3)}}
        cols = list(EXERCISE_FEATURES[ex])
        ci = [FEATURES.index(c) for c in cols]
        Xtr, Xva, Xte = Xtr_f[:, ci], Xva_f[:, ci], Xte_f[:, ci]

        results: dict[str, dict] = {}
        # --- baselines ---
        maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
        results["majority"] = classification_report(yte, None, maj.predict(Xte))
        results["majority"]["note"] = f"always predicts {'clean' if maj.predict(Xte[:1])[0] else 'faulty'}"
        pr_va, pr_te = _rule_proba(ex, Xva_f), _rule_proba(ex, Xte_f)
        thr_rule = pick_threshold(yva, pr_va)
        results["rules"] = classification_report(yte, pr_te, (pr_te >= thr_rule).astype(int))
        results["rules"]["threshold"] = thr_rule
        results["rules"]["note"] = "rule score / 100 used as P(clean); not a calibrated probability"

        # --- learned models: fit on train, select + tune on validation ---
        fitted, val_auc = {}, {}
        for name, m in _candidates(cfg).items():
            m.fit(Xtr, ytr)
            fitted[name] = m
            val_auc[name] = classification_report(yva, m.predict_proba(Xva)[:, 1], m.predict(Xva))["roc_auc"]
        best = max(val_auc, key=val_auc.get)
        for name, m in fitted.items():
            p_va = m.predict_proba(Xva)[:, 1]
            thr = pick_threshold(yva, p_va)
            conf = pick_abstain_margin(yva, p_va, thr, cfg["selective_target_accuracy"])
            p_te = m.predict_proba(Xte)[:, 1]
            results[name] = classification_report(yte, p_te, (p_te >= thr).astype(int), conf)
            results[name].update({"threshold": thr, "val_roc_auc": val_auc[name]})
            if name == best:
                bundle["thresholds"][ex], bundle["abstain_conf"][ex] = thr, conf
                results[name]["fault_type_recall"] = _fault_recall(fte, (p_te >= thr).astype(int))
                if out_dir:
                    _dump_errors(err_path, ex, cols, Xte_f, yte, fte, p_te, thr, cfg["errors_per_side"])

        model = fitted[best]
        ex_rep = {"selected": best, "features": cols, "results": results, "dataset": datasets[ex]}
        if hasattr(model, "feature_importances_"):
            imp = sorted(zip(cols, model.feature_importances_), key=lambda t: -t[1])
            ex_rep["feature_importance"] = {k: round(float(v), 3) for k, v in imp}
        report["exercises"][ex] = ex_rep
        bundle["models"][ex] = model
        bundle["columns"][ex] = cols
        bundle["ood_bounds"][ex] = _ood_bounds(Xtr, cols, cfg["ood_quantiles"], cfg["ood_margin"])
        r = results[best]
        print(f"{ex:8s} {best:18s} auc={r['roc_auc']} pr_auc={r['pr_auc_faulty']} f1_fault={r['faulty']['f1']} "
              f"ece={r['ece']} | rules auc={results['rules']['roc_auc']} f1={results['rules']['faulty']['f1']} "
              f"| majority acc={results['majority']['accuracy']}")

    version = f"{time.strftime('%Y%m%d')}-{dataset_hash(*[np.frombuffer(json.dumps(datasets, sort_keys=True).encode(), np.uint8)])[:8]}"
    bundle["version"] = version
    report["model_version"] = version
    report["train_seconds"] = round(time.time() - t0, 1)
    card = model_card(cfg, report, datasets, version)

    if save_model:
        ART.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, ART / "form_model.joblib", compress=3)
        (ART / "form_model.card.json").write_text(json.dumps(card, indent=2))
        (ART / "metrics.json").write_text(json.dumps(report, indent=2))
        print("saved", ART / "form_model.joblib", f"{(ART / 'form_model.joblib').stat().st_size / 1e6:.2f} MB")
    if out_dir:
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
        (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    return report


def model_card(cfg: dict, report: dict, datasets: dict, version: str) -> dict:
    summary = {}
    for ex, r in report["exercises"].items():
        b = r["results"][r["selected"]]
        summary[ex] = {
            "model": r["selected"],
            "roc_auc": b["roc_auc"], "pr_auc_faulty": b["pr_auc_faulty"], "f1_faulty": b["faulty"]["f1"],
            "accuracy": b["accuracy"], "brier": b["brier"], "ece": b["ece"],
            "threshold": b["threshold"], "selective": b.get("selective"),
            "baseline_rules": {"roc_auc": r["results"]["rules"]["roc_auc"], "f1_faulty": r["results"]["rules"]["faulty"]["f1"]},
            "baseline_logreg": {"roc_auc": r["results"]["logreg"]["roc_auc"], "f1_faulty": r["results"]["logreg"]["faulty"]["f1"]},
            "baseline_majority": {"accuracy": r["results"]["majority"]["accuracy"], "f1_faulty": r["results"]["majority"]["faulty"]["f1"]},
        }
    return {
        "name": "formfit-form-quality",
        "model_version": version,
        "trained_at": report["env"]["timestamp"],
        "git_commit": report["env"]["git_commit"],
        "task": "Binary classification per exercise: is a rep clean (no form fault)? Output P(clean).",
        "dataset": {
            "name": "formfit-simulated-reps",
            "synthetic": True,
            "simulator": "backend/ml/simulate.py",
            "simulator_version": SIMULATOR_VERSION,
            "sizes": {"train": cfg["n_train"], "val": cfg["n_val"], "test": cfg["n_test"]},
            "hashes": datasets,
        },
        "preprocessing_version": FEATURES_VERSION,
        "features": {ex: r["features"] for ex, r in report["exercises"].items()},
        "hyperparameters": cfg["models"],
        "seed": cfg["seed"],
        "libraries": {k: report["env"][k] for k in ("python", "numpy", "scikit_learn", "joblib")},
        "decision": {
            "threshold": "chosen on validation to maximise faulty-class F1",
            "abstention": f"prediction marked 'uncertain' below the confidence level that reached "
                          f"{cfg['selective_target_accuracy']:.0%} accuracy on validation; 'out_of_distribution' when a "
                          f"feature is outside the training range (quantiles {cfg['ood_quantiles']} widened by {cfg['ood_margin']:.0%})",
        },
        "metrics_test_synthetic": summary,
        "intended_use": "A second opinion next to the rule score in a personal training app. Not for medical, "
                        "rehabilitation or injury-risk decisions.",
        "limitations": [
            "Trained and evaluated only on simulated reps. Accuracy on real lifters is unknown.",
            "Labels come from the simulator's own fault definitions, not from coaches.",
            "2-D single-camera features; view-dependent measurements are approximations.",
        ],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    c = json.loads(a.config.read_text()) if a.config else None
    train(c, a.out_dir, save_model=not a.no_save)
