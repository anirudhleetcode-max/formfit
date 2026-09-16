"""Loads the trained form-quality classifiers once (at startup) and scores reps.

Per rep the service returns
  model_score       100 * P(clean) (int) or None when the model abstains
  model_confidence  max(P, 1 - P) or None
  model_status      ok | uncertain | out_of_distribution | not_scored | unavailable
  model_ood         feature names outside the training range (when out_of_distribution)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ml.features import vector

log = logging.getLogger("formfit.model")
BACKEND_DIR = Path(__file__).resolve().parents[2]

STATUSES = ("ok", "uncertain", "out_of_distribution", "not_scored", "unavailable")


class FormModel:
    def __init__(self, path: str | Path):
        p = Path(path)
        self.path = p if p.is_absolute() else BACKEND_DIR / p
        self.bundle: dict | None = None
        self.metrics: dict | None = None
        self.card: dict | None = None
        self.real_eval: dict | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            bundle = joblib.load(self.path)
            if not isinstance(bundle, dict) or "models" not in bundle:
                raise ValueError("unexpected artifact format")
            self.bundle = bundle
            self.error = None
            for name, attr in (("metrics.json", "metrics"), ("form_model.card.json", "card"),
                               ("real_clips_eval.json", "real_eval")):
                f = self.path.with_name(name)
                setattr(self, attr, json.loads(f.read_text()) if f.exists() else None)
            log.info("form model %s loaded from %s", bundle.get("version"), self.path.name)
        except Exception as e:  # missing file, corrupt file or incompatible scikit-learn version
            self.bundle = None
            self.error = f"{type(e).__name__}: {e}. Run `python -m ml.train`."
            log.warning("form model unavailable: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def _ood(self, exercise: str, row: list[float], cols: list[str]) -> list[str]:
        bounds = self.bundle.get("ood_bounds", {}).get(exercise, {})
        return [c for c, v in zip(cols, row) if c in bounds and not (bounds[c][0] <= v <= bounds[c][1])]

    def predict(self, exercise: str, feature_dicts: list[dict], scored: list[bool] | None = None) -> list[dict]:
        """CPU work: call in a thread. `scored[i] = False` means the browser abstained (low tracking
        confidence); such reps are not given a model score either."""
        n = len(feature_dicts)
        scored = scored if scored is not None else [True] * n
        empty = lambda status: {"model_score": None, "model_confidence": None, "model_status": status, "model_ood": []}  # noqa: E731
        if not self.ready or exercise not in self.bundle["models"]:
            return [empty("unavailable") for _ in range(n)]
        out = [empty("not_scored") for _ in range(n)]
        idx = [i for i in range(n) if scored[i]]
        if not idx:
            return out
        cols = self.bundle.get("columns", {}).get(exercise, self.bundle["features"])
        rows = [vector(feature_dicts[i], cols) for i in idx]
        proba = self.bundle["models"][exercise].predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
        min_conf = self.bundle.get("abstain_conf", {}).get(exercise, 0.0)
        for i, row, p in zip(idx, rows, proba):
            ood = self._ood(exercise, row, cols)
            conf = float(max(p, 1 - p))
            if ood:
                out[i] = {"model_score": None, "model_confidence": None, "model_status": "out_of_distribution", "model_ood": ood}
            else:
                out[i] = {"model_score": int(round(p * 100)), "model_confidence": round(conf, 3),
                          "model_status": "ok" if conf >= min_conf else "uncertain", "model_ood": []}
        return out

    def score(self, exercise: str, feature_dicts: list[dict]) -> list[int | None]:
        """Backward-compatible helper: model scores only."""
        return [r["model_score"] for r in self.predict(exercise, feature_dicts)]

    def info(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "version": self.bundle.get("version") if self.ready else None,
            "features": self.bundle["features"] if self.ready else None,
            "thresholds": self.bundle.get("thresholds") if self.ready else None,
            "abstain_confidence": self.bundle.get("abstain_conf") if self.ready else None,
            "sklearn_version": sklearn.__version__,
            "synthetic": True,
            "training_data": "SYNTHETIC reps from backend/ml/simulate.py (biomechanical simulation); "
                             "real-lifter accuracy unknown",
            "card": self.card,
            "metrics": _metrics_summary(self.metrics),
            # rep counting on real public clips (experiments/real_clips); None if never run
            "real_clip_eval": self.real_eval,
        }


def _metrics_summary(m: dict | None) -> dict | None:
    """Compact, backward-compatible view (n_train/n_test + per-exercise model/rule_baseline)."""
    if not m:
        return None
    cfg = m.get("config", {})
    ex_out = {}
    for ex, r in m.get("exercises", {}).items():
        best = r["results"][r["selected"]]
        rules_ = r["results"]["rules"]
        pick = lambda d: {"accuracy": d["accuracy"], "f1_faulty": d["faulty"]["f1"], "roc_auc": d.get("roc_auc"),  # noqa: E731
                          "pr_auc_faulty": d.get("pr_auc_faulty"), "ece": d.get("ece"), "brier": d.get("brier")}
        ex_out[ex] = {"selected": r["selected"], "model": pick(best), "rule_baseline": pick(rules_),
                      "logreg": pick(r["results"]["logreg"]), "majority": pick(r["results"]["majority"]),
                      "selective": best.get("selective")}
    return {"n_train": cfg.get("n_train"), "n_val": cfg.get("n_val"), "n_test": cfg.get("n_test"),
            "synthetic": True, "exercises": ex_out, "model_version": m.get("model_version")}


_model: FormModel | None = None


def get_model() -> FormModel:
    global _model
    if _model is None:
        from ..config import get_settings
        _model = FormModel(get_settings().model_path)
        _model.load()
    return _model


def reset_model(path: str | Path | None = None) -> FormModel:
    """Reload (tests / after retraining)."""
    global _model
    from ..config import get_settings
    _model = FormModel(path or get_settings().model_path)
    _model.load()
    return _model
