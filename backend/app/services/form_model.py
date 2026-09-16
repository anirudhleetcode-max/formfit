"""Loads the trained form-quality classifiers once (at startup) and scores reps."""
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


class FormModel:
    def __init__(self, path: str):
        p = Path(path)
        self.path = p if p.is_absolute() else BACKEND_DIR / p
        self.bundle: dict | None = None
        self.metrics: dict | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.bundle = joblib.load(self.path)
            mpath = self.path.with_name("metrics.json")
            if mpath.exists():
                self.metrics = json.loads(mpath.read_text())
            log.info("form model loaded from %s", self.path)
        except Exception as e:  # missing file or incompatible scikit-learn version
            self.bundle = None
            self.error = f"{type(e).__name__}: {e}. Run `python -m ml.train`."
            log.warning("form model unavailable: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def score(self, exercise: str, feature_dicts: list[dict]) -> list[int | None]:
        """Probability that each rep is 'good', as a 0-100 score. CPU work: call in a thread."""
        if not self.ready or not feature_dicts or exercise not in self.bundle["models"]:
            return [None] * len(feature_dicts)
        cols = self.bundle.get("columns", {}).get(exercise, self.bundle["features"])
        X = np.asarray([vector(f, cols) for f in feature_dicts], dtype=np.float32)
        proba = self.bundle["models"][exercise].predict_proba(X)[:, 1]
        return [int(round(p * 100)) for p in proba]

    def info(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "features": self.bundle["features"] if self.ready else None,
            "sklearn_version": sklearn.__version__,
            "metrics": self.metrics,
            "training_data": "synthetic reps from ml/simulate.py (biomechanical simulation)",
        }


_model: FormModel | None = None


def get_model() -> FormModel:
    global _model
    if _model is None:
        from ..config import get_settings
        _model = FormModel(get_settings().model_path)
        _model.load()
    return _model
