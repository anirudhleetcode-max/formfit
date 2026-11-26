"""Exercise classification: pose-based (single frame) vs temporal models.

Protocols
  standard    train on `train` (+ mirrored copies), test on `test` (same distribution, new people,
              cameras and seeds), `test_hard` (2-3x landmark noise, occluders in 70% of sequences)
              and the camera-yaw sweep.
  cross_view  train only on side and oblique cameras, test on front and rear cameras the models
              never saw: how much does each approach depend on the viewpoint?
  short       the same models on 1-second windows (15 frames; time-resampled to the model input).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cv.classify import (CLS_IDX, WIN, fit_frame_model, fit_stats_model, frame_model_predict, latency_ms,  # noqa: E402
                         load_split, report, save_tcn, stats_features, tcn_predict, train_tcn, windows)
from cv.skeleton import CLASSES  # noqa: E402

ARTIFACT = ROOT / "backend" / "ml" / "artifacts" / "exercise_tcn.npz"
SIDEISH = {"side", "oblique", "rear-oblique"}


def _eval_all(models: dict, X, y) -> dict:
    out = {}
    for name, pred in models.items():
        out[name] = report(y, pred(X).argmax(1))
    return out


def _short(X: np.ndarray, n=15) -> np.ndarray:
    """Keep the last n frames and resample them to WIN frames (the model input length)."""
    src = X[:, -n:]
    pos = np.linspace(0, n - 1, WIN)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    a = (pos - lo)[None, :, None]
    return src[:, lo] * (1 - a) + src[:, hi] * a


def _train_three(train, val_X, val_y, steps, log):
    t = time.time()
    fm = fit_frame_model(train)
    t_frame = time.time() - t
    tX, ty, _ = windows(train, 10)
    t = time.time()
    sm = fit_stats_model(tX, ty)
    t_stats = time.time() - t
    tcn = train_tcn(train, val_X, val_y, steps=steps, log=log)
    models = {
        "frame (pose-based)": lambda X: frame_model_predict(fm, X),
        "stats (window features)": lambda X: sm.predict_proba(stats_features(X)),
        "tcn (temporal)": lambda X: tcn_predict(tcn, X),
    }
    return models, tcn, {"frame": round(t_frame, 1), "stats": round(t_stats, 1), "tcn": tcn["train_s"]}


def run(cfg: dict, out: Path, log=print) -> dict:
    steps = cfg.get("tcn_steps", 2500)
    log("loading splits ...")
    train = load_split("train", augment_mirror=True)
    val, test, hard, sweep = (load_split(n) for n in ("val", "test", "test_hard", "sweep"))
    vX, vy, _ = windows(val, 15)
    X, y, where = windows(test, 15)
    hX, hy, _ = windows(hard, 15)
    sX, sy, swhere = windows(sweep, 15)
    log(f"windows: test {len(y)}, hard {len(hy)}, sweep {len(sy)}")

    models, tcn, train_s = _train_three(train, vX, vy, steps, log)
    save_tcn(tcn, ARTIFACT)
    rep = {"window_frames": WIN, "fps": 15, "classes": CLASSES, "train_seconds": train_s,
           "n_train_sequences": len(train), "n_test_windows": int(len(y))}
    rep["standard_test"] = _eval_all(models, X, y)
    rep["hard_test"] = _eval_all(models, hX, hy)
    rep["short_1s_test"] = _eval_all(models, _short(X), y)

    # per camera view on the test split and per yaw on the sweep
    views = np.array([test[si].view for si, _ in where])
    rep["by_view_macro_f1"] = {}
    for v in ["front", "oblique", "side", "rear-oblique", "rear"]:
        k = views == v
        rep["by_view_macro_f1"][v] = {n: report(y[k], p(X[k]).argmax(1))["macro_f1"] for n, p in models.items()}
    yaws = np.array([round(sweep[si].meta["camera"]["yaw"]) % 360 for si, _ in swhere])
    rep["sweep_accuracy_by_yaw"] = {
        str(a): {n: round(float((p(sX[yaws == a]).argmax(1) == sy[yaws == a]).mean()), 4) for n, p in models.items()}
        for a in sorted(set(yaws.tolist()))}

    # sequence level: majority vote of window predictions
    seq_acc = {}
    for n, p in models.items():
        pr = p(X).argmax(1)
        votes = {}
        for (si, _), c in zip(where, pr):
            votes.setdefault(si, []).append(c)
        seq_acc[n] = round(float(np.mean([np.bincount(v).argmax() == test[si].label for si, v in votes.items()])), 4)
    rep["sequence_accuracy"] = seq_acc

    # cross-view: train on side-ish cameras only, test on front/rear
    log("cross-view protocol ...")
    tr_cv = [s for s in train if s.view in SIDEISH]
    val_cv = [s for s in val if s.view in SIDEISH]
    cvX, cvy, _ = windows(val_cv, 15)
    m_cv, _, _ = _train_three(tr_cv, cvX, cvy, steps, log)
    k = np.isin(views, ["front", "rear"])
    rep["cross_view"] = {"train_views": sorted(SIDEISH), "test_views": ["front", "rear"],
                         "n_train_sequences": len(tr_cv), "n_test_windows": int(k.sum()),
                         "results": _eval_all(m_cv, X[k], y[k])}

    # latency: one 2-s window, batch size 1, single thread NumPy / sklearn on this machine
    one = X[:1]
    rep["latency_ms_per_window"] = {n: latency_ms(p, one, 100) for n, p in models.items()}
    rep["tcn"] = {"params": int(sum(v.size for v in tcn["params"].values())), "history": tcn["history"],
                  "artifact": str(ARTIFACT.relative_to(ROOT)), "artifact_kb": round(ARTIFACT.stat().st_size / 1024, 1)}

    (out / "metrics.json").write_text(json.dumps(rep, indent=1))
    return rep
