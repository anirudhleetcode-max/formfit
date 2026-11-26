"""Export the exercise TCN to JSON for the browser and write a Python/TypeScript parity fixture.

    python -m cv.export_tcn

* frontend/public/models/exercise_tcn.json  weights + feature standardisation (served with the app)
* frontend/src/pose/__tests__/fixtures/tcn_parity.json  raw landmark windows + the probabilities
  NumPy computes for them; the vitest suite checks the TypeScript port reproduces them.

In the browser the features are computed from the last 30 frames only (online), so the body-size
scale is the median over that window rather than over the whole sequence as in training. This script
also measures accuracy with window-only features on the test split, so the cost of that change is
known (it is printed and stored in the JSON as `window_only_test`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .classify import CLS_IDX, TCN_LAYERS, WIN, load_tcn, tcn_predict
from .dataset import DATA, load_meta
from .normalize import FRAME_FEATURES, frame_features, frames_to_array
from .skeleton import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "backend" / "ml" / "artifacts" / "exercise_tcn.npz"
OUT = ROOT / "frontend" / "public" / "models" / "exercise_tcn.json"
FIX = ROOT / "frontend" / "src" / "pose" / "__tests__" / "fixtures" / "tcn_parity.json"


def window_eval(model, n_seq=300, stride=15) -> dict:
    y, pred = [], []
    for m in load_meta("test")[:n_seq]:
        s = json.loads((DATA / "test" / f"{m['id']}.json").read_text())
        lm, present = frames_to_array(s["frames"])
        act = np.array(s["active"])
        aspect = s["width"] / s["height"]
        for st in range(0, len(lm) - WIN + 1, stride):
            if m["label"] != "other" and act[st:st + WIN].mean() < 0.5:
                continue
            f = frame_features(lm[st:st + WIN], present[st:st + WIN], aspect)
            pred.append(int(tcn_predict(model, f[None])[0].argmax()))
            y.append(CLS_IDX[m["label"]])
    y, pred = np.array(y), np.array(pred)
    return {"windows": int(len(y)), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "accuracy": round(float((y == pred).mean()), 4)}


def main() -> None:
    model = load_tcn(SRC)
    r6 = lambda a: np.round(a.astype(float), 6).tolist()  # noqa: E731
    w = window_eval(model)
    payload = {"version": 1, "classes": CLASSES, "window": WIN, "fps": 15, "features": FRAME_FEATURES,
               "layers": [{"kernel": k, "dilation": d} for k, d in TCN_LAYERS],
               "mu": r6(model["mu"]), "sd": r6(model["sd"]),
               "params": {k: r6(v) for k, v in model["params"].items()}, "window_only_test": w,
               "data": "trained on SYNTHETIC skeleton sequences only (backend/cv/skeleton.py)"}
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB); window-only features on test: {w}")

    cases = []
    metas = load_meta("test")
    for cls in CLASSES:
        m = next(x for x in metas if x["label"] == cls)
        s = json.loads((DATA / "test" / f"{m['id']}.json").read_text())
        st = len(s["frames"]) // 2 - WIN // 2
        frames = s["frames"][st:st + WIN]
        lm, present = frames_to_array(frames)
        f = frame_features(lm, present, s["width"] / s["height"])
        p = tcn_predict(load_tcn(SRC), f[None])[0]
        cases.append({"label": cls, "aspect": s["width"] / s["height"], "frames": [fr["lm"] for fr in frames],
                      "features": np.round(f, 5).tolist(), "probs": np.round(p, 6).tolist()})
    FIX.parent.mkdir(parents=True, exist_ok=True)
    FIX.write_text(json.dumps({"cases": cases}))
    print(f"wrote {FIX}")


if __name__ == "__main__":
    main()
