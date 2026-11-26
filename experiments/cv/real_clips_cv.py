"""Computer-vision checks on the REAL clips in data/real_clips/manifest.json (11 clips, 217 reps).

* Rep counting through the shipped TypeScript engine in three configurations: baseline, + landmark
  guard, + guard + exercise-TCN gate (the app's default).
* False repetitions on real video: every clip is also run with every OTHER exercise selected; reps per
  minute without and with the gate.
* Exercise classification on real video: 2-s windows (stride 1 s) labelled with the clip's exercise,
  classified by the shipped TCN and by the single-frame pose model (retrained here on the synthetic
  `train` split, as in cv_classification).

Needs: python -m experiments.real_clips.fetch, landmarks from the real_clips_repcount run, and the
synthetic `train` split (only for the frame model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cv.classify import CLS_IDX, WIN, fit_frame_model, frame_model_predict, load_split, load_tcn, tcn_predict  # noqa: E402
from cv.engine_eval import run_engine  # noqa: E402
from cv.normalize import frame_features, frames_to_array  # noqa: E402
from cv.skeleton import CLASSES, EXERCISES  # noqa: E402

MANIFEST = ROOT / "data" / "real_clips" / "manifest.json"
LM = ROOT / "data" / "real_clips" / "landmarks"
TCN = ROOT / "backend" / "ml" / "artifacts" / "exercise_tcn.npz"
GATE = 0.8


def _counts(rows):
    err = np.array([r["pred"] - r["gt"] for r in rows])
    return {"clips": len(rows), "gt_reps": int(sum(r["gt"] for r in rows)), "pred_reps": int(sum(r["pred"] for r in rows)),
            "mae": round(float(np.abs(err).mean()), 3), "exact": round(float((err == 0).mean()), 3),
            "within_1": round(float((np.abs(err) <= 1).mean()), 3)}


def run(cfg: dict, out: Path, log=print) -> dict:
    clips = [c for c in json.loads(MANIFEST.read_text())["clips"] if (LM / f"{c['id']}.json").exists()]
    log(f"{len(clips)} clips with landmarks")
    variants = {"baseline": {"guard": False}, "guard": {"guard": True}, "guard_gate": {"guard": True, "gate": GATE}}
    jobs = [{"id": f"{c['id']}|{v}|{c['exercise']}", "file": str(LM / f"{c['id']}.json"), "exercise": c["exercise"], **o}
            for c in clips for v, o in variants.items()]
    jobs += [{"id": f"{c['id']}|{v}|{sel}", "file": str(LM / f"{c['id']}.json"), "exercise": sel, **o}
             for c in clips for sel in EXERCISES if sel != c["exercise"]
             for v, o in variants.items() if v != "baseline"]
    res = run_engine(jobs)
    res.pop("_ms_per_frame")

    rep: dict = {"data": "REAL video (data/real_clips/manifest.json); 5 of 11 clips are rendered 3-D avatars",
                 "gate_tau": GATE, "per_clip": []}
    rows = {v: [] for v in variants}
    for c in clips:
        line = {"id": c["id"], "exercise": c["exercise"], "view": c["view"], "source": c["source_type"], "gt": c["gt_reps"]}
        for v in variants:
            r = res[f"{c['id']}|{v}|{c['exercise']}"]
            line[v] = len(r["reps"])
            if v == "guard_gate":
                line["vetoed"] = r["vetoed"]
            rows[v].append({"gt": c["gt_reps"], "pred": len(r["reps"]), "source": c["source_type"]})
        rep["per_clip"].append(line)
    rep["rep_counting"] = {v: _counts(rs) | {"filmed": _counts([x for x in rs if x["source"] == "filmed"])}
                           for v, rs in rows.items()}

    # false reps: wrong exercise selected
    fr = {}
    for v in ("guard", "guard_gate"):
        n = mins = 0.0
        cells = {}
        for c in clips:
            d = json.loads((LM / f"{c['id']}.json").read_text())
            m = len(d["frames"]) / d["fps"] / 60
            for sel in EXERCISES:
                if sel == c["exercise"]:
                    continue
                k = len(res[f"{c['id']}|{v}|{sel}"]["reps"])
                n += k
                mins += m
                key = f"{sel}<-{c['exercise']}"
                cells[key] = cells.get(key, 0) + k
        fr[v] = {"reps_per_min": round(n / mins, 3), "reps": int(n), "minutes": round(mins, 2), "by_selected_actual": cells}
    rep["false_reps_wrong_exercise"] = fr
    log(f"real false reps/min: {fr['guard']['reps_per_min']} -> {fr['guard_gate']['reps_per_min']} with gate")

    # exercise classification on real windows
    tcn = load_tcn(TCN)
    X, y, wid = [], [], []
    for c in clips:
        d = json.loads((LM / f"{c['id']}.json").read_text())
        lm, present = frames_to_array(d["frames"])
        aspect = d["width"] / d["height"]
        for st in range(0, len(lm) - WIN + 1, 15):
            X.append(frame_features(lm[st:st + WIN], present[st:st + WIN], aspect))
            y.append(CLS_IDX[c["exercise"]])
            wid.append(c["id"])
    X, y = np.array(X, np.float32), np.array(y)
    log(f"{len(y)} real windows; training the frame model on the synthetic train split ...")
    fm = fit_frame_model(load_split("train", augment_mirror=True))
    present = sorted(set(y.tolist()))
    cls_rep = {}
    for name, pred in (("tcn (temporal)", tcn_predict(tcn, X).argmax(1)),
                       ("frame (pose-based)", frame_model_predict(fm, X).argmax(1))):
        per = {}
        for c in sorted(set(wid)):
            k = np.array([w == c for w in wid])
            per[c] = {"windows": int(k.sum()), "accuracy": round(float((pred[k] == y[k]).mean()), 3),
                      "predicted": {CLASSES[i]: int((pred[k] == i).sum()) for i in set(pred[k].tolist())}}
        cls_rep[name] = {
            "window_accuracy": round(float((pred == y).mean()), 4),
            "macro_f1_present_classes": round(float(f1_score(y, pred, labels=present, average="macro")), 4),
            "clip_majority_accuracy": round(float(np.mean([np.bincount(pred[np.array([w == c for w in wid])]).argmax()
                                                           == CLS_IDX[next(x['exercise'] for x in clips if x['id'] == c)]
                                                           for c in set(wid)])), 3),
            "per_clip": per}
    rep["exercise_classification"] = {"windows": int(len(y)), "classes_present": [CLASSES[i] for i in present],
                                      "results": cls_rep}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep
