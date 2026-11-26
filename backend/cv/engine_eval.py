"""Run synthetic sequences through the app's TypeScript rep engine and score it against ground truth.

The engine under test is frontend/src/pose/engine.ts itself (via frontend/scripts/eval-synthetic.ts),
so these numbers describe the shipped code, not a Python re-implementation.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import DATA
from .skeleton import EXERCISES, FAULTS

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


def run_engine(jobs: list[dict]) -> dict:
    """jobs: [{id, file, exercise, guard}] -> {id: result}"""
    with tempfile.TemporaryDirectory() as d:
        jp, op = Path(d) / "jobs.json", Path(d) / "out.json"
        jp.write_text(json.dumps({"jobs": jobs}))
        subprocess.run(["npx", "tsx", "scripts/eval-synthetic.ts", str(jp), str(op)], cwd=FRONTEND, check=True,
                       capture_output=True, text=True)
        out = json.loads(op.read_text())
    return {r["id"]: r for r in out["results"]} | {"_ms_per_frame": out["ms_per_frame"]}


def jobs_for(dataset: str, metas: list[dict], guard=False, selected: str | None = None, suffix="") -> list[dict]:
    jobs = []
    for m in metas:
        ex = selected or m["label"]
        if ex not in EXERCISES:
            continue
        jobs.append({"id": m["id"] + suffix, "file": str(DATA / dataset / f"{m['id']}.json"), "exercise": ex,
                     "guard": guard})
    return jobs


def match_reps(gt: list[dict], pred: list[dict], late=1.2):
    """Greedy time matching. A predicted rep (time = when the engine closed it) matches a GT rep if it
    falls between the GT turning point and GT end + `late` seconds. Returns (pairs, fp_idx, fn_idx, dup_idx)."""
    used = set()
    pairs, fp, dup = [], [], []
    for j, p in enumerate(pred):
        cands = [i for i, g in enumerate(gt) if g["turn"] - 0.05 <= p["t"] <= g["end"] + late]
        free = [i for i in cands if i not in used]
        if free:
            used.add(free[0])
            pairs.append((free[0], j))
        elif cands:
            dup.append(j)
        else:
            fp.append(j)
    fn = [i for i in range(len(gt)) if i not in used]
    return pairs, fp, fn, dup


def _near(events: list[list[float]], t: float, pad=1.0) -> bool:
    return any(a - pad <= t <= b + pad for a, b in events)


def classify_fp(meta: dict, p: dict) -> str:
    obs = meta["observation"]
    ev = list(obs["swaps"])
    if obs["occluder"]:
        ev.append([obs["occluder"]["t0"], obs["occluder"]["t1"]])
    return "tracking_glitch" if _near(ev, p["t"]) else "spurious"


def classify_fn(meta: dict, g: dict, exercise: str) -> str:
    obs = meta["observation"]
    if obs["occluder"] and not (obs["occluder"]["t1"] < g["start"] or obs["occluder"]["t0"] > g["end"]):
        return "occlusion"
    if obs.get("cut"):
        return "out_of_frame"
    if SHALLOW[exercise] in g["faults"]:
        return "partial_range"
    if meta["camera"]["view"] in ("front", "rear") and exercise in ("squat", "lunge", "curl", "pushup"):
        return "unfavourable_view"
    return "other"


SHALLOW = {"squat": "shallow_depth", "pushup": "shallow_pushup", "curl": "partial_curl", "press": "short_press",
           "lunge": "shallow_lunge"}


def summarize_counts(rows: list[dict]) -> dict:
    if not rows:
        return {}
    err = np.array([r["pred"] - r["gt"] for r in rows])
    err_full = np.array([r["pred"] - r["gt_full"] for r in rows])
    return {"n": len(rows), "gt_reps": int(sum(r["gt"] for r in rows)), "pred_reps": int(sum(r["pred"] for r in rows)),
            "mae": round(float(np.abs(err).mean()), 3), "bias": round(float(err.mean()), 3),
            "exact": round(float((err == 0).mean()), 3), "within_1": round(float((np.abs(err) <= 1).mean()), 3),
            "rep_precision": round(sum(r["tp"] for r in rows) / max(1, sum(r["pred"] for r in rows)), 3),
            "rep_recall": round(sum(r["tp"] for r in rows) / max(1, sum(r["gt"] for r in rows)), 3),
            # the app deliberately does not count partial-range reps ("Go lower"); against full-range reps only:
            "mae_full_range": round(float(np.abs(err_full).mean()), 3),
            "exact_full_range": round(float((err_full == 0).mean()), 3)}


def evaluate_counting(dataset: str, metas: list[dict], results: dict, gate=None) -> dict:
    """Rep counting + false-rep / missed-rep taxonomy + posture precision/recall on matched, scored reps.
    gate(meta, exercise, rep) -> bool can veto predicted reps (used for the classifier gate)."""
    rows, fp_types, fn_types = [], Counter(), Counter()
    fault_tp, fault_fp, fault_fn = Counter(), Counter(), Counter()
    any_tp = any_fp = any_fn = any_tn = 0
    scored = matched = 0
    by_view_posture = defaultdict(lambda: [0, 0, 0])
    for m in metas:
        if m["label"] not in EXERCISES or m["id"] not in results:
            continue
        ex = m["label"]
        pred = results[m["id"]]["reps"]
        if gate is not None:
            pred = [p for p in pred if gate(m, ex, p)]
        pairs, fp, fn, dup = match_reps(m["reps"], pred)
        for j in fp:
            fp_types[classify_fp(m, pred[j])] += 1
        fp_types["double_count"] += len(dup)
        for i in fn:
            fn_types[classify_fn(m, m["reps"][i], ex)] += 1
        rows.append({"id": m["id"], "exercise": ex, "view": m["camera"]["view"], "yaw": m["camera"]["yaw"],
                     "occluded": m["observation"]["occluder"] is not None,
                     "gt": len(m["reps"]), "pred": len(pred), "tp": len(pairs),
                     "gt_full": sum(SHALLOW[ex] not in g["faults"] for g in m["reps"])})
        for gi, pj in pairs:
            matched += 1
            p, g = pred[pj], m["reps"][gi]
            if not p["scored"]:
                continue
            scored += 1
            gt_f, pr_f = set(g["faults"]), set(p["faults"]) & set(FAULTS[ex])
            for f in gt_f & pr_f: fault_tp[f] += 1
            for f in pr_f - gt_f: fault_fp[f] += 1
            for f in gt_f - pr_f: fault_fn[f] += 1
            a, b = bool(gt_f), bool(pr_f)
            any_tp += a and b; any_fp += (not a) and b; any_fn += a and not b; any_tn += (not a) and not b
            v = by_view_posture[m["camera"]["view"]]
            v[0] += a and b; v[1] += (not a) and b; v[2] += a and not b

    def pr(tp, fp_, fn_):
        return {"precision": round(tp / (tp + fp_), 3) if tp + fp_ else None,
                "recall": round(tp / (tp + fn_), 3) if tp + fn_ else None, "support": tp + fn_}

    per_fault = {f: pr(fault_tp[f], fault_fp[f], fault_fn[f]) for ex in EXERCISES for f in FAULTS[ex]
                 if fault_tp[f] + fault_fp[f] + fault_fn[f]}
    by_ex = {ex: summarize_counts([r for r in rows if r["exercise"] == ex]) for ex in EXERCISES}
    by_view = {v: summarize_counts([r for r in rows if r["view"] == v])
               for v in ["front", "oblique", "side", "rear-oblique", "rear"]}
    return {
        "counting": summarize_counts(rows), "counting_by_exercise": by_ex, "counting_by_view": by_view,
        "counting_occluded": summarize_counts([r for r in rows if r["occluded"]]),
        "false_reps": dict(fp_types), "missed_reps": dict(fn_types),
        "posture": {"matched_reps": matched, "scored_reps": scored,
                    "coverage": round(scored / max(1, matched), 3),
                    "any_fault": pr(any_tp, any_fp, any_fn) | {"tn": any_tn},
                    "per_fault": per_fault,
                    "any_fault_by_view": {k: pr(*v) for k, v in by_view_posture.items()}},
        "rows": rows,
    }


def false_rep_matrix(metas: list[dict], results_by_sel: dict[str, dict], gate=None) -> dict:
    """Reps counted when the user selected exercise S but did activity A != S (A may be non-exercise).
    Returns reps per minute of footage for every (selected, actual) cell."""
    cells = defaultdict(lambda: [0, 0.0])
    for sel, res in results_by_sel.items():
        for m in metas:
            if m["label"] == sel:
                continue
            r = res.get(m["id"] + f"@{sel}")
            if r is None:
                continue
            reps = r["reps"] if gate is None else [p for p in r["reps"] if gate(m, sel, p)]
            key = (sel, m["activity"])
            cells[key][0] += len(reps)
            cells[key][1] += m["n_frames"] / m["fps"] / 60
    out = {f"{s}|{a}": round(n / mins, 2) for (s, a), (n, mins) in cells.items()}
    tot_n = sum(v[0] for v in cells.values())
    tot_m = sum(v[1] for v in cells.values())
    dis = [(n, mi) for (s, a), (n, mi) in cells.items() if a not in EXERCISES]
    return {"per_cell_reps_per_min": out, "overall_reps_per_min": round(tot_n / max(tot_m, 1e-9), 3),
            "non_exercise_reps_per_min": round(sum(n for n, _ in dis) / max(sum(mi for _, mi in dis), 1e-9), 3)}
