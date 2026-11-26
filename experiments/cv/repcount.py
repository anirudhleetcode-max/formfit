"""Rep counting, false-repetition analysis and posture evaluation through the app's TypeScript engine.

All on synthetic 3-D skeleton sequences (backend/cv/skeleton.py) with exact ground truth:
  * counting on `test` and `test_hard`, with the landmark guard (pose/guard.ts) off and on;
  * counting and posture precision/recall per camera yaw (`sweep`, 8 yaws x 5 exercises);
  * false repetitions: every sequence is also run with every OTHER exercise selected (the user
    picked "squat" but is lunging, marching, waving ...), reported as reps per minute, without and
    with the temporal-classifier gate (a rep is dropped when the TCN is confident, p >= tau, that the
    2 s before it were a different movement);
  * the gate's cost: true reps lost on correctly selected exercises.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cv.classify import CLS_IDX, WIN, load_tcn, tcn_predict  # noqa: E402
from cv.dataset import DATA, load_meta  # noqa: E402
from cv.engine_eval import evaluate_counting, false_rep_matrix, jobs_for, run_engine  # noqa: E402
from cv.normalize import frame_features, frames_to_array  # noqa: E402
from cv.skeleton import EXERCISES  # noqa: E402

TCN = ROOT / "backend" / "ml" / "artifacts" / "exercise_tcn.npz"


class Gate:
    """Veto a counted rep when the temporal classifier is confident the window before it is another class."""

    def __init__(self, dataset: str, tau: float = 0.6):
        self.model = load_tcn(TCN)
        self.dataset, self.tau = dataset, tau
        self.cache: dict[str, np.ndarray] = {}
        self.checked = self.vetoed = 0

    def probs(self, meta: dict, t: float) -> np.ndarray:
        f = self.cache.get(meta["id"])
        if f is None:
            s = json.loads((DATA / self.dataset / f"{meta['id']}.json").read_text())
            lm, present = frames_to_array(s["frames"])
            f = frame_features(lm, present, s["width"] / s["height"])
            self.cache = {meta["id"]: f}
        end = min(len(f), int(round(t * meta["fps"])) + 1)
        st = max(0, end - WIN)
        w = f[st:end]
        if len(w) < WIN:
            w = np.concatenate([np.repeat(w[:1], WIN - len(w), 0), w])
        return tcn_predict(self.model, w[None])[0]

    def __call__(self, meta: dict, selected: str, rep: dict) -> bool:
        p = self.probs(meta, rep["t"])
        self.checked += 1
        top = int(p.argmax())
        veto = bool(top != CLS_IDX[selected] and p[top] >= self.tau)
        self.vetoed += int(veto)
        return not veto


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "rows"}


def run(cfg: dict, out: Path, log=print) -> dict:
    rep: dict = {"engine": "frontend/src/pose/engine.ts via scripts/eval-synthetic.ts", "data": "synthetic"}
    test, hard, sweep = load_meta("test"), load_meta("test_hard"), load_meta("sweep")

    for name, metas in (("test", test), ("test_hard", hard)):
        for g in (False, True):
            res = run_engine(jobs_for(name, metas, guard=g))
            r = evaluate_counting(name, metas, res)
            rep[f"{name}_guard_{'on' if g else 'off'}"] = _strip(r) | {"engine_ms_per_frame": round(res["_ms_per_frame"], 4)}
            log(f"{name} guard={g}: {r['counting']}")
            if name == "test" and g:
                (out / "per_sequence_test.json").write_text(json.dumps(r["rows"]))

    # camera yaw sweep (guard on)
    res = run_engine(jobs_for("sweep", sweep, guard=True))
    by_yaw = {}
    for yaw in range(0, 360, 45):
        ms = [m for m in sweep if round(m["camera"]["yaw"]) % 360 == yaw]
        r = evaluate_counting("sweep", ms, res)
        by_yaw[str(yaw)] = {"counting": r["counting"], "by_exercise_mae": {e: r["counting_by_exercise"][e].get("mae")
                                                                            for e in EXERCISES},
                            "posture_any_fault": r["posture"]["any_fault"], "posture_coverage": r["posture"]["coverage"]}
    rep["sweep_by_yaw"] = by_yaw

    # false repetitions: every exercise selected on every sequence of another activity
    log("false-rep matrix ...")
    per_sel = {}
    for sel in EXERCISES:
        jobs = [j | {"id": j["id"] + f"@{sel}"} for j in jobs_for("test", [m for m in test if m["label"] != sel],
                                                                   guard=True, selected=sel)]
        per_sel[sel] = run_engine(jobs)
    taus = cfg.get("gate_taus", [0.5, 0.6, 0.8, 0.9])
    rep["false_reps_no_gate"] = false_rep_matrix(test, per_sel)
    log(f"no gate: {rep['false_reps_no_gate']['overall_reps_per_min']} reps/min "
        f"(non-exercise {rep['false_reps_no_gate']['non_exercise_reps_per_min']})")
    rep["false_reps_gate"] = {}
    rep["gate_cost_on_correct_selection"] = {}
    base = run_engine(jobs_for("test", test, guard=True))
    base_eval = evaluate_counting("test", test, base)["counting"]
    for tau in taus:
        gate = Gate("test", tau)
        rep["false_reps_gate"][str(tau)] = false_rep_matrix(test, per_sel, gate=gate)
        g2 = Gate("test", tau)
        r = evaluate_counting("test", test, base, gate=g2)["counting"]
        rep["gate_cost_on_correct_selection"][str(tau)] = {
            "rep_recall_before": base_eval["rep_recall"], "rep_recall_after": r["rep_recall"],
            "mae_before": base_eval["mae"], "mae_after": r["mae"], "vetoed": g2.vetoed, "checked": g2.checked}
        log(f"gate tau={tau}: {rep['false_reps_gate'][str(tau)]['overall_reps_per_min']} reps/min, "
            f"recall {base_eval['rep_recall']} -> {r['rep_recall']}")
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep
