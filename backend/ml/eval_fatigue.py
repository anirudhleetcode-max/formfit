"""Measure the fatigue detector on simulated sets with known fatigue patterns (SYNTHETIC).

    python -m ml.eval_fatigue                     # writes ml/artifacts/fatigue_eval.json
    python -m experiments.run --config experiments/configs/fatigue_synthetic.json

The detector thresholds (shift >= 0.12, t >= 3.0) were chosen while looking at simulated sets
generated with seed 3 ("tuning seed"). The reported rates use a different seed (default 2024),
so the false-alarm rate is not measured on the data the thresholds were picked from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services.analytics import FATIGUE_SHIFT, FATIGUE_T, detect_fatigue
from ml import rules
from ml.simulate import SIMULATOR_VERSION, simulate_rep

PATTERNS = {
    "no_fatigue": lambda i, n: 0.0,
    "mild_progressive": lambda i, n: (i / n) ** 2 * 0.5,
    "strong_progressive": lambda i, n: (i / n) ** 2 * 1.0,
    "sudden_after_rep_8": lambda i, n: 0.9 if i > 7 else 0.0,
}

DEFAULT = {"name": "fatigue_synthetic", "task": "fatigue", "trials": 300, "reps_per_set": 12,
           "exercises": ["squat"], "seed": 2024, "tuning_seed": 3}


def run(cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT, **(cfg or {})}
    rng = np.random.default_rng(cfg["seed"])
    out = {"synthetic": True, "simulator_version": SIMULATOR_VERSION,
           "detector": {"shift_min": FATIGUE_SHIFT, "t_min": FATIGUE_T, "min_reps": 6},
           "config": cfg, "flag_rate": {}, "onset_error": {}}
    n = cfg["reps_per_set"]
    for ex in cfg["exercises"]:
        for name, fn in PATTERNS.items():
            hits, onset_err = 0, []
            for _ in range(cfg["trials"]):
                rs = []
                for i in range(n):
                    f, _, _ = simulate_rep(ex, rng, fatigue=fn(i, n), fault_rate=0.1, view="side")
                    score, _ = rules.evaluate(ex, f)
                    rs.append({"con_s": f["con_s"], "rom": f["rom"], "score": score})
                d = detect_fatigue(rs)
                hits += d["detected"]
                if d["detected"] and name == "sudden_after_rep_8":
                    onset_err.append(abs(d["onset_rep"] - 9))  # true tired segment starts at rep 9
            key = f"{ex}/{name}" if len(cfg["exercises"]) > 1 else name
            out["flag_rate"][key] = round(hits / cfg["trials"], 3)
            if onset_err:
                out["onset_error"][key] = {"mean_abs_reps": round(float(np.mean(onset_err)), 2),
                                           "exact": round(float(np.mean([e == 0 for e in onset_err])), 3)}
            print(f"{key:28s} flagged {hits / cfg['trials']:.1%}")
    out["false_alarm_rate"] = _mean_rate(out["flag_rate"], "no_fatigue")
    out["detection_rate"] = {p: _mean_rate(out["flag_rate"], p) for p in PATTERNS if p != "no_fatigue"}
    return out


def _mean_rate(flag_rate: dict, pattern: str) -> float | None:
    """Average flag rate for one pattern over exercises (keys are `pattern` or `exercise/pattern`)."""
    vals = [v for k, v in flag_rate.items() if k.split("/")[-1] == pattern]
    return round(float(np.mean(vals)), 3) if vals else None


if __name__ == "__main__":
    res = run()
    path = Path(__file__).resolve().parent / "artifacts" / "fatigue_eval.json"
    path.write_text(json.dumps(res, indent=2))
