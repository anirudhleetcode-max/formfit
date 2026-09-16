"""Measure the fatigue detector on simulated sets with known fatigue patterns.

    python -m ml.eval_fatigue     # writes ml/artifacts/fatigue_eval.json
"""
import json
from pathlib import Path

import numpy as np

from app.services.analytics import detect_fatigue
from ml import rules
from ml.simulate import simulate_rep

PATTERNS = {
    "no_fatigue": lambda i, n: 0.0,
    "mild_progressive": lambda i, n: (i / n) ** 2 * 0.5,
    "strong_progressive": lambda i, n: (i / n) ** 2 * 1.0,
    "sudden_after_rep_8": lambda i, n: 0.9 if i > 7 else 0.0,
}


def run(trials: int = 300, reps: int = 12, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    out = {"trials": trials, "reps_per_set": reps, "exercise": "squat", "flag_rate": {}}
    for name, fn in PATTERNS.items():
        hits = 0
        for _ in range(trials):
            rs = []
            for i in range(reps):
                f, _, _ = simulate_rep("squat", rng, fatigue=fn(i, reps), fault_rate=0.1, view="side")
                score, _ = rules.evaluate("squat", f)
                rs.append({"con_s": f["con_s"], "rom": f["rom"], "score": score})
            hits += detect_fatigue(rs)["detected"]
        out["flag_rate"][name] = round(hits / trials, 3)
        print(f"{name:22s} flagged {hits / trials:.1%}")
    path = Path(__file__).resolve().parent / "artifacts" / "fatigue_eval.json"
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
