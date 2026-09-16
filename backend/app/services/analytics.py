"""Server-side session analytics: fatigue detection and progress trends (pure numpy)."""
from __future__ import annotations

import numpy as np

MIN_REPS_FOR_FATIGUE = 6
FATIGUE_SHIFT = 0.12   # mean fatigue-index jump (after vs before) needed to flag fatigue
FATIGUE_T = 3.0        # Welch t-statistic of that jump (strict: the split point is searched)


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def fatigue_index(con_s, rom, score) -> np.ndarray:
    """Per-rep fatigue index relative to the first reps of the session.

    Fatigue shows up as a slower lifting (concentric) phase, a shrinking range of motion and
    worse form. Each signal is expressed as a relative change vs the baseline (median of the
    first 3 reps), signed so that 'more tired' is positive, and the three are averaged."""
    con = np.asarray(con_s, float)
    rom = np.asarray(rom, float)
    sc = np.asarray(score, float)
    k = min(3, len(con))
    b_con = max(np.median(con[:k]), 0.2)
    b_rom = max(np.median(rom[:k]), 5.0)
    b_sc = np.median(sc[:k])
    parts = np.vstack([
        np.clip((con - b_con) / b_con, -1, 2),
        np.clip((b_rom - rom) / b_rom, -1, 1),
        np.clip((b_sc - sc) / 100.0, -1, 1),
    ])
    return parts.mean(axis=0)


def change_point(y: np.ndarray, min_seg: int = 3) -> tuple[int, float, float]:
    """Best single mean-shift change point (least squares). Returns (index, shift, t-stat)."""
    n = len(y)
    best_k, best_sse = -1, np.inf
    for k in range(min_seg, n - min_seg + 1):
        a, b = y[:k], y[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best_sse:
            best_k, best_sse = k, sse
    if best_k < 0:
        return -1, 0.0, 0.0
    a, b = y[:best_k], y[best_k:]
    shift = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) + 1e-6
    return best_k, shift, shift / se


def detect_fatigue(reps: list[dict]) -> dict:
    """reps: in performed order, each with con_s, rom, score."""
    n = len(reps)
    out = {"detected": False, "onset_rep": None, "shift": 0.0, "min_reps": MIN_REPS_FOR_FATIGUE,
           "slopes": {"con_s": 0.0, "rom": 0.0, "score": 0.0}, "index": []}
    if n == 0:
        return out
    con = [r["con_s"] for r in reps]
    rom = [r["rom"] for r in reps]
    sc = [r["score"] for r in reps]
    idx = fatigue_index(con, rom, sc)
    out["index"] = [round(float(v), 3) for v in idx]
    out["slopes"] = {"con_s": round(_slope(np.array(con)), 4), "rom": round(_slope(np.array(rom)), 3),
                     "score": round(_slope(np.array(sc, float)), 3)}
    if n < MIN_REPS_FOR_FATIGUE:
        return out
    k, shift, t = change_point(idx)
    out["shift"] = round(shift, 3)
    out["t_stat"] = round(t, 2)
    # Flag only a sustained, statistically clear upward shift that the overall trend agrees with.
    if k > 0 and shift >= FATIGUE_SHIFT and t >= FATIGUE_T and _slope(idx) > 0:
        out["detected"] = True
        out["onset_rep"] = k + 1  # 1-based rep number where the tired segment starts
    return out


def trend_per_week(days: list[float], values: list[float]) -> float | None:
    """Least-squares slope of values over time, in units per week."""
    if len(days) < 3 or max(days) - min(days) < 3:
        return None
    return round(float(np.polyfit(np.asarray(days), np.asarray(values, float), 1)[0] * 7), 2)
