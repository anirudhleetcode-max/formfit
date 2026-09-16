"""Builds the stored session document from raw reps (shared by the API and the seed script)."""
from __future__ import annotations

from collections import Counter

from .analytics import detect_fatigue


def build_finished(reps: list[dict], model_scores: list[int | None]) -> dict:
    """reps: dicts with set, score, faults, ecc_s, con_s, rom, t, features (performed order)."""
    reps = sorted(reps, key=lambda r: (r["set"], r["t"]))
    out_reps = []
    for i, (r, ms) in enumerate(zip(reps, model_scores), start=1):
        out_reps.append({**r, "i": i, "model_score": ms})

    sets: dict[int, list[dict]] = {}
    for r in out_reps:
        sets.setdefault(r["set"], []).append(r)
    set_rows = [
        {"set": s, "reps": len(rs), "avg_score": round(sum(x["score"] for x in rs) / len(rs), 1)}
        for s, rs in sorted(sets.items())
    ]
    n = len(out_reps)
    ms = [r["model_score"] for r in out_reps if r["model_score"] is not None]
    faults = Counter(f for r in out_reps for f in r["faults"])
    summary = {
        "total_reps": n,
        "sets": len(set_rows),
        "avg_score": round(sum(r["score"] for r in out_reps) / n, 1) if n else None,
        "avg_model_score": round(sum(ms) / len(ms), 1) if ms else None,
        "best_set_reps": max((s["reps"] for s in set_rows), default=0),
        "clean_reps": sum(1 for r in out_reps if not r["faults"]),
        "avg_ecc_s": round(sum(r["ecc_s"] for r in out_reps) / n, 2) if n else None,
        "avg_con_s": round(sum(r["con_s"] for r in out_reps) / n, 2) if n else None,
        "top_faults": [{"code": c, "count": k} for c, k in faults.most_common(3)],
    }
    return {
        "reps": out_reps,
        "sets": set_rows,
        "summary": summary,
        "fatigue": detect_fatigue(out_reps),
    }
