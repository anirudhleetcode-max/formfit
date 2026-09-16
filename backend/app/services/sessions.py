"""Builds the stored session document from raw reps (shared by the API and the seed script)."""
from __future__ import annotations

from collections import Counter

from .analytics import detect_fatigue


def _avg(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 1) if xs else None


def build_finished(reps: list[dict], model_out: list) -> dict:
    """reps: dicts with set, score (None = not scored), faults, ecc_s, con_s, rom, t, features,
    confidence, scored (performed order). model_out: per-rep dicts from FormModel.predict (or plain
    model scores, older callers)."""
    order = sorted(range(len(reps)), key=lambda k: (reps[k]["set"], reps[k]["t"]))
    out_reps = []
    for i, k in enumerate(order, start=1):
        r, mo = reps[k], model_out[k]
        if not isinstance(mo, dict):
            mo = {"model_score": mo, "model_confidence": None,
                  "model_status": "ok" if mo is not None else "unavailable", "model_ood": []}
        r = {**r, "scored": r.get("scored", r.get("score") is not None)}
        out_reps.append({**r, "i": i, **mo})

    sets: dict[int, list[dict]] = {}
    for r in out_reps:
        sets.setdefault(r["set"], []).append(r)
    set_rows = [
        {"set": s, "reps": len(rs), "avg_score": _avg([x["score"] for x in rs if x["score"] is not None]),
         "scored_reps": sum(1 for x in rs if x["scored"])}
        for s, rs in sorted(sets.items())
    ]
    n = len(out_reps)
    scored = [r for r in out_reps if r["scored"]]
    ms = [r["model_score"] for r in out_reps if r["model_score"] is not None]
    confs = [r["confidence"] for r in out_reps if r.get("confidence") is not None]
    faults = Counter(f for r in out_reps for f in r["faults"])
    statuses = Counter(r["model_status"] for r in out_reps)
    summary = {
        "total_reps": n,
        "sets": len(set_rows),
        "avg_score": _avg([r["score"] for r in scored]),
        "avg_model_score": _avg(ms),
        "best_set_reps": max((s["reps"] for s in set_rows), default=0),
        "clean_reps": sum(1 for r in scored if not r["faults"]),
        "scored_reps": len(scored),
        "unscored_reps": n - len(scored),
        "avg_confidence": round(sum(confs) / len(confs), 2) if confs else None,
        "model_status_counts": dict(statuses),
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
