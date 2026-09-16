"""Real-video rep-count evaluation through the app's TypeScript pipeline.

    python -m experiments.run --config experiments/configs/real_clips_repcount.json

Steps: fetch clips (manifest) -> extract landmarks (Python MediaPipe, 15 fps, cached) ->
`npx tsx frontend/scripts/eval-clips.ts` (src/pose/engine.ts) -> metrics.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .extract import LANDMARKS, extract
from .fetch import MANIFEST, ROOT, fetch, load_manifest, sha256


def _landmarks_for(clip: dict, fps: float) -> Path | None:
    dst = LANDMARKS / f"{clip['id']}.json"
    src = fetch(clip)
    if src is None:
        return dst if dst.exists() else None
    digest = sha256(src)
    if dst.exists():
        meta = json.loads(dst.read_text())
        if meta.get("source_sha256") == digest and meta.get("fps") == fps and meta.get("segment") == clip.get("segment"):
            return dst
    print(f"[extract] {clip['id']} ...")
    extract(src, dst, fps=fps, segment=clip.get("segment"), source_sha256=digest)
    return dst


def _metrics(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("gt_reps") is not None and "pred_reps" in r]
    if not rows:
        return {"n_clips": 0}
    err = [r["pred_reps"] - r["gt_reps"] for r in rows]
    gt_total = sum(r["gt_reps"] for r in rows)
    return {
        "n_clips": len(rows),
        "gt_reps_total": gt_total,
        "pred_reps_total": sum(r["pred_reps"] for r in rows),
        "mae": round(sum(abs(e) for e in err) / len(err), 3),
        "mean_signed_error": round(sum(err) / len(err), 3),
        "exact_match_rate": round(sum(e == 0 for e in err) / len(err), 3),
        "within_1_rate": round(sum(abs(e) <= 1 for e in err) / len(err), 3),
        "relative_count_error": round(sum(abs(e) for e in err) / max(1, gt_total), 3),
        "scored_share_of_predicted": round(sum(r["scored_reps"] for r in rows) / max(1, sum(r["pred_reps"] for r in rows)), 3),
    }


def _model_scores(rows: list[dict]) -> None:
    """Score the real reps with the served (synthetic-trained) classifier, as the API would."""
    from app.services.form_model import BACKEND_DIR, FormModel
    model = FormModel(BACKEND_DIR / "ml" / "artifacts" / "form_model.joblib")
    model.load()
    for r in rows:
        reps = r.get("reps") or []
        preds = model.predict(r["exercise"], [x["api_features"] for x in reps], [x["scored"] for x in reps]) if reps else []
        for x, p in zip(reps, preds):
            x.update(p)
        r["model_status_counts"] = dict(Counter(p["model_status"] for p in preds))


def _fault_agreement(labels: list[list[str]] | None, reps: list[dict]) -> dict | None:
    """Rep-level clean/faulty agreement of the rule verdict with hand labels (only when the counts match,
    so rep i is the same physical rep; unscored reps are skipped)."""
    if not labels or len(labels) != len(reps):
        return None
    pairs = [(not lab, not x["faults"]) for lab, x in zip(labels, reps) if x["scored"]]
    return {"compared": len(pairs), "agree": sum(a == b for a, b in pairs)}


def _model_summary(rows: list[dict]) -> dict:
    reps = [x for r in rows for x in r.get("reps") or []]
    counts = Counter(x.get("model_status") for x in reps)
    ood = Counter(f for x in reps for f in x.get("model_ood") or [])
    return {"reps": len(reps), "status_counts": dict(counts), "ood_features": dict(ood.most_common()),
            "note": "Real reps scored by the classifier trained on SYNTHETIC data; no fault labels, so no accuracy."}


def run(cfg: dict, out_dir: Path) -> dict:
    manifest = load_manifest()
    fps = cfg.get("analysis_fps", 15.0)
    status = {}
    for clip in manifest["clips"]:
        p = _landmarks_for(clip, fps)
        status[clip["id"]] = "ok" if p else "unavailable"
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RuntimeError("npx not found: install Node.js and run `npm install` in frontend/")
    raw = out_dir / "per_clip.json"
    subprocess.run([npx, "tsx", "scripts/eval-clips.ts", str(MANIFEST), str(LANDMARKS), str(raw)],
                   cwd=ROOT / "frontend", check=True, timeout=600)
    rows = json.loads(raw.read_text())
    _model_scores(rows)
    by_id = {c["id"]: c for c in manifest["clips"]}
    for r in rows:
        c = by_id[r["id"]]
        r.update({"license": c["license"], "view": c.get("view"), "gt_method": c.get("gt_method"),
                  "status": status[r["id"]], "source_type": c.get("source_type"),
                  "conditions": c.get("conditions", []), "gt_verified_by_hand": c.get("gt_verified_by_hand")})
        r["fault_agreement"] = _fault_agreement(c.get("fault_labels"), r.get("reps") or [])
    raw.write_text(json.dumps(rows, indent=1))
    per_ex = defaultdict(list)
    for r in rows:
        per_ex[r["exercise"]].append(r)
    table = [{k: r.get(k) for k in ("id", "exercise", "view", "gt_reps", "pred_reps", "scored_reps",
                                     "mean_frame_confidence", "frames", "frames_with_pose", "model_status_counts",
                                     "source_type", "conditions", "gt_verified_by_hand",
                                     "status")} for r in rows]
    report = {
        "dataset": {"name": "formfit-real-clips", "synthetic": False, "manifest_version": manifest.get("version"),
                    "manifest_sha256": sha256(MANIFEST), "n_clips_listed": len(manifest["clips"])},
        "analysis_fps": fps,
        "overall": _metrics(rows),
        "per_exercise": {ex: _metrics(rs) for ex, rs in sorted(per_ex.items())},
        "per_source_type": {st: _metrics([r for r in rows if r.get("source_type") == st]) for st in ("filmed", "rendered")},
        "hand_verified_only": _metrics([r for r in rows if r.get("gt_verified_by_hand")]),
        "model_on_real_reps": _model_summary(rows),
        "clips": table,
        "note": "Tiny sample: 1 Wikimedia Commons clip + 10 PushUpBench clips (5 filmed people, 5 rendered 3D avatars). Ground truth per manifest gt_method.",
    }
    # compact copy served by /api/model so the UI can show the measured numbers
    summary = {"run_id": out_dir.name, "dataset": report["dataset"], "overall": report["overall"],
               "per_exercise": report["per_exercise"], "per_source_type": report["per_source_type"],
               "hand_verified_only": report["hand_verified_only"], "model_on_real_reps": report["model_on_real_reps"],
               "note": report["note"]}
    (ROOT / "backend" / "ml" / "artifacts" / "real_clips_eval.json").write_text(json.dumps(summary, indent=2))
    # error file: every clip whose count is wrong
    errs = [r for r in rows if r.get("gt_reps") is not None and r.get("pred_reps") != r.get("gt_reps")]
    (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
    return report
