"""Real-video rep-count evaluation through the app's TypeScript pipeline.

    python -m experiments.run --config experiments/configs/real_clips_repcount.json

Steps: fetch clips (manifest) -> extract landmarks (Python MediaPipe, 15 fps, cached) ->
`npx tsx frontend/scripts/eval-clips.ts` (src/pose/engine.ts) -> metrics.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
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
    by_id = {c["id"]: c for c in manifest["clips"]}
    for r in rows:
        c = by_id[r["id"]]
        r.update({"license": c["license"], "view": c.get("view"), "gt_method": c.get("gt_method"),
                  "status": status[r["id"]]})
    per_ex = defaultdict(list)
    for r in rows:
        per_ex[r["exercise"]].append(r)
    table = [{k: r.get(k) for k in ("id", "exercise", "view", "gt_reps", "pred_reps", "scored_reps",
                                     "mean_frame_confidence", "frames", "frames_with_pose", "status")} for r in rows]
    report = {
        "dataset": {"name": "formfit-real-clips", "synthetic": False, "manifest_version": manifest.get("version"),
                    "manifest_sha256": sha256(MANIFEST), "n_clips_listed": len(manifest["clips"])},
        "analysis_fps": fps,
        "overall": _metrics(rows),
        "per_exercise": {ex: _metrics(rs) for ex, rs in sorted(per_ex.items())},
        "clips": table,
        "note": "Tiny sample of public Wikimedia Commons clips; ground truth counted by hand (see manifest gt_method).",
    }
    # error file: every clip whose count is wrong
    errs = [r for r in rows if r.get("gt_reps") is not None and r.get("pred_reps") != r.get("gt_reps")]
    (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
    return report
