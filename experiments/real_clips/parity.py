"""Browser vs Python landmark parity on the bundled sample clip.

The real-clip evaluation extracts landmarks with MediaPipe's Python build. The app runs the WASM
build of the same model in the browser. This check compares the two on the same clip:
  - the e2e run saves the browser's raw landmarks from Upload mode to e2e/.cache/upload_trace.json
  - this script compares them frame by frame with the Python landmarks and replays both through
    the TypeScript engine (frontend/scripts/eval-clips.ts)

    cd e2e && ./run_e2e.sh
    python -m experiments.run --config experiments/configs/browser_parity.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .evaluate import _landmarks_for
from .fetch import ROOT, load_manifest

BODY = slice(11, 29)  # shoulders .. heels (the joints the app uses)


def run(cfg: dict, out_dir: Path) -> dict:
    trace_path = ROOT / cfg["browser_trace"]
    if not trace_path.exists():
        raise SystemExit(f"{trace_path} not found: run e2e/run_e2e.sh first")
    clip = next(c for c in load_manifest()["clips"] if c["id"] == cfg["clip_id"])
    py_path = _landmarks_for(clip, cfg.get("analysis_fps", 15.0))
    py = json.loads(py_path.read_text())
    br = json.loads(trace_path.read_text())
    br = {"width": py["width"], "height": py["height"], "fps": py["fps"],
          "frames": [{"t": round(f["t"], 3), "lm": f["lm"]} for f in br["frames"]]}

    n = min(len(br["frames"]), len(py["frames"]))
    diffs, both_missing, one_missing = [], 0, 0
    for a, b in zip(br["frames"][:n], py["frames"][:n]):
        if a["lm"] is None or b["lm"] is None:
            both_missing += a["lm"] is None and b["lm"] is None
            one_missing += (a["lm"] is None) != (b["lm"] is None)
            continue
        A, B = np.asarray(a["lm"])[BODY, :2], np.asarray(b["lm"])[BODY, :2]
        diffs.append(float(np.abs(A - B).mean()))
    d = np.asarray(diffs)

    # replay both landmark sets through the same TS engine
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "browser.json").write_text(json.dumps(br))
        (tmp / "python.json").write_text(json.dumps(py))
        man = {"clips": [{"id": k, "exercise": clip["exercise"], "gt_reps": clip["gt_reps"]} for k in ("browser", "python")]}
        (tmp / "manifest.json").write_text(json.dumps(man))
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        subprocess.run([npx, "tsx", "scripts/eval-clips.ts", str(tmp / "manifest.json"), str(tmp), str(tmp / "out.json")],
                       cwd=ROOT / "frontend", check=True, timeout=300)
        replay = {r["id"]: r for r in json.loads((tmp / "out.json").read_text())}

    def rep_summary(r: dict) -> dict:
        return {"pred_reps": r["pred_reps"], "scored_reps": r["scored_reps"],
                "min_angles": [x["min_angle"] for x in r["reps"]], "scores": [x["score"] for x in r["reps"]]}

    return {
        "clip": clip["id"], "gt_reps": clip["gt_reps"],
        "frames": {"browser": len(br["frames"]), "python": len(py["frames"]), "compared": len(diffs),
                   "pose_missing_in_one": one_missing, "pose_missing_in_both": both_missing},
        "landmark_abs_diff_normalised": {"mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                                         "p95": round(float(np.percentile(d, 95)), 4), "max": round(float(d.max()), 4)},
        "replay": {"browser": rep_summary(replay["browser"]), "python": rep_summary(replay["python"])},
        "note": "Browser = MediaPipe WASM (CPU delegate, headless Chromium) in Upload mode; Python = mediapipe "
                f"{_mp_version()}. Same pose_landmarker_lite.task, 15 analysed fps. Differences are in units of frame width/height.",
    }


def _mp_version() -> str:
    try:
        import mediapipe
        return mediapipe.__version__
    except Exception:  # noqa: BLE001
        return "unknown"
