"""Run one experiment from a JSON config and record it under experiments/results/<run_id>/.

    python -m experiments.run --config experiments/configs/form_classifier_synthetic.json
    python -m experiments.run --config experiments/configs/fatigue_synthetic.json
    python -m experiments.run --config experiments/configs/real_clips_repcount.json

Every run writes: config.json (snapshot), env.json (git commit, library versions, timestamp),
metrics.json, and task-specific files (errors.csv / errors.json / per_clip.json / model_card.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))  # ml/ and app/ live in backend/

from ml.runinfo import env_info  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--save-model", action="store_true", help="form_classifier: also replace the served artifact")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg['name']}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "env.json").write_text(json.dumps(env_info(), indent=2))

    task = cfg["task"]
    if task == "form_classifier":
        from ml.train import train
        report = train(cfg, out, save_model=args.save_model)
    elif task == "fatigue":
        from ml.eval_fatigue import run as run_fatigue
        report = run_fatigue(cfg)
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
    elif task == "real_clip_repcount":
        from experiments.real_clips.evaluate import run as run_clips
        report = run_clips(cfg, out)
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
    else:
        raise SystemExit(f"unknown task {task}")
    print(f"run {run_id} -> {out.relative_to(ROOT)}")
    if task == "real_clip_repcount":
        print(json.dumps(report["overall"], indent=1))


if __name__ == "__main__":
    main()
