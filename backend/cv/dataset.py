"""Generate synthetic sequence datasets to data/synthetic/<name>/ (git-ignored).

    python -m cv.dataset --name train --n 1200 --seed 101
    python -m cv.dataset --name sweep --sweep --per-cell 12 --seed 303

Each sequence is stored as <id>.json in the landmark format the TypeScript evaluation scripts read
({width, height, fps, frames}) plus the ground truth; meta.jsonl has one line of metadata per
sequence (no frames) for fast loading.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .skeleton import EXERCISES, make_sequence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"
META_KEYS = ["id", "label", "activity", "width", "height", "fps", "camera", "body_height_m", "observation",
             "gt_reps", "reps"]


def _one(args):
    out_dir, seq_id, seed, cls, yaw, kw = args
    rng = np.random.default_rng(seed)
    s = make_sequence(seq_id, rng, cls, yaw, **kw)
    s.pop("clean")
    (Path(out_dir) / f"{seq_id}.json").write_text(json.dumps(s, separators=(",", ":")))
    meta = {k: s[k] for k in META_KEYS}
    meta["n_frames"] = len(s["frames"])
    meta["active_frac"] = float(np.mean(s["active"]))
    return meta


def generate(name: str, n: int = 0, seed: int = 0, sweep: bool = False, per_cell: int = 10,
             occlusion_p: float = 0.25, workers: int = 4, noise: tuple[float, float] = (0.003, 0.01)) -> Path:
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if sweep:  # every exercise x 8 camera yaws, no random occluders: isolates the view effect
        i = 0
        for ex in EXERCISES:
            for yaw in range(0, 360, 45):
                for _ in range(per_cell):
                    jobs.append((str(out), f"{name}_{i:05d}", seed * 100_000 + i, ex, float(yaw),
                                 {"occlusion_p": 0.0}))
                    i += 1
    else:
        for i in range(n):
            jobs.append((str(out), f"{name}_{i:05d}", seed * 100_000 + i, None, None,
                         {"occlusion_p": occlusion_p, "noise": tuple(noise)}))
    with ProcessPoolExecutor(workers) as ex:
        metas = list(ex.map(_one, jobs, chunksize=8))
    with (out / "meta.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    (out / "dataset.json").write_text(json.dumps({"name": name, "n": len(jobs), "seed": seed, "sweep": sweep,
                                                  "occlusion_p": occlusion_p, "noise": list(noise)}, indent=1))
    return out


def load_meta(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name / "meta.jsonl").read_text().splitlines()]


def load_seq(name: str, seq_id: str) -> dict:
    return json.loads((DATA / name / f"{seq_id}.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--occlusion-p", type=float, default=0.25)
    ap.add_argument("--noise", type=float, nargs=2, default=(0.003, 0.01), help="pixel noise range, x body height")
    a = ap.parse_args()
    p = generate(a.name, a.n, a.seed, a.sweep, a.per_cell, a.occlusion_p, noise=tuple(a.noise))
    print(f"wrote {p}")
