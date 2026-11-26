"""Landmark-level experiments on REAL video frames: model variants, latency, low light, occlusion.

There is no keypoint ground truth for the bundled clip, so accuracy here is AGREEMENT with a
reference: MediaPipe's largest model (`heavy`) on the unmodified frames. That measures how much the
app's `lite` model and each image degradation move the landmarks; it does not measure absolute
accuracy (which needs labelled keypoints, see experiments/cv/coco_pck.py).

Steps
1. Decode the clip at 15 fps (as Upload mode does).
2. lite / full / heavy in VIDEO mode on clean frames: detection rate, latency per frame (CPU),
   PCK@{0.05,0.1,0.2} x torso length and NME on 12 body joints against heavy.
3. Low light: brightness x {0.5, 0.25, 0.12, 0.06} with gamma, shot + read noise and 8-bit
   quantisation; each level evaluated raw, with the browser-feasible fix (linear gain to a
   target mean luminance, what a canvas `brightness()` filter can do) and with an offline fix
   (adaptive gamma + CLAHE on the L channel).
4. Occlusion: grey boxes over a body joint for bursts of frames (a dumbbell or a passer-by).
   Landmark error on the covered joints vs the rest, and rep count with the landmark guard off/on.
5. Every condition's landmarks go through the app's TypeScript rep engine (squat, guard off and on).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cv.normalize import BODY12  # noqa: E402

MODELS = {
    "lite": ROOT / "frontend" / "public" / "models" / "pose_landmarker_lite.task",
    "full": ROOT / "data" / "models" / "pose_landmarker_full.task",
    "heavy": ROOT / "data" / "models" / "pose_landmarker_heavy.task",
}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
CLIP = ROOT / "samples" / "squat_demo.webm"
GT_REPS = 4


def ensure_models():
    import urllib.request
    for v, p in MODELS.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL.format(v=v), p)


def read_frames(path: Path, fps=15.0) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out, i, next_t = [], 0, 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts = i / src_fps
        i += 1
        if ts + 1e-6 < next_t:
            continue
        next_t += 1 / fps
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return out


def run_model(model: str, frames: list[np.ndarray], fps=15.0):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELS[model])),
                                        running_mode=vision.RunningMode.VIDEO)
    lm = vision.PoseLandmarker.create_from_options(opts)
    out, ms = [], []
    for k, fr in enumerate(frames):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr))
        t = time.perf_counter()
        r = lm.detect_for_video(img, int(k * 1000 / fps))
        ms.append((time.perf_counter() - t) * 1000)
        out.append([[p.x, p.y, p.visibility] for p in r.pose_landmarks[0]] if r.pose_landmarks else None)
    lm.close()
    return out, ms


def agreement(pred: list, ref: list, h_over_w: float, joints=BODY12, mask=None) -> dict:
    """PCK / NME of pred vs ref on frames where both exist. Units: torso length of the reference."""
    errs, n_ref, n_both = [], 0, 0
    for k, (p, r) in enumerate(zip(pred, ref)):
        if r is None:
            continue
        n_ref += 1
        if p is None:
            continue
        n_both += 1
        P, R = np.array(p)[:, :2], np.array(r)[:, :2]
        P[:, 0] /= h_over_w; R[:, 0] /= h_over_w  # aspect-correct (x in image-height units)
        torso = np.linalg.norm((R[11] + R[12]) / 2 - (R[23] + R[24]) / 2) + 1e-6
        js = joints if mask is None else [j for j in joints if mask[k][j]]
        errs.extend((np.linalg.norm(P[js] - R[js], axis=1) / torso).tolist())
    e = np.array(errs) if errs else np.array([np.nan])
    return {"detection_rate": round(n_both / max(1, n_ref), 3), "n_joint_obs": len(errs),
            "pck@0.05": round(float((e < 0.05).mean()), 3), "pck@0.1": round(float((e < 0.1).mean()), 3),
            "pck@0.2": round(float((e < 0.2).mean()), 3), "nme": round(float(np.nanmean(e)), 4)}


# ---------------------------------------------------------------- image degradations and fixes

def darken(img: np.ndarray, k: float, rng) -> np.ndarray:
    x = (img.astype(np.float32) / 255.0) ** 2.2 * k            # linear light, scaled exposure
    x = rng.poisson(x * 400.0) / 400.0 + rng.normal(0, 0.004, x.shape)  # shot + read noise
    x = np.clip(x, 0, 1) ** (1 / 2.2)
    return (x * 255).round().astype(np.uint8)


def luminance(img) -> float:
    return float((0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).mean() / 255)


def gain_fix(img: np.ndarray, target=0.45, max_gain=4.0) -> np.ndarray:
    """What the browser can do cheaply: a canvas `brightness(g)` filter = linear gain, capped."""
    g = min(max_gain, max(1.0, target / max(luminance(img), 1e-3)))
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def clahe_fix(img: np.ndarray) -> np.ndarray:
    import cv2
    L = max(luminance(img), 1e-3)
    gamma = float(np.clip(np.log(0.45) / np.log(L), 0.3, 1.0))
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    x = cv2.LUT(img, lut)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def occlude(frames: list[np.ndarray], ref: list, rng, burst=(6, 18), joints=(25, 26, 27, 28, 13, 14, 15, 16)):
    """Grey boxes over one joint (size ~ 0.6 torso) in bursts. Returns frames and a per-frame mask of covered joints."""
    out = [f.copy() for f in frames]
    mask = [np.zeros(33, bool) for _ in frames]
    k = int(rng.integers(5, 15))
    H, W = frames[0].shape[:2]
    while k < len(frames):
        n = int(rng.integers(*burst))
        j = int(rng.choice(joints))
        for i in range(k, min(len(frames), k + n)):
            r = ref[i]
            if r is None:
                continue
            R = np.array(r)
            torso = np.linalg.norm(((R[11, :2] + R[12, :2]) / 2 - (R[23, :2] + R[24, :2]) / 2) * [W, H])
            cx, cy, s = R[j, 0] * W, R[j, 1] * H, 0.3 * torso
            x0, x1, y0, y1 = int(cx - s), int(cx + s), int(cy - s), int(cy + s)
            out[i][max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 128
            for q in range(33):
                if x0 <= R[q, 0] * W <= x1 and y0 <= R[q, 1] * H <= y1:
                    mask[i][q] = True
        k += n + int(rng.integers(10, 30))
    return out, mask


# ---------------------------------------------------------------- rep counting through the TS engine

def count_reps(landmark_sets: dict[str, list], width: int, height: int, fps=15.0) -> dict:
    with tempfile.TemporaryDirectory() as d:
        jobs = []
        for name, lms in landmark_sets.items():
            f = Path(d) / f"{name}.json"
            f.write_text(json.dumps({"width": width, "height": height, "fps": fps,
                                     "frames": [{"t": round(i / fps, 3), "lm": lm} for i, lm in enumerate(lms)]}))
            for g in (False, True):
                jobs.append({"id": f"{name}|{'guard' if g else 'raw'}", "file": str(f), "exercise": "squat", "guard": g})
        jp, op = Path(d) / "jobs.json", Path(d) / "out.json"
        jp.write_text(json.dumps({"jobs": jobs}))
        subprocess.run(["npx", "tsx", "scripts/eval-synthetic.ts", str(jp), str(op)], cwd=ROOT / "frontend",
                       check=True, capture_output=True)
        res = json.loads(op.read_text())["results"]
    out = {}
    for r in res:
        name, mode = r["id"].split("|")
        out.setdefault(name, {})[mode] = {"reps": len(r["reps"]), "scored": sum(x["scored"] for x in r["reps"]),
                                          "usable_frames": r["frames_usable"]}
    return out


def run(cfg: dict, out: Path, log=print) -> dict:
    import cv2  # noqa: F401  (fail early if opencv is missing)
    ensure_models()
    rng = np.random.default_rng(cfg.get("seed", 7))
    frames = read_frames(CLIP)
    H, W = frames[0].shape[:2]
    hw = H / W
    log(f"{len(frames)} frames {W}x{H}")

    res: dict = {"clip": str(CLIP.relative_to(ROOT)), "frames": len(frames), "size": [W, H], "gt_reps": GT_REPS,
                 "reference": "MediaPipe heavy on unmodified frames (pseudo ground truth, not human labels)"}
    lms, lat = {}, {}
    for m in MODELS:
        lms[m], ms = run_model(m, frames)
        lat[m] = {"median_ms": round(float(np.median(ms[3:])), 2), "p95_ms": round(float(np.percentile(ms[3:], 95)), 2),
                  "model_mb": round(MODELS[m].stat().st_size / 1e6, 1)}
        log(f"{m}: {lat[m]}")
    ref = lms["heavy"]
    res["latency_cpu"] = lat
    res["model_agreement_vs_heavy"] = {m: agreement(lms[m], ref, hw) for m in ("lite", "full")}

    sets = {"clean_lite": lms["lite"]}
    low = {}
    for k in cfg.get("light_levels", [0.5, 0.25, 0.12, 0.06]):
        dark = [darken(f, k, rng) for f in frames]
        row = {"mean_luminance": round(float(np.mean([luminance(f) for f in dark[::10]])), 3)}
        for fix, fn in (("raw", None), ("gain", gain_fix), ("gamma_clahe", clahe_fix)):
            imgs = dark if fn is None else [fn(f) for f in dark]
            p, _ = run_model("lite", imgs)
            row[fix] = agreement(p, ref, hw)
            sets[f"light{k}_{fix}"] = p
            if k == cfg.get("save_example_level", 0.12) and fix != "raw":
                cv2_write(out / f"lowlight_{k}_{fix}.jpg", imgs[len(imgs) // 3])
            if k == cfg.get("save_example_level", 0.12) and fix == "raw":
                cv2_write(out / f"lowlight_{k}_raw.jpg", imgs[len(imgs) // 3])
        low[str(k)] = row
        log(f"light {k}: raw pck@0.1 {row['raw']['pck@0.1']} gain {row['gain']['pck@0.1']} clahe {row['gamma_clahe']['pck@0.1']}")
    res["low_light"] = low

    occ_frames, mask = occlude(frames, ref, rng)
    p_occ, _ = run_model("lite", occ_frames)
    covered = [np.array(m_) for m_ in mask]
    res["occlusion"] = {
        "frames_with_box": int(sum(m_.any() for m_ in covered)),
        "covered_joints": agreement(p_occ, ref, hw, mask=covered),
        "uncovered_joints": agreement(p_occ, ref, hw, mask=[~m_ for m_ in covered]),
        "same_joints_without_box": agreement(lms["lite"], ref, hw, mask=covered),
        "visibility_on_covered": round(float(np.mean([p_occ[i][j][2] for i in range(len(p_occ)) if p_occ[i]
                                                      for j in range(33) if covered[i][j]] or [np.nan])), 3),
        "visibility_same_joints_clean": round(float(np.mean([lms["lite"][i][j][2] for i in range(len(p_occ)) if lms["lite"][i]
                                                             for j in range(33) if covered[i][j]] or [np.nan])), 3),
    }
    cv2_write(out / "occlusion_example.jpg", occ_frames[next(i for i, m_ in enumerate(covered) if m_.any())])
    sets["occluded_lite"] = p_occ
    for m in ("full", "heavy"):
        sets[f"clean_{m}"] = lms[m]
    res["rep_count"] = count_reps(sets, W, H)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    return res


def cv2_write(path: Path, rgb: np.ndarray) -> None:
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
