"""Run MediaPipe PoseLandmarker (the same pose_landmarker_lite.task the browser uses) over a clip
and store the landmarks at a fixed analysis rate, like the app's Upload mode (15 fps).

Output: data/real_clips/landmarks/<id>.json  {width, height, fps, source_sha256, segment, frames}
Requires: pip install -r backend/requirements-eval.txt  (mediapipe, opencv-python)

Note: this is MediaPipe's Python build. The browser uses the WASM build of the same model; on the
sample clip both give the same rep count (4), see experiments/README.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "frontend" / "public" / "models" / "pose_landmarker_lite.task"
LANDMARKS = ROOT / "data" / "real_clips" / "landmarks"


def extract(src: Path, dst: Path, fps: float = 15.0, segment: list[float] | None = None,
            source_sha256: str | None = None) -> dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                        running_mode=vision.RunningMode.VIDEO)
    lm = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start, end = (segment or [0, float("inf")])
    frames, next_t, i = [], start, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts = i / src_fps
        i += 1
        if ts + 1e-6 < next_t:
            continue
        if ts > end:
            break
        next_t += 1 / fps
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, int(ts * 1000))
        pts = None
        if r.pose_landmarks:
            pts = [[round(p.x, 4), round(p.y, 4), round(p.visibility, 3)] for p in r.pose_landmarks[0]]
        frames.append({"t": round(ts - start, 3), "lm": pts})
    lm.close()
    out = {"width": w, "height": h, "fps": fps, "source_sha256": source_sha256, "segment": segment, "frames": frames}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, separators=(",", ":")))
    return out


if __name__ == "__main__":
    s, d = Path(sys.argv[1]), Path(sys.argv[2])
    res = extract(s, d)
    print(f"{len(res['frames'])} frames, {sum(f['lm'] is not None for f in res['frames'])} with pose -> {d}")
