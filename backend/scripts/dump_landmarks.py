"""Record MediaPipe landmarks of a video at 15 fps into a JSON fixture (for frontend tests).

    python -m scripts.dump_landmarks ../samples/squat_demo.webm ../frontend/src/pose/__tests__/fixtures/squat_demo.json
"""
import json
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "../frontend/public/models/pose_landmarker_lite.task"


def main(src: str, dst: str, fps: float = 15.0):
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO)
    lm = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, t, i = [], 0.0, 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts = i / src_fps
        i += 1
        if ts + 1e-6 < t:
            continue
        t += 1 / fps
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(ts * 1000))
        pts = None
        if r.pose_landmarks:
            pts = [[round(p.x, 4), round(p.y, 4), round(p.visibility, 3)] for p in r.pose_landmarks[0]]
        frames.append({"t": round(ts, 3), "lm": pts})
    json.dump({"width": w, "height": h, "fps": fps, "frames": frames}, open(dst, "w"), separators=(",", ":"))
    print(f"{len(frames)} frames, {sum(f['lm'] is not None for f in frames)} with pose -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
