"""Landmark detection accuracy against HUMAN keypoint labels (COCO val2017).

    python -m experiments.cv.coco_pck --ann annotations/person_keypoints_val2017.json \
        --images val2017/ --model lite --max 500 --out experiments/results/coco_pck_lite.json

Needs the COCO 2017 validation images and keypoint annotations (cocodataset.org, CC BY 4.0
annotations; ~1 GB images + 241 MB annotation zip). They could not be downloaded in the environment
where the other experiments ran, so the README reports this metric as "dataset required" until the
script is run.

Protocol
* Images with exactly one annotated person, >= 12 labelled keypoints and box area >= 96x96 px
  (MediaPipe PoseLandmarker is a single-person model; multi-person images would test the detector's
  person choice, not landmark accuracy).
* The image is cropped to the person box expanded by 25%, then run in IMAGE mode.
* COCO's 17 keypoints map 1:1 to BlazePose indices (below). Only labelled keypoints (v > 0) count.
* PCK@a: a predicted keypoint is correct when it is within a x torso diameter (distance from the
  left shoulder to the right hip, as in LSP-style PCK) of the label.
* OKS: COCO's object keypoint similarity with the official per-keypoint sigmas; mean OKS and the
  share of people with OKS >= 0.5 / 0.75 (an AP-like summary without the ranking step).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COCO_TO_BLAZE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
              "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, help="official person_keypoints_val2017.json")
    ap.add_argument("--images", type=Path, help="official val2017/ image folder")
    ap.add_argument("--parquet", type=Path, help="alternative: Hugging Face rom1x38/COCO_keypoints val parquet "
                    "(image, keypoints, bboxes; no segmentation area, so OKS uses the box area)")
    ap.add_argument("--model", default="lite", choices=["lite", "full", "heavy"])
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    model = (ROOT / "frontend/public/models/pose_landmarker_lite.task" if a.model == "lite"
             else ROOT / f"data/models/pose_landmarker_{a.model}.task")
    def samples():
        """Yields (image_id, RGB image, keypoints [17, 3], bbox [x, y, w, h], area) for single-person images."""
        if a.parquet:
            import io

            import pyarrow.parquet as pq
            from PIL import Image
            t = pq.read_table(a.parquet)
            n = 0
            for i in range(t.num_rows):
                row = t.slice(i, 1).to_pylist()[0]
                kps, boxes = row["keypoints"], row["bboxes"]
                if len(kps) != 1:
                    continue
                kp = np.array(kps[0], float)
                x, y, w, h = boxes[0]
                if (kp[:, 2] > 0).sum() < 12 or w * h < 96 * 96:
                    continue
                img = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
                yield row["image_id"], img, kp, boxes[0], w * h
                n += 1
                if n >= a.max:
                    return
            return
        coco = json.loads(a.ann.read_text())
        by_img = defaultdict(list)
        for an in coco["annotations"]:
            if not an.get("iscrowd"):
                by_img[an["image_id"]].append(an)
        files = {im["id"]: im["file_name"] for im in coco["images"]}
        picks = [(i, anns[0]) for i, anns in by_img.items()
                 if len(anns) == 1 and anns[0]["num_keypoints"] >= 12 and anns[0]["bbox"][2] * anns[0]["bbox"][3] >= 96 * 96]
        for img_id, an in sorted(picks)[: a.max]:
            img = cv2.cvtColor(cv2.imread(str(a.images / files[img_id])), cv2.COLOR_BGR2RGB)
            yield img_id, img, np.array(an["keypoints"], float).reshape(17, 3), an["bbox"], an["area"]

    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.IMAGE))
    hits = {t: np.zeros(17) for t in (0.05, 0.1, 0.2)}
    labelled = np.zeros(17)
    labelled_det = np.zeros(17)
    oks_all, missed = [], 0
    people = 0
    for img_id, img, kp, bbox, area in samples():
        people += 1
        x, y, w, h = bbox
        cx, cy, s = x + w / 2, y + h / 2, 1.25
        x0, y0 = int(max(0, cx - w * s / 2)), int(max(0, cy - h * s / 2))
        x1, y1 = int(min(img.shape[1], cx + w * s / 2)), int(min(img.shape[0], cy + h * s / 2))
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        vis = kp[:, 2] > 0
        labelled += vis
        if not r.pose_landmarks:
            missed += 1
            oks_all.append(0.0)
            continue
        labelled_det += vis
        P = np.array([[r.pose_landmarks[0][j].x * crop.shape[1] + x0, r.pose_landmarks[0][j].y * crop.shape[0] + y0]
                      for j in COCO_TO_BLAZE])
        d = np.linalg.norm(P - kp[:, :2], axis=1)
        torso = np.linalg.norm(kp[5, :2] - kp[12, :2]) if vis[5] and vis[12] else np.sqrt(w * h) / 2
        for t in hits:
            hits[t] += (d < t * torso) & vis
        area = area or w * h
        e = d ** 2 / (2 * area * (2 * SIGMAS) ** 2 + 1e-9)
        oks_all.append(float(np.exp(-e)[vis].mean()))
    lm.close()
    oks = np.array(oks_all)
    out = {"model": a.model, "people": people, "not_detected": missed,
           "source": str(a.parquet.name if a.parquet else a.ann.name),
           "oks_area": "bounding-box area (lenient approximation)" if a.parquet else "COCO segmentation area",
           "pck": {str(t): round(float(hits[t].sum() / labelled.sum()), 4) for t in hits},
           "pck_detected_only": {str(t): round(float(hits[t].sum() / max(1, labelled_det.sum())), 4) for t in hits},
           "pck@0.2_per_joint": {n: round(float(hits[0.2][i] / max(1, labelled[i])), 3) for i, n in enumerate(COCO_NAMES)},
           "mean_oks": round(float(oks.mean()), 4), "oks>=0.5": round(float((oks >= 0.5).mean()), 4),
           "oks>=0.75": round(float((oks >= 0.75).mean()), 4)}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
