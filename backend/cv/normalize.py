"""Coordinate normalization and per-frame pose features.

MediaPipe returns x / image width and y / image height, so raw coordinates depend on the image
size, where the person stands, how far away they are and which way the camera is rotated. A model
that should recognise *movement* must not depend on any of that. The steps, in order:

1. Aspect correction: x *= width / height, so both axes are in units of image height (angles
   measured on uncorrected coordinates are wrong for any non-square video).
2. Translation: subtract the hip midpoint -> position in the frame does not matter.
3. Scale: divide by a body size -> distance to the camera does not matter. The torso length alone
   collapses when the torso points at the camera (push-up seen from the front), so the size is
   max(torso, 0.5 x hip-to-ankle, shoulder width) per frame, then the median over the window.
4. (Training only) mirror augmentation: x -> -x with left/right labels swapped, so the model does not
   learn which side of the body faces the camera.

Rotation is deliberately NOT normalised: the direction of gravity (image "down") is what
separates a push-up (horizontal body) from a squat.
"""
from __future__ import annotations

import numpy as np

from .skeleton import (L_ANK, L_EL, L_HIP, L_KNEE, L_SH, L_WR, LEFT_IDX, R_ANK, R_EL, R_HIP, R_KNEE, R_SH,
                       R_WR, RIGHT_IDX)

BODY12 = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]
SWAP33 = np.arange(33)
SWAP33[LEFT_IDX], SWAP33[RIGHT_IDX] = RIGHT_IDX, LEFT_IDX


def frames_to_array(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """[{t, lm}] -> (lm [T, 33, 3] with NaN for missing frames, present [T] bool)."""
    T = len(frames)
    arr = np.full((T, 33, 3), np.nan)
    present = np.zeros(T, bool)
    for i, f in enumerate(frames):
        if f["lm"] is not None:
            arr[i] = f["lm"]
            present[i] = True
    return arr, present


def forward_fill(arr: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = arr.copy()
    last = None
    for i in range(len(out)):
        if present[i]:
            last = out[i]
        elif last is not None:
            out[i] = last
            out[i, :, 2] = 0.0
    first = np.argmax(present) if present.any() else None
    if first:
        out[:first] = out[first]
        out[:first, :, 2] = 0.0
    return np.nan_to_num(out)


def body_size(xy: np.ndarray) -> np.ndarray:
    """Per-frame body size [T] in aspect-corrected units."""
    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    ank = (xy[:, L_ANK] + xy[:, R_ANK]) / 2
    torso = np.linalg.norm(sh - hip, axis=-1)
    leg = np.linalg.norm(hip - ank, axis=-1)
    shw = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=-1)
    return np.maximum.reduce([torso, 0.5 * leg, shw])


def normalize(lm: np.ndarray, aspect: float, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """lm [T, 33, 3] (x, y, vis) -> normalised xy [T, 33, 2] (hip-centred, body-size units, y down), vis [T, 33]."""
    xy = lm[..., :2].copy()
    xy[..., 0] *= aspect
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    xy -= hip[:, None, :]
    s = scale if scale is not None else float(np.median(body_size(xy))) or 1.0
    return xy / max(s, 1e-6), lm[..., 2]


def mirror(lm: np.ndarray) -> np.ndarray:
    """Horizontal flip of normalised MediaPipe landmarks (x -> 1 - x) with left/right swapped."""
    out = lm[:, SWAP33].copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _from_vertical(lo, hi):
    d = hi - lo
    return np.degrees(np.arctan2(d[..., 0], -d[..., 1]))  # signed, 0 = upright, image y down


FRAME_FEATURES = (
    [f"x{i}" for i in BODY12] + [f"y{i}" for i in BODY12] + [f"v{i}" for i in BODY12]
    + ["knee_l", "knee_r", "hip_l", "hip_r", "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
       "trunk_sin", "trunk_cos", "shank_l", "shank_r", "width_ratio", "wrist_height", "present"]
)


def frame_features(lm: np.ndarray, present: np.ndarray, aspect: float) -> np.ndarray:
    """[T, 33, 3] raw landmarks -> [T, F] features (F = len(FRAME_FEATURES)), scaled to roughly unit range."""
    filled = forward_fill(lm, present)
    xy, vis = normalize(filled, aspect)
    P = lambda i: xy[:, i]  # noqa: E731
    sh, hip = (P(L_SH) + P(R_SH)) / 2, (P(L_HIP) + P(R_HIP)) / 2
    trunk = np.radians(_from_vertical(hip, sh))
    torso = np.linalg.norm(sh - hip, axis=-1) + 1e-6
    feats = [
        xy[:, BODY12, 0], xy[:, BODY12, 1], vis[:, BODY12],
        np.stack([
            _angle(P(L_HIP), P(L_KNEE), P(L_ANK)) / 180, _angle(P(R_HIP), P(R_KNEE), P(R_ANK)) / 180,
            _angle(P(L_SH), P(L_HIP), P(L_KNEE)) / 180, _angle(P(R_SH), P(R_HIP), P(R_KNEE)) / 180,
            _angle(P(L_SH), P(L_EL), P(L_WR)) / 180, _angle(P(R_SH), P(R_EL), P(R_WR)) / 180,
            _angle(P(L_EL), P(L_SH), P(L_HIP)) / 180, _angle(P(R_EL), P(R_SH), P(R_HIP)) / 180,
            np.sin(trunk), np.cos(trunk),
            np.abs(_from_vertical(P(L_ANK), P(L_KNEE))) / 180, np.abs(_from_vertical(P(R_ANK), P(R_KNEE))) / 180,
            np.linalg.norm(P(L_SH) - P(R_SH), axis=-1) / torso / 2,
            ((P(L_WR) + P(R_WR)) / 2 - sh)[:, 1],
            present.astype(float),
        ], -1),
    ]
    return np.nan_to_num(np.concatenate(feats, -1).astype(np.float32))
