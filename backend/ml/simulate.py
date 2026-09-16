"""Biomechanical rep simulator -> labelled synthetic rep feature vectors.

WHY SYNTHETIC: there is no public dataset of per-rep pose features with expert form labels
that we can redistribute, so the form classifier is trained on simulated reps. The simulator is
a simplified model, not ground truth about real lifters. Real-world accuracy is unknown until the
model is validated on labelled real reps (see README, "Limitations").

HOW IT WORKS (per rep)
1. Sample a subject: height, segment-length ratios (Winter's anthropometric table, +-noise),
   ankle dorsiflexion mobility, and the camera view (side / front / oblique).
2. Sample the *true* movement parameters for that rep (bottom and top joint angle, tempo, knee
   valgus, torso lean ...). Each parameter comes from a mixture: a "clean" component, a "fault"
   component, and a wide "borderline" component that straddles the threshold, so the classes
   overlap like they would in real data.
3. For the squat the torso lean is not sampled freely: it is solved from a 2-D sagittal
   balance model (whole-body centre of mass must stay over mid-foot given the shank tilt allowed
   by ankle mobility and the thigh angle). Long femurs / stiff ankles therefore force more lean,
   which is the real reason people "fold" in a squat.
4. Build a 30 fps joint-angle time series (cosine-eased descent, pause, ascent), add landmark
   jitter (camera-view dependent) and measure the features exactly the way the browser does
   (min / max over the series, threshold-crossing tempo, projection losses in non-side views).
5. Label = 1 ("good") if none of the ground-truth fault conditions hold on the TRUE parameters,
   else 0. The classifier only sees the noisy measurements, so it has to learn a robust boundary.

Run:  python -m ml.simulate --n 200   (prints a small sample)
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .features import EXERCISES, FEATURES, NEUTRAL

FPS = 30
SIMULATOR_VERSION = "1.1"  # bump when the generator changes (part of the dataset id)

# Winter (2009) segment lengths as a fraction of body height, and segment mass fractions.
SHANK_L, THIGH_L, TRUNK_L = 0.246, 0.245, 0.288
M_SHANK, M_THIGH, M_UPPER = 0.0465, 0.100, 0.678  # per leg for shank/thigh; HAT for upper


def _mix(rng, clean, fault, border, p_fault, p_border):
    """Sample from clean / fault / borderline uniform ranges."""
    u = rng.random()
    lo, hi = fault if u < p_fault else border if u < p_fault + p_border else clean
    return float(rng.uniform(lo, hi))


def _trajectory(top, bottom, ecc, pause, con, hold=0.25):
    """Joint angle over one rep: hold at top, ease down, pause, ease up, hold."""
    t1 = np.linspace(0, 1, max(2, int(ecc * FPS)))
    t2 = np.linspace(0, 1, max(2, int(con * FPS)))
    down = top + (bottom - top) * (1 - np.cos(np.pi * t1)) / 2
    up = bottom + (top - bottom) * (1 - np.cos(np.pi * t2)) / 2
    return np.concatenate([
        np.full(int(hold * FPS), top), down, np.full(int(pause * FPS), bottom), up,
        np.full(int(hold * FPS), top),
    ])


def _ema(x, alpha=0.5):
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def _measure_tempo(series, top_ref, bottom_ref):
    """Browser-style tempo: leave-top crossing -> minimum -> return-to-top crossing."""
    thr = top_ref - 0.2 * (top_ref - bottom_ref)
    below = np.where(series < thr)[0]
    if len(below) == 0:
        return NEUTRAL["ecc_s"], NEUTRAL["con_s"]
    i0, i2 = below[0], below[-1]
    i1 = i0 + int(np.argmin(series[i0:i2 + 1]))
    return (i1 - i0) / FPS, (i2 - i1) / FPS


def _squat_geometry(knee_angle, height, femur_ratio, dorsi_max, heel_fault):
    """Solve shank tilt, torso lean, depth and heel lift for a given knee angle (sagittal plane)."""
    ls, lt, lk = SHANK_L * height, THIGH_L * height * femur_ratio, TRUNK_L * height
    flex = math.radians(180 - knee_angle)
    want = 0.5 * flex                     # shank tilt a lifter would like
    lim = math.radians(dorsi_max)
    heel = 0.0
    if want > lim:
        if heel_fault:                    # lifter rises onto toes to get the knee forward
            heel = min(0.6, (want - lim) * 1.2)
            shank = want
        else:
            shank = lim
    else:
        shank = want
    thigh = flex - shank                  # thigh angle from vertical
    xk = ls * math.sin(shank)
    xh = xk - lt * math.sin(thigh)
    yk = ls * math.cos(shank)
    yh = yk + lt * math.cos(thigh)
    x_target = 0.03 * height              # mid-foot, a little in front of the ankle
    # 2*shank + 2*thigh + upper-body COM (at 0.55 of trunk + arms forward) over mid-foot
    m_total = 2 * M_SHANK + 2 * M_THIGH + M_UPPER
    x_legs = 2 * M_SHANK * (xk / 2) + 2 * M_THIGH * ((xk + xh) / 2)
    need = (m_total * x_target - x_legs) / M_UPPER - xh   # horizontal offset upper COM must have
    s = np.clip(need / (0.85 * lk), -0.2, 0.98)  # arms held forward lengthen the lever
    lean = math.degrees(math.asin(s))
    depth = (yk - yh) / lt                # >0 when hip is below knee
    return max(0.0, lean), depth, heel


def simulate_rep(exercise: str, rng: np.random.Generator, fatigue: float = 0.0,
                 fault_rate: float = 0.11, view: str | None = None):
    """Return (features: dict, label: int, faults: list[str]) for one simulated rep.

    fatigue in [0, 1] slows the concentric phase, shrinks range of motion and makes faults
    more likely (used by the seed script to generate realistic within-set fatigue)."""
    p = min(0.9, fault_rate + 0.35 * fatigue)
    pb = 0.10
    view = view or rng.choice(["side", "front", "oblique"], p=[0.5, 0.3, 0.2])
    jitter = {"side": 2.0, "oblique": 3.0, "front": 4.0}[view]
    height = rng.normal(1.72, 0.09)
    faults: list[str] = []
    f = dict(NEUTRAL)

    ecc = _mix(rng, (0.8, 2.0), (0.25, 0.48), (0.4, 0.8), p * 0.5 if exercise == "squat" else 0.05, pb * 0.5)
    con = rng.uniform(0.6, 1.4) * (1 + 0.9 * fatigue)
    pause = rng.uniform(0.0, 0.4)

    if exercise == "squat":
        femur = rng.normal(1.0, 0.06)
        dorsi = rng.normal(34, 7)
        bottom = _mix(rng, (38, 60), (82, 125), (56, 80), p, pb) + 12 * fatigue
        top = _mix(rng, (165, 178), (140, 158), (155, 166), p * 0.4, pb * 0.5)
        heel_fault = rng.random() < p * 0.5
        extra_lean = _mix(rng, (0, 6), (15, 30), (5, 16), p * 0.7, pb)
        valgus = _mix(rng, (0.9, 1.25), (0.55, 0.78), (0.74, 0.9), p * 0.8, pb)
        lean, depth, heel = _squat_geometry(bottom, height, femur, dorsi, heel_fault)
        lean += extra_lean
        if depth < -0.15:
            faults.append("shallow_depth")
        if lean > 45:
            faults.append("forward_lean")
        if valgus < 0.8:
            faults.append("knee_valgus")
        if heel > 0.2:
            faults.append("heel_lift")
        if ecc < 0.5:
            faults.append("rushed_descent")
        if top < 160:
            faults.append("no_lockout")
        # measurement
        proj = {"side": 1.0, "oblique": 0.75, "front": 0.35}[view]
        f["torso_lean_max"] = lean * proj + rng.normal(0, jitter)
        f["torso_sway"] = lean * proj * rng.uniform(0.8, 1.0) + abs(rng.normal(0, 1.5))
        f["depth"] = depth * (1.0 if view == "side" else 0.6) + rng.normal(0, 0.04 if view == "side" else 0.12)
        f["heel_rise"] = max(0.0, heel * proj + rng.normal(0, 0.04))
        f["valgus_min"] = (valgus + rng.normal(0, 0.05)) if view != "side" else NEUTRAL["valgus_min"]
        f["hip_line_min"] = NEUTRAL["hip_line_min"] + rng.normal(0, 3)
        f["elbow_drift_max"] = NEUTRAL["elbow_drift_max"] + rng.normal(0, 3)
        knee_bias = {"side": 0, "oblique": 6, "front": 14}[view]  # knee angle reads larger off-axis
        series = _trajectory(top, bottom, ecc, pause, con) + knee_bias * (1 - _trajectory(1, 0, ecc, pause, con))

    elif exercise == "pushup":
        bottom = _mix(rng, (55, 90), (100, 130), (85, 102), p, pb) + 10 * fatigue
        top = _mix(rng, (160, 178), (130, 152), (150, 162), p * 0.5, pb)
        u = rng.random()
        line = _mix(rng, (166, 180), (140, 160), (157, 168), p, pb)
        sag = u < 0.6
        if line < 162:
            faults.append("hip_sag" if sag else "hip_pike")
        if bottom > 95:
            faults.append("shallow_pushup")
        if top < 155:
            faults.append("no_lockout")
        f["hip_line_min"] = line - abs(rng.normal(0, jitter))
        f["torso_lean_max"] = rng.normal(78, 5)
        f["torso_sway"] = abs(rng.normal(4, 2)) + (180 - line) * 0.3
        f["elbow_drift_max"] = rng.normal(50, 10)
        f["depth"], f["heel_rise"], f["valgus_min"] = 0.0, 0.0, 1.0
        series = _trajectory(top, bottom, ecc, pause, con)

    elif exercise == "curl":
        bottom = _mix(rng, (28, 60), (75, 105), (55, 78), p, pb) + 10 * fatigue   # flexed end
        top = _mix(rng, (155, 176), (118, 146), (143, 156), p, pb)                # extended end
        drift = _mix(rng, (3, 22), (32, 55), (20, 34), p + 0.2 * fatigue, pb)
        sway = _mix(rng, (0.5, 8), (13, 28), (7, 14), p + 0.2 * fatigue, pb)
        if drift > 28:
            faults.append("elbow_drift")
        if sway > 11:
            faults.append("torso_swing")
        if top < 150:
            faults.append("partial_extension")
        if bottom > 70:
            faults.append("partial_curl")
        if sway > 11:
            con *= 0.6  # swinging = momentum = faster lift
        proj = {"side": 1.0, "oblique": 0.8, "front": 0.4}[view]
        f["elbow_drift_max"] = drift * proj + rng.normal(0, jitter)
        f["torso_sway"] = sway * proj + abs(rng.normal(0, 1.2))
        f["torso_lean_max"] = rng.uniform(0, 6) + sway * proj * 0.7
        f["hip_line_min"] = 178 - sway * 0.4 + rng.normal(0, 2)
        f["depth"], f["heel_rise"], f["valgus_min"] = 0.0, 0.0, 1.0
        # curl: rest = extended arm; the first half (flexing) is the lift, i.e. concentric
        series = _trajectory(top, bottom, con, pause, ecc)

    elif exercise == "press":
        rack = _mix(rng, (55, 88), (95, 120), (85, 98), p * 0.4, pb)
        lock = _mix(rng, (162, 178), (128, 154), (152, 164), p, pb) - 8 * fatigue
        arch = _mix(rng, (0, 9), (15, 30), (8, 16), p + 0.2 * fatigue, pb)
        if lock < 158:
            faults.append("short_press")
        if arch > 13:
            faults.append("back_arch")
        proj = {"side": 1.0, "oblique": 0.7, "front": 0.3}[view]
        f["torso_lean_max"] = arch * proj + abs(rng.normal(0, jitter))
        f["torso_sway"] = arch * proj * 0.8 + abs(rng.normal(0, 1.5))
        f["hip_line_min"] = 180 - arch * proj * 0.9 + rng.normal(0, 2)
        f["elbow_drift_max"] = rng.normal(15, 6)
        f["depth"], f["heel_rise"], f["valgus_min"] = 0.0, 0.0, 1.0
        # press signal is 180 - elbow angle so that the rack position is the "top" of the signal
        series = 180 - _trajectory(180 - rack, 180 - lock, con, pause, ecc)
        top, bottom = lock, rack

    elif exercise == "lunge":
        bottom = _mix(rng, (75, 102), (112, 140), (98, 115), p, pb) + 10 * fatigue
        top = rng.uniform(158, 178)
        lean = _mix(rng, (0, 16), (24, 42), (14, 26), p, pb)
        valgus = _mix(rng, (0.9, 1.2), (0.55, 0.78), (0.74, 0.9), p * 0.6, pb)
        if bottom > 108:
            faults.append("shallow_lunge")
        if lean > 22:
            faults.append("torso_lean_lunge")
        if valgus < 0.8:
            faults.append("knee_valgus")
        proj = {"side": 1.0, "oblique": 0.75, "front": 0.35}[view]
        f["torso_lean_max"] = lean * proj + rng.normal(0, jitter)
        f["torso_sway"] = lean * proj * 0.9 + abs(rng.normal(0, 1.5))
        f["valgus_min"] = (valgus + rng.normal(0, 0.05)) if view != "side" else 1.0
        f["depth"] = (95 - bottom) / 60 + rng.normal(0, 0.05)
        f["hip_line_min"] = 175 + rng.normal(0, 3)
        f["elbow_drift_max"] = NEUTRAL["elbow_drift_max"] + rng.normal(0, 3)
        f["heel_rise"] = 0.0
        series = _trajectory(top, bottom, ecc, pause, con)
    else:
        raise ValueError(exercise)

    noisy = _ema(series + rng.normal(0, jitter, size=len(series)), 0.5)
    f["min_angle"] = float(noisy.min())
    f["max_angle"] = float(noisy.max())
    f["rom"] = f["max_angle"] - f["min_angle"]
    if exercise == "press":
        # the press starts at the rack (small elbow angle); tempo measured on the inverted signal
        con_m, ecc_m = _measure_tempo(180 - noisy, 180 - bottom, 180 - top)
        f["ecc_s"], f["con_s"] = ecc_m, con_m
    elif exercise == "curl":
        f["con_s"], f["ecc_s"] = _measure_tempo(noisy, top, bottom)
    else:
        f["ecc_s"], f["con_s"] = _measure_tempo(noisy, top, bottom)
    f = {k: round(float(f[k]), 3) for k in FEATURES}
    return f, int(not faults), faults


def generate(exercise: str, n: int, seed: int = 0, with_faults: bool = False):
    """n synthetic reps -> X (n x len(FEATURES)), y (1 = clean) [, list of true fault codes]."""
    rng = np.random.default_rng(seed)
    X, y, F = [], [], []
    for _ in range(n):
        fat = float(rng.beta(1.2, 4))
        feats, label, faults = simulate_rep(exercise, rng, fatigue=fat)
        X.append([feats[k] for k in FEATURES])
        y.append(label)
        F.append(faults)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)
    return (X, y, F) if with_faults else (X, y)


def dataset_hash(*arrays: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    for ex in EXERCISES:
        X, y = generate(ex, args.n, seed=1)
        print(f"{ex:8s} n={len(y)} good-rate={y.mean():.2f}")
        f, lab, faults = simulate_rep(ex, rng)
        print("   sample:", f, "good" if lab else faults)
