"""3-D synthetic movement sequences in MediaPipe's 33-landmark topology, with ground truth.

WHY: real videos with per-frame keypoints, rep boundaries, exercise labels and form-fault labels
across many camera angles do not exist in a form we can download here. A 3-D kinematic model gives
all of these labels for free, and lets us sweep exactly the variables we want to test (camera yaw
and height, occlusion, tracking noise, left/right swaps, non-exercise movement).

WHAT IT IS NOT: real people. The bodies are rigid-segment stick figures driven by hand-written
joint trajectories, and the "detector errors" are a noise model, not MediaPipe. Every number
measured on these sequences is labelled *synthetic* in the reports.

Pipeline per sequence
1. Sample a body (height, Winter (2009) segment ratios with +-5% individual variation).
2. Sample an activity: one of 5 exercises (a set of 4-12 reps, per-rep tempo, depth and faults)
   or a non-exercise distractor (idle, marching, arm waving, hip hinge, walking across).
3. Build 3-D joints per frame with forward kinematics + two-bone IK (subject frame:
   +X = subject's left, +Y = up, +Z = facing direction).
4. Place a pinhole camera at yaw/height/distance, project, frame the subject, add a
   MediaPipe-like error model: temporally correlated pixel noise, self-occlusion-driven
   visibility, out-of-frame visibility drop, occluder boxes, detection dropouts and left/right
   swap episodes in side views.
5. Ground truth: class, per-frame "active" flag, rep boundaries, per-rep true metrics and fault
   labels computed on the TRUE 3-D pose with the same thresholds the app's rules use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FPS = 15.0  # the app's Upload mode analyses 15 frames per second
EXERCISES = ["squat", "pushup", "curl", "press", "lunge"]
DISTRACTORS = ["idle", "march", "arm_wave", "hinge", "walk"]
CLASSES = EXERCISES + ["other"]

# MediaPipe BlazePose indices
NOSE, L_EYE_IN, L_EYE, L_EYE_OUT, R_EYE_IN, R_EYE, R_EYE_OUT, L_EAR, R_EAR, MOUTH_L, MOUTH_R = range(11)
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_THUMB, R_THUMB = range(11, 23)
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = range(23, 33)
LEFT_IDX = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RIGHT_IDX = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
X, Y, Z = np.eye(3)


@dataclass
class Body:
    H: float
    thigh: float
    shank: float
    trunk: float
    uarm: float
    farm: float
    hand: float
    hip_w: float
    sh_w: float
    foot: float
    ankle_h: float
    neck: float

    @staticmethod
    def sample(rng: np.random.Generator) -> "Body":
        H = float(rng.uniform(1.55, 1.95))
        j = lambda r: H * r * float(rng.uniform(0.95, 1.05))  # noqa: E731
        return Body(H=H, thigh=j(0.245), shank=j(0.246), trunk=j(0.288), uarm=j(0.186), farm=j(0.146),
                    hand=j(0.08), hip_w=j(0.105), sh_w=j(0.215), foot=j(0.152), ankle_h=j(0.039), neck=j(0.13))


def _n(v):
    return v / (np.linalg.norm(v) + 1e-12)


def sag(phi_deg: float) -> np.ndarray:
    """Unit vector in the sagittal plane, phi from vertical-up, positive = forward (+Z)."""
    p = math.radians(phi_deg)
    return np.array([0.0, math.cos(p), math.sin(p)])


def two_bone(root, target, l1, l2, pole):
    """Two-bone IK: returns (mid, end). The end is pulled in when the target is out of reach."""
    d_vec = target - root
    d = float(np.linalg.norm(d_vec))
    d_dir = d_vec / (d + 1e-12)
    d = min(max(d, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pv = _n(pole - np.dot(pole, d_dir) * d_dir)
    mid = root + a * d_dir + h * pv
    return mid, root + d * d_dir


def interior(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def lean_from_vertical(lo, hi) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, _n(hi - lo)[1]))))


# --------------------------------------------------------------------------------------------
# assembling 33 landmarks from the main joints
# --------------------------------------------------------------------------------------------

def assemble(J: dict, b: Body) -> np.ndarray:
    P = np.zeros((33, 3))
    for k, i in [("sh_l", L_SH), ("sh_r", R_SH), ("el_l", L_EL), ("el_r", R_EL), ("wr_l", L_WR), ("wr_r", R_WR),
                 ("hip_l", L_HIP), ("hip_r", R_HIP), ("kn_l", L_KNEE), ("kn_r", R_KNEE), ("an_l", L_ANK),
                 ("an_r", R_ANK), ("heel_l", L_HEEL), ("heel_r", R_HEEL), ("toe_l", L_TOE), ("toe_r", R_TOE)]:
        P[i] = J[k]
    sh_mid = (J["sh_l"] + J["sh_r"]) / 2
    hip_mid = (J["hip_l"] + J["hip_r"]) / 2
    lat = _n(J["sh_l"] - J["sh_r"])                 # subject's left
    up = _n(sh_mid - hip_mid)
    up = _n(up - np.dot(up, lat) * lat)
    fwd = _n(np.cross(lat, up))                     # facing direction of the chest
    head_up = _n(J.get("head_up", up))
    head_fwd = _n(np.cross(lat, head_up))
    H = b.H
    c = sh_mid + head_up * b.neck
    P[NOSE] = c + head_fwd * 0.055 * H
    for s, (i_in, i_eye, i_out, i_ear, i_mouth) in ((1, (L_EYE_IN, L_EYE, L_EYE_OUT, L_EAR, MOUTH_L)),
                                                   (-1, (R_EYE_IN, R_EYE, R_EYE_OUT, R_EAR, MOUTH_R))):
        base = c + head_fwd * 0.045 * H + head_up * 0.022 * H
        P[i_in] = base + s * lat * 0.012 * H
        P[i_eye] = base + s * lat * 0.022 * H
        P[i_out] = base + s * lat * 0.032 * H
        P[i_ear] = c + s * lat * 0.045 * H + head_up * 0.012 * H
        P[i_mouth] = c + head_fwd * 0.05 * H - head_up * 0.025 * H + s * lat * 0.016 * H
    for s, (wr, el, pk, ix, th) in ((1, (L_WR, L_EL, L_PINKY, L_INDEX, L_THUMB)),
                                    (-1, (R_WR, R_EL, R_PINKY, R_INDEX, R_THUMB))):
        d = _n(P[wr] - P[el])
        side = _n(np.cross(d, fwd)) * s if abs(np.dot(d, fwd)) < 0.95 else lat * s
        P[ix] = P[wr] + d * b.hand + side * 0.012 * H
        P[pk] = P[wr] + d * b.hand * 0.85 + side * 0.035 * H
        P[th] = P[wr] + d * b.hand * 0.5 - side * 0.02 * H + fwd * 0.01 * H
    return P


def standing_feet(b: Body, half_stance: float, ground=0.0):
    """Ankle/heel/toe for flat feet pointing +Z."""
    out = {}
    for s, k in ((1, "l"), (-1, "r")):
        an = np.array([s * half_stance, ground + b.ankle_h, 0.0])
        out["an_" + k] = an
        out["heel_" + k] = an + np.array([0, -b.ankle_h * 0.9, -0.25 * b.foot])
        out["toe_" + k] = an + np.array([0, -b.ankle_h * 0.9, 0.75 * b.foot])
    return out


def arms_at_sides(J, b: Body, lean_deg=0.0, abd=8.0, el_flex=10.0):
    for s, k in ((1, "l"), (-1, "r")):
        down = -sag(lean_deg)
        d = _n(down + s * X * math.tan(math.radians(abd)))
        J["el_" + k] = J["sh_" + k] + d * b.uarm
        f = _n(d * math.cos(math.radians(el_flex)) + sag(lean_deg + 90) * math.sin(math.radians(el_flex)))
        J["wr_" + k] = J["el_" + k] + f * b.farm


def upper_body(J, b: Body, lean_deg: float, lateral_shift=0.0):
    hip_mid = (J["hip_l"] + J["hip_r"]) / 2
    sh_mid = hip_mid + sag(lean_deg) * b.trunk + X * lateral_shift
    J["sh_l"] = sh_mid + X * b.sh_w / 2
    J["sh_r"] = sh_mid - X * b.sh_w / 2


# --------------------------------------------------------------------------------------------
# exercises: pose(u, rep params) with u in [0, 1] (0 = rest position, 1 = turning point)
# --------------------------------------------------------------------------------------------

def _squat(u, r, b: Body):
    hs = r["half_stance"]
    J = standing_feet(b, hs)
    heel_ang = u * r["heel"]
    knee_flex = (180 - r["top"]) + u * (r["top"] - r["bottom"])
    a = u * r["dorsi"] + (180 - r["top"]) * 0.3
    for s, k in ((1, "l"), (-1, "r")):
        toe = J["toe_" + k]
        if heel_ang > 0:  # rotate the foot about the toe
            c_, s_ = math.cos(math.radians(heel_ang)), math.sin(math.radians(heel_ang))
            rot = lambda v: np.array([v[0], v[1] * c_ - v[2] * s_, v[1] * s_ + v[2] * c_])  # noqa: E731  heel goes up
            J["an_" + k] = toe + rot(J["an_" + k] - toe)
            J["heel_" + k] = toe + rot(J["heel_" + k] - toe)
        an = J["an_" + k]
        kn = an + sag(a) * b.shank
        kn[0] = an[0] * (1 - u * r["valgus"])
        hip = kn + sag(a - knee_flex) * b.thigh
        hip[0] = s * b.hip_w / 2
        J["kn_" + k], J["hip_" + k] = kn, hip
    hip_z = (J["hip_l"][2] + J["hip_r"][2]) / 2
    target = 0.2 * b.foot
    lean = math.degrees(math.asin(max(-0.3, min(0.97, (target - hip_z) / b.trunk))))
    lean = max(lean, 3.0) + u * r["extra_lean"]
    upper_body(J, b, lean)
    for s, k in ((1, "l"), (-1, "r")):  # arms reach forward for balance
        wr = J["sh_" + k] + np.array([-s * 0.02 * b.H, -0.04 * b.H * (1 - u), 0.9 * (b.uarm + b.farm)])
        J["el_" + k], J["wr_" + k] = two_bone(J["sh_" + k], wr, b.uarm, b.farm, -Y + 0.2 * s * X)
    J["head_up"] = sag(lean * 0.6)
    return J


def _lunge(u, r, b: Body):
    lead = r["lead"]  # +1 left leg forward, -1 right
    J = {}
    stride = r["stride"]
    for s, k in ((1, "l"), (-1, "r")):
        front = s == lead
        x = s * b.hip_w * 0.55
        if front:
            an = np.array([x, b.ankle_h, stride * 0.5])
            J["heel_" + k] = an + np.array([0, -b.ankle_h * 0.9, -0.25 * b.foot])
            J["toe_" + k] = an + np.array([0, -b.ankle_h * 0.9, 0.75 * b.foot])
        else:  # back foot on the toes
            toe = np.array([x, 0.0, -stride * 0.5])
            an = toe + np.array([0, 0.6 * b.foot, -0.55 * b.foot]) * 0.8
            J["toe_" + k] = toe
            J["heel_" + k] = an + np.array([0, -0.02 * b.H, -0.22 * b.foot])
        J["an_" + k] = an
    leg = b.thigh + b.shank
    # pelvis height is solved so that min(knee angles) follows the requested trajectory
    target = r["top"] + u * (r["bottom"] - r["top"])
    lo, hi = 0.2 * leg, leg * 1.05 + b.ankle_h
    hip_z = stride * 0.08
    for _ in range(30):
        y = (lo + hi) / 2
        ang = []
        for s, k in ((1, "l"), (-1, "r")):
            hip = np.array([s * b.hip_w / 2, y, hip_z])
            kn, _ = two_bone(hip, J["an_" + k], b.thigh, b.shank, Z + 0.15 * s * X * r["valgus"] * -1)
            ang.append(interior(hip, kn, J["an_" + k]))
        if min(ang) > target:
            hi = y
        else:
            lo = y
    for s, k in ((1, "l"), (-1, "r")):
        hip = np.array([s * b.hip_w / 2, (lo + hi) / 2, hip_z])
        kn, an = two_bone(hip, J["an_" + k], b.thigh, b.shank, Z)
        J["hip_" + k], J["kn_" + k] = hip, kn
    lean = r["base_lean"] + u * r["extra_lean"]
    upper_body(J, b, lean)
    for s, k in ((1, "l"), (-1, "r")):  # hands on hips
        wr = J["hip_" + k] + np.array([s * 0.04 * b.H, 0.05 * b.H, 0.0])
        J["el_" + k], J["wr_" + k] = two_bone(J["sh_" + k], wr, b.uarm, b.farm, s * X - 0.5 * Z)
    J["head_up"] = sag(lean * 0.5)
    return J


def _circle_intersect(c1, r1, c2, r2):
    """2-D circle intersection (y, z), returns the higher point."""
    d = np.linalg.norm(c2 - c1)
    d = min(max(d, abs(r1 - r2) + 1e-6), r1 + r2 - 1e-6)
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0))
    e = (c2 - c1) / (np.linalg.norm(c2 - c1) + 1e-12)
    p = c1 + a * e
    perp = np.array([-e[1], e[0]])
    cand = [p + h * perp, p - h * perp]
    return max(cand, key=lambda q: q[0])


def _pushup(u, r, b: Body):
    J = {}
    el = r["top"] + u * (r["bottom"] - r["top"])
    d_sw = math.sqrt(b.uarm ** 2 + b.farm ** 2 - 2 * b.uarm * b.farm * math.cos(math.radians(el)))
    leg = b.thigh + b.shank
    body = leg + b.trunk
    ank = np.array([b.ankle_h * 1.6, 0.0])                      # (y, z) of the ankles, toes on the floor
    wrist = np.array([0.02 * b.H, r["wrist_z"]])
    sh = _circle_intersect(wrist, d_sw, ank, body)              # shoulder (y, z)
    sh3 = np.array([0.0, sh[0], sh[1]])
    an3 = np.array([0.0, ank[0], ank[1]])
    axis = _n(sh3 - an3)
    perp_down = _n(np.cross(X, axis))                           # points to the floor side
    if perp_down[1] > 0:
        perp_down = -perp_down
    sag_off = r["sag"] * (0.6 + 0.4 * u)
    hip_mid = an3 + axis * leg + perp_down * sag_off
    for s, k in ((1, "l"), (-1, "r")):
        xs = s * b.sh_w / 2
        J["sh_" + k] = sh3 + X * xs
        J["hip_" + k] = hip_mid + X * s * b.hip_w / 2
        J["an_" + k] = an3 + X * s * b.hip_w * 0.5
        J["kn_" + k] = (J["hip_" + k] + J["an_" + k]) / 2 + perp_down * sag_off * 0.3
        J["toe_" + k] = J["an_" + k] + np.array([0, -b.ankle_h * 1.5, 0.04 * b.H])
        J["heel_" + k] = J["an_" + k] + np.array([0, 0.035 * b.H, -0.03 * b.H])
        wr = np.array([s * b.sh_w * 0.62, wrist[0], wrist[1]])
        flare = math.radians(r["flare"])  # elbow flare away from the body (15-55 deg)
        J["el_" + k], J["wr_" + k] = two_bone(J["sh_" + k], wr, b.uarm, b.farm,
                                              -Z * math.cos(flare) + s * X * math.sin(flare))
    J["head_up"] = axis
    return J


def _curl(u, r, b: Body):
    J = standing_feet(b, r["half_stance"])
    for s, k in ((1, "l"), (-1, "r")):
        J["hip_" + k] = np.array([s * b.hip_w / 2, b.ankle_h + b.shank + b.thigh - 0.005 * b.H, 0.0])
        J["kn_" + k] = np.array([s * (b.hip_w / 2 + r["half_stance"]) / 2, b.ankle_h + b.shank, 0.01 * b.H])
    lean = r["base_lean"] - r["swing"] * math.sin(math.pi * u)  # swings back while lifting
    upper_body(J, b, lean)
    el = r["top"] + u * (r["bottom"] - r["top"])
    drift = r["drift"] * u
    for s, k in ((1, "l"), (-1, "r")):
        down = sag(lean + 180 - drift)  # pointing down, rotated forward by drift
        down = _n(down + s * X * 0.08)
        J["el_" + k] = J["sh_" + k] + down * b.uarm
        flex = 180 - el
        fdir = _n(down * math.cos(math.radians(flex)) + sag(lean + 90 - drift) * math.sin(math.radians(flex)))
        J["wr_" + k] = J["el_" + k] + fdir * b.farm
    J["head_up"] = sag(lean)
    return J


def _press(u, r, b: Body):
    J = standing_feet(b, r["half_stance"])
    for s, k in ((1, "l"), (-1, "r")):
        J["hip_" + k] = np.array([s * b.hip_w / 2, b.ankle_h + b.shank + b.thigh - 0.005 * b.H, 0.0])
        J["kn_" + k] = np.array([s * (b.hip_w / 2 + r["half_stance"]) / 2, b.ankle_h + b.shank, 0.01 * b.H])
    lean = r["base_lean"] - r["arch"] * u  # arching = leaning back
    upper_body(J, b, lean)
    alpha = r["rack"] + u * (r["top"] - r["rack"])  # upper-arm elevation == elbow angle with a vertical forearm
    up = sag(lean)
    for s, k in ((1, "l"), (-1, "r")):
        plane = _n(s * X * math.cos(math.radians(30)) + sag(lean + 90) * math.sin(math.radians(30)))
        d = _n(-up * math.cos(math.radians(alpha)) + plane * math.sin(math.radians(alpha)))
        J["el_" + k] = J["sh_" + k] + d * b.uarm
        J["wr_" + k] = J["el_" + k] + _n(up - s * X * 0.05 * u) * b.farm
    J["head_up"] = sag(lean)
    return J


POSE = {"squat": _squat, "lunge": _lunge, "pushup": _pushup, "curl": _curl, "press": _press}


# --------------------------------------------------------------------------------------------
# per-rep parameters and fault labels
# --------------------------------------------------------------------------------------------

def _pick(rng, clean, fault, active):
    return float(rng.uniform(*(fault if active else clean)))


FAULTS = {
    "squat": ["shallow_depth", "forward_lean", "knee_valgus", "heel_lift", "rushed_descent", "no_lockout"],
    "pushup": ["hip_sag", "hip_pike", "shallow_pushup", "no_lockout"],
    "curl": ["elbow_drift", "torso_swing", "partial_extension", "partial_curl"],
    "press": ["short_press", "back_arch"],
    "lunge": ["shallow_lunge", "torso_lean_lunge"],
}


def rep_params(ex: str, rng, set_p: dict, b: Body, q: float) -> dict:
    """Sample one rep. q = per-fault probability for this set."""
    on = {f: rng.random() < q for f in FAULTS[ex]}
    if ex == "pushup" and on["hip_sag"] and on["hip_pike"]:
        on["hip_pike"] = False
    r = dict(set_p)
    r["faults_injected"] = [f for f, v in on.items() if v]
    if ex == "squat":
        r["bottom"] = _pick(rng, (42, 64), (95, 118), on["shallow_depth"])
        r["top"] = _pick(rng, (168, 178), (152, 158), on["no_lockout"])
        r["dorsi"] = float(rng.uniform(22, 36))
        r["extra_lean"] = _pick(rng, (0, 6), (18, 32), on["forward_lean"])
        r["valgus"] = _pick(rng, (0.0, 0.08), (0.3, 0.5), on["knee_valgus"])
        r["heel"] = _pick(rng, (0, 0), (16, 26), on["heel_lift"])
        r["ecc"] = _pick(rng, (0.9, 2.0), (0.3, 0.45), on["rushed_descent"])
        r["con"] = float(rng.uniform(0.7, 1.4))
    elif ex == "pushup":
        r["bottom"] = _pick(rng, (55, 85), (102, 118), on["shallow_pushup"])
        r["top"] = _pick(rng, (165, 178), (146, 152), on["no_lockout"])
        sag_mag = b.H * float(rng.uniform(0.085, 0.13))
        r["sag"] = sag_mag if on["hip_sag"] else -sag_mag if on["hip_pike"] else b.H * float(rng.uniform(-0.012, 0.012))
        r["ecc"], r["con"] = float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.6, 1.2))
    elif ex == "curl":
        r["top"] = _pick(rng, (160, 175), (132, 145), on["partial_extension"])
        r["bottom"] = _pick(rng, (35, 60), (78, 95), on["partial_curl"])
        r["drift"] = _pick(rng, (0, 14), (36, 50), on["elbow_drift"])
        r["swing"] = _pick(rng, (0, 5), (16, 26), on["torso_swing"])
        r["ecc"], r["con"] = float(rng.uniform(1.0, 2.0)), float(rng.uniform(0.7, 1.4))
    elif ex == "press":
        r["rack"] = float(rng.uniform(62, 80))
        r["top"] = _pick(rng, (164, 177), (147, 154), on["short_press"])
        r["arch"] = _pick(rng, (0, 5), (16, 26), on["back_arch"])
        r["ecc"], r["con"] = float(rng.uniform(0.9, 1.8)), float(rng.uniform(0.7, 1.4))
    elif ex == "lunge":
        r["top"] = float(rng.uniform(165, 176))
        r["bottom"] = _pick(rng, (80, 100), (113, 128), on["shallow_lunge"])
        r["extra_lean"] = _pick(rng, (0, 8), (24, 38), on["torso_lean_lunge"])
        r["valgus"] = 0.0
        r["ecc"], r["con"] = float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.8, 1.5))
    r["pause"] = float(rng.uniform(0.0, 0.35))
    r["rest"] = float(rng.uniform(0.2, 0.9))
    return r


def set_params(ex: str, rng, b: Body) -> dict:
    s = {"half_stance": b.H * float(rng.uniform(0.07, 0.11))}
    if ex == "lunge":
        s.update(lead=int(rng.choice([1, -1])), stride=b.H * float(rng.uniform(0.5, 0.65)),
                 base_lean=float(rng.uniform(0, 8)))
    if ex in ("curl", "press"):
        s["base_lean"] = float(rng.uniform(-2, 3))
    if ex == "pushup":
        s["wrist_z"] = 0.0  # set below once the top pose is known
        s["flare"] = float(rng.uniform(15, 55))
    return s


def true_faults(ex: str, rep_frames: list[np.ndarray], r: dict, b: Body) -> tuple[list[str], dict]:
    """Fault labels from the TRUE 3-D pose of one rep, with the thresholds in pose/rules.ts."""
    m: dict = {}
    if ex in ("squat", "lunge"):
        kl = [interior(P[L_HIP], P[L_KNEE], P[L_ANK]) for P in rep_frames]
        kr = [interior(P[R_HIP], P[R_KNEE], P[R_ANK]) for P in rep_frames]
        knee = [min(a, c) for a, c in zip(kl, kr)] if ex == "lunge" else [(a + c) / 2 for a, c in zip(kl, kr)]
        m["min_angle"], m["max_angle"] = min(knee), max(knee)
        lean = [lean_from_vertical((P[L_HIP] + P[R_HIP]) / 2, (P[L_SH] + P[R_SH]) / 2) for P in rep_frames]
        m["torso_lean_max"] = max(lean)
    if ex == "squat":
        m["depth"] = max((((P[L_KNEE] + P[R_KNEE]) / 2)[1] - ((P[L_HIP] + P[R_HIP]) / 2)[1]) / b.thigh for P in rep_frames)
        m["valgus_min"] = min(abs(P[L_KNEE][0] - P[R_KNEE][0]) / max(1e-6, abs(P[L_ANK][0] - P[R_ANK][0])) for P in rep_frames)
        m["heel_rise"] = max(max(0.0, (P[L_HEEL][1] - P[L_TOE][1])) / b.foot for P in rep_frames)
        m["ecc_s"] = r["ecc"]
        f = []
        if m["depth"] < -0.15: f.append("shallow_depth")
        if m["torso_lean_max"] > 45: f.append("forward_lean")
        if m["valgus_min"] < 0.8: f.append("knee_valgus")
        if m["heel_rise"] > 0.2: f.append("heel_lift")
        if m["ecc_s"] < 0.5: f.append("rushed_descent")
        if m["max_angle"] < 160: f.append("no_lockout")
        return f, m
    if ex == "lunge":
        f = []
        if m["min_angle"] > 108: f.append("shallow_lunge")
        if m["torso_lean_max"] > 22: f.append("torso_lean_lunge")
        return f, m
    side = (L_SH, L_EL, L_WR, L_HIP)
    el = [interior(P[side[0]], P[side[1]], P[side[2]]) for P in rep_frames]
    m["min_angle"], m["max_angle"] = min(el), max(el)
    if ex == "pushup":
        hl = [interior((P[L_SH] + P[R_SH]) / 2, (P[L_HIP] + P[R_HIP]) / 2, (P[L_ANK] + P[R_ANK]) / 2) for P in rep_frames]
        m["hip_line_min"] = min(hl)
        f = []
        if m["hip_line_min"] < 160: f.append("hip_sag" if r["sag"] > 0 else "hip_pike")
        if m["min_angle"] > 95: f.append("shallow_pushup")
        if m["max_angle"] < 155: f.append("no_lockout")
        return f, m
    lean = [lean_from_vertical((P[L_HIP] + P[R_HIP]) / 2, (P[L_SH] + P[R_SH]) / 2) for P in rep_frames]
    m["torso_lean_max"], m["torso_sway"] = max(lean), max(lean) - min(lean)
    if ex == "curl":
        m["elbow_drift_max"] = max(interior(P[L_EL], P[L_SH], P[L_HIP]) for P in rep_frames)
        f = []
        if m["elbow_drift_max"] > 28: f.append("elbow_drift")
        if m["torso_sway"] > 11: f.append("torso_swing")
        if m["max_angle"] < 150: f.append("partial_extension")
        if m["min_angle"] > 70: f.append("partial_curl")
        return f, m
    f = []  # press
    if m["max_angle"] < 158: f.append("short_press")
    if m["torso_lean_max"] > 13: f.append("back_arch")
    return f, m


def _ease(n):
    t = np.linspace(0, 1, max(2, n))
    return (1 - np.cos(np.pi * t)) / 2


def exercise_sequence(ex: str, rng, b: Body, n_reps: int, q: float):
    """3-D frames + ground truth for one set. Returns (frames[T,33,3], gt dict)."""
    sp = set_params(ex, rng, b)
    if ex == "pushup":  # wrists under the shoulders in the top position
        leg = b.thigh + b.shank
        body = leg + b.trunk
        top_h = 0.98 * (b.uarm + b.farm)
        horiz = math.sqrt(max(body ** 2 - (top_h + 0.02 * b.H - b.ankle_h * 1.6) ** 2, 0.01))
        sp["wrist_z"] = horiz + float(rng.uniform(-0.03, 0.05)) * b.H
    frames, active, reps = [], [], []
    t = 0.0
    lead = float(rng.uniform(1.0, 2.2))
    first = rep_params(ex, rng, sp, b, q)
    rest_pose = POSE[ex](0.0, first, b)
    for _ in range(int(lead * FPS)):
        frames.append(assemble(rest_pose, b)); active.append(False)
    r = first
    for i in range(n_reps):
        if i:
            r = rep_params(ex, rng, sp, b, q)
        us = np.concatenate([_ease(int(r["ecc"] * FPS)), np.ones(int(r["pause"] * FPS)),
                             1 - _ease(int(r["con"] * FPS)), np.zeros(int(r["rest"] * FPS))])
        start = len(frames) / FPS
        rep_P = []
        turn_i = int(np.argmax(us))
        for j, u in enumerate(us):
            P = assemble(POSE[ex](float(u), r, b), b)
            frames.append(P); active.append(True); rep_P.append(P)
        end = start + (int(r["ecc"] * FPS) + int(r["pause"] * FPS) + int(r["con"] * FPS)) / FPS
        faults, m = true_faults(ex, rep_P, r, b)
        reps.append({"start": round(start, 3), "turn": round(start + turn_i / FPS, 3), "end": round(end, 3),
                     "faults": faults, "metrics": {k: round(float(v), 2) for k, v in m.items()}})
    tail = assemble(POSE[ex](0.0, r, b), b)
    for _ in range(int(float(rng.uniform(0.8, 1.6)) * FPS)):
        frames.append(tail); active.append(False)
    return np.array(frames), {"reps": reps, "active": active}


# --------------------------------------------------------------------------------------------
# non-exercise movement (distractors)
# --------------------------------------------------------------------------------------------

def _standing(b: Body, hs, lean=2.0, sway=0.0):
    J = standing_feet(b, hs)
    for s, k in ((1, "l"), (-1, "r")):
        J["hip_" + k] = np.array([s * b.hip_w / 2 + sway, b.ankle_h + b.shank + b.thigh - 0.005 * b.H, 0.0])
        J["kn_" + k] = np.array([s * (b.hip_w / 2 + hs) / 2 + sway / 2, b.ankle_h + b.shank, 0.01 * b.H])
    upper_body(J, b, lean, sway * 0.5)
    J["head_up"] = sag(lean)
    return J


def distractor_sequence(kind: str, rng, b: Body, dur: float):
    n = int(dur * FPS)
    hs = b.H * float(rng.uniform(0.06, 0.1))
    frames = []
    ph = float(rng.uniform(0, 2 * math.pi))
    period = float(rng.uniform(0.9, 1.6))
    # sparse random "events" for wave / hinge
    events = np.zeros(n)
    if kind in ("arm_wave", "hinge"):
        t = float(rng.uniform(0.3, 1.5))
        while t < dur - 1:
            d = float(rng.uniform(1.0, 2.5))
            i0, i1 = int(t * FPS), min(n, int((t + d) * FPS))
            events[i0:i1] = np.sin(np.linspace(0, math.pi, i1 - i0)) * float(rng.uniform(0.5, 1.0))
            t += d + float(rng.uniform(0.3, 2.0))
    elev = float(rng.uniform(60, 160))
    for i in range(n):
        tt = i / FPS
        if kind == "idle":
            J = _standing(b, hs, lean=2 + math.sin(tt * 0.7 + ph), sway=0.01 * b.H * math.sin(tt * 0.5 + ph))
            arms_at_sides(J, b, abd=8 + 4 * math.sin(tt * 0.3), el_flex=10 + 25 * max(0, math.sin(tt * 0.4 + ph)))
        elif kind == "walk" or kind == "march":
            J = _standing(b, hs)
            phase = 2 * math.pi * tt / period + ph
            fwd_shift = (tt * float(0.0 if kind == "march" else 0.9)) % (2.0) - 1.0 if kind == "walk" else 0.0
            for s, k in ((1, "l"), (-1, "r")):
                lift = max(0.0, math.sin(phase + (0 if s > 0 else math.pi)))
                hip = J["hip_" + k]
                if kind == "march":
                    hf, kf = 70 * lift, 85 * lift
                else:
                    sw = math.sin(phase + (0 if s > 0 else math.pi))
                    hf, kf = 25 * sw, 45 * lift
                kn = hip + sag(180 - hf) * b.thigh
                an = kn + sag(180 - hf + kf) * b.shank
                J["kn_" + k], J["an_" + k] = kn, an
                J["heel_" + k] = an + np.array([0, -b.ankle_h * 0.9, -0.25 * b.foot])
                J["toe_" + k] = an + np.array([0, -b.ankle_h * 0.9, 0.75 * b.foot])
            ground = min(J["toe_l"][1], J["toe_r"][1], J["heel_l"][1], J["heel_r"][1])
            for key in list(J):
                if key != "head_up":
                    J[key] = J[key] - Y * ground + X * fwd_shift
            arms_at_sides(J, b, abd=6, el_flex=15 + 20 * abs(math.sin(phase)))
            if kind == "walk":  # walk across the frame: rotate the body 90 deg about Y
                c = (J["hip_l"] + J["hip_r"]) / 2
                for key in list(J):
                    if key == "head_up":
                        J[key] = np.array([J[key][2], J[key][1], -J[key][0]])
                        continue
                    v = J[key] - c
                    J[key] = c + np.array([v[2], v[1], -v[0]])
        elif kind == "arm_wave":
            J = _standing(b, hs)
            e = events[i]
            for s, k in ((1, "l"), (-1, "r")):
                a = 10 + e * elev
                plane = _n(s * X + 0.5 * Z)
                d = _n(-Y * math.cos(math.radians(a)) + plane * math.sin(math.radians(a)))
                J["el_" + k] = J["sh_" + k] + d * b.uarm
                flex = 15 + 70 * e * (0.5 + 0.5 * math.sin(tt * 6 + s))
                f = _n(d * math.cos(math.radians(flex)) + Y * math.sin(math.radians(flex)))
                J["wr_" + k] = J["el_" + k] + f * b.farm
        else:  # hinge: bend at the hips to touch the shins, knees soft
            e = events[i]
            J = standing_feet(b, hs)
            for s, k in ((1, "l"), (-1, "r")):
                an = J["an_" + k]
                kn = an + sag(8 * e) * b.shank
                hip = kn + sag(8 * e - 25 * e) * b.thigh
                hip[0] = s * b.hip_w / 2
                J["kn_" + k], J["hip_" + k] = kn, hip
            lean = 3 + 80 * e
            upper_body(J, b, lean)
            arms_at_sides(J, b, lean_deg=0.0, abd=5, el_flex=10)
            J["head_up"] = sag(lean)
        frames.append(assemble(J, b))
    return np.array(frames), {"reps": [], "active": [True] * n}


# --------------------------------------------------------------------------------------------
# camera + detector error model
# --------------------------------------------------------------------------------------------

@dataclass
class Camera:
    yaw: float      # deg; 0 = in front of the subject, 90 = subject's left side
    height: float   # m above the floor
    dist: float     # m
    width: int
    height_px: int
    fov: float      # vertical field of view, deg


def sample_camera(rng, yaw: float | None = None) -> Camera:
    portrait = rng.random() < 0.3
    w, h = (720, 1280) if portrait else (1280, 720)
    return Camera(yaw=float(rng.uniform(-180, 180)) if yaw is None else float(yaw),
                  height=float(rng.uniform(0.5, 1.5)), dist=float(rng.uniform(2.8, 5.0)),
                  width=w, height_px=h, fov=float(rng.uniform(45, 65)))


def project(P3: np.ndarray, cam: Camera, center: np.ndarray):
    """P3: [T, 33, 3] world -> pixel coords [T, 33, 2] and depth along the view axis."""
    yaw = math.radians(cam.yaw)
    pos = center + np.array([math.sin(yaw), 0, math.cos(yaw)]) * cam.dist
    pos[1] = cam.height
    f = _n(center - pos)
    r = _n(np.cross(f, Y))
    u = np.cross(r, f)
    rel = P3 - pos
    xc, yc, zc = rel @ r, rel @ u, rel @ f
    F = (cam.height_px / 2) / math.tan(math.radians(cam.fov) / 2)
    px = np.stack([cam.width / 2 + F * xc / zc, cam.height_px / 2 - F * yc / zc], -1)
    return px, -f  # -f: direction from the subject to the camera


def observe(P3: np.ndarray, cam: Camera, rng, *, noise: float = 0.006, occlusion_p: float = 0.25,
            swap_rate: float = 0.03, dropout_rate: float = 0.01, fill: float = 0.75,
            out_of_frame_p: float = 0.05) -> tuple[list, np.ndarray, dict]:
    """Project + frame + MediaPipe-like errors. Returns (frames for the TS pipeline, clean 2-D px, info)."""
    T = P3.shape[0]
    center = P3[:, [L_HIP, R_HIP, L_SH, R_SH]].reshape(-1, 3).mean(0)
    px, to_cam = project(P3, cam, center)
    # framing: scale/translate the image so the whole movement fills `fill` of the frame
    lo, hi = px.reshape(-1, 2).min(0), px.reshape(-1, 2).max(0)
    s = fill * min(cam.width / (hi[0] - lo[0] + 1e-6), cam.height_px / (hi[1] - lo[1] + 1e-6))
    cut = rng.random() < out_of_frame_p   # badly framed: part of the body leaves the frame
    shift = np.array([cam.width, cam.height_px]) / 2 - s * (lo + hi) / 2
    if cut:
        shift[1] += cam.height_px * float(rng.uniform(0.15, 0.3))
    px = px * s + shift
    body_h = s * (hi[1] - lo[1])

    # visibility: self-occlusion from the lateral normal (left landmarks face +X, right -X) and the face
    lat_dot = float(np.dot(X, to_cam))
    face_dot = float(np.dot(Z, to_cam))
    vis = np.full((T, 33), 0.97)
    vis[:, LEFT_IDX] -= 0.42 * max(0.0, -lat_dot)
    vis[:, RIGHT_IDX] -= 0.42 * max(0.0, lat_dot)
    vis[:, :11] -= 0.45 * max(0.0, -face_dot)
    vis += rng.normal(0, 0.03, vis.shape)

    # temporally correlated pixel noise (AR(1)), larger on occluded landmarks
    sigma = noise * body_h
    eps = np.zeros((T, 33, 2))
    e = rng.normal(0, sigma, (33, 2))
    for t in range(T):
        e = 0.6 * e + 0.8 * rng.normal(0, sigma, (33, 2))
        eps[t] = e
    eps *= (1 + 1.5 * (vis < 0.7))[..., None]
    obs = px + eps

    info = {"occluder": None, "swaps": [], "dropouts": 0, "cut": bool(cut), "body_h_px": round(float(body_h), 1)}
    if rng.random() < occlusion_p:  # an object/person covers a part of the frame for a while
        dur = int(float(rng.uniform(0.8, 3.0)) * FPS)
        t0 = int(rng.integers(0, max(1, T - dur)))
        kind = rng.choice(["lower", "strip"])
        if kind == "lower":
            y0 = float(np.percentile(px[:, :, 1], float(rng.uniform(55, 75))))
            box = (0, y0, cam.width, cam.height_px)
        else:
            x0 = float(rng.uniform(lo[0] * s + shift[0], hi[0] * s + shift[0]))
            box = (x0 - 0.12 * body_h, 0, x0 + 0.12 * body_h, cam.height_px)
        inside = ((obs[..., 0] >= box[0]) & (obs[..., 0] <= box[2]) & (obs[..., 1] >= box[1]) & (obs[..., 1] <= box[3]))
        win = np.zeros(T, bool)
        win[t0:t0 + dur] = True
        m = inside & win[:, None]
        vis[m] = rng.uniform(0.03, 0.35, int(m.sum()))
        walk = np.cumsum(rng.normal(0, 0.02 * body_h, (T, 33, 2)), 0)
        obs[m] += walk[m]
        info["occluder"] = {"kind": str(kind), "t0": round(t0 / FPS, 2), "t1": round((t0 + dur) / FPS, 2)}

    side_view = abs(lat_dot) > 0.8
    if side_view:  # left/right identity swaps, a known failure in side views
        t = 0
        while t < T:
            if rng.random() < swap_rate / FPS * 3:
                d = int(rng.integers(3, 10))
                obs[t:t + d, LEFT_IDX], obs[t:t + d, RIGHT_IDX] = obs[t:t + d, RIGHT_IDX].copy(), obs[t:t + d, LEFT_IDX].copy()
                vis[t:t + d, LEFT_IDX], vis[t:t + d, RIGHT_IDX] = vis[t:t + d, RIGHT_IDX].copy(), vis[t:t + d, LEFT_IDX].copy()
                info["swaps"].append([round(t / FPS, 2), round((t + d) / FPS, 2)])
                t += d
            t += 1

    outside = (obs[..., 0] < 0) | (obs[..., 0] > cam.width) | (obs[..., 1] < 0) | (obs[..., 1] > cam.height_px)
    vis[outside] *= 0.3
    vis = np.clip(vis, 0.0, 1.0)

    drop = np.zeros(T, bool)
    t = 0
    while t < T:
        if rng.random() < dropout_rate:
            d = int(rng.integers(1, 6))
            drop[t:t + d] = True
            t += d
        t += 1
    info["dropouts"] = int(drop.sum())

    out = []
    for t in range(T):
        if drop[t]:
            out.append({"t": round(t / FPS, 3), "lm": None})
            continue
        lm = [[round(float(obs[t, i, 0] / cam.width), 4), round(float(obs[t, i, 1] / cam.height_px), 4),
               round(float(vis[t, i]), 3)] for i in range(33)]
        out.append({"t": round(t / FPS, 3), "lm": lm})
    clean = px / np.array([cam.width, cam.height_px])
    return out, clean, info


def view_name(yaw: float) -> str:
    a = abs(((yaw + 180) % 360) - 180)  # 0..180
    if a <= 22.5:
        return "front"
    if a >= 157.5:
        return "rear"
    if 67.5 <= a <= 112.5:
        return "side"
    return "oblique" if a < 90 else "rear-oblique"


def make_sequence(seq_id: str, rng, cls: str | None = None, yaw: float | None = None, *,
                  noise: tuple[float, float] = (0.003, 0.01), occlusion_p: float = 0.25,
                  fault_q: tuple[float, float] = (0.0, 0.35)) -> dict:
    """One labelled sequence. cls: an exercise, a distractor kind, or None (random)."""
    if cls is None:
        cls = str(rng.choice(EXERCISES + DISTRACTORS, p=[0.15] * 5 + [0.05] * 5))
    b = Body.sample(rng)
    if cls in EXERCISES:
        q = float(rng.uniform(*fault_q))
        P3, gt = exercise_sequence(cls, rng, b, int(rng.integers(4, 13)), q)
        label = cls
    else:
        P3, gt = distractor_sequence(cls, rng, b, float(rng.uniform(8, 18)))
        label = "other"
    cam = sample_camera(rng, yaw)
    frames, clean, info = observe(P3, cam, rng, noise=float(rng.uniform(*noise)), occlusion_p=occlusion_p)
    return {
        "id": seq_id, "label": label, "activity": cls, "width": cam.width, "height": cam.height_px, "fps": FPS,
        "camera": {"yaw": round(cam.yaw, 1), "view": view_name(cam.yaw), "height_m": round(cam.height, 2),
                   "dist_m": round(cam.dist, 2), "fov": round(cam.fov, 1)},
        "body_height_m": round(b.H, 2), "observation": info,
        "gt_reps": len(gt["reps"]), "reps": gt["reps"], "active": gt["active"],
        "frames": frames, "clean": np.round(clean, 4).tolist(),
    }
