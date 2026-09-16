"""Shared definition of the per-rep feature vector.

The browser (frontend/src/pose/engine.ts) computes exactly these keys for every rep and sends
them to the backend. The simulator produces the same keys, so the classifier trained on
simulated reps can score real reps.
"""

FEATURES_VERSION = "features-v2"  # v2: aspect-corrected x, per-exercise feature subsets

EXERCISES = ("squat", "pushup", "curl", "press", "lunge")

FEATURES = (
    "min_angle",        # primary joint angle at the bottom of the rep (deg)
    "max_angle",        # primary joint angle at the top / lockout (deg)
    "rom",              # max_angle - min_angle (deg)
    "ecc_s",            # eccentric (lowering) time, seconds
    "con_s",            # concentric (lifting) time, seconds
    "torso_lean_max",   # max torso angle from vertical during the rep (deg)
    "torso_sway",       # range (max - min) of torso angle during the rep (deg)
    "hip_line_min",     # min shoulder-hip-ankle angle (deg), 180 = straight line
    "valgus_min",       # min knee-width / ankle-width ratio (1 = knees over ankles, <1 caving)
    "elbow_drift_max",  # max angle between upper arm and torso (deg)
    "depth",            # (hip_y - knee_y) / thigh length at bottom; >0 means hip below knee
    "heel_rise",        # max heel lift as fraction of foot length
)

# Features each exercise's classifier actually uses. Measurements that do not describe the
# movement (e.g. elbow angle to torso in a barbell squat, where the elbows are flared by the bar)
# are excluded so they cannot push real reps out of the training distribution.
_BASE = ("min_angle", "max_angle", "rom", "ecc_s", "con_s")
EXERCISE_FEATURES = {
    "squat": _BASE + ("torso_lean_max", "torso_sway", "valgus_min", "depth", "heel_rise"),
    "pushup": _BASE + ("hip_line_min", "torso_sway"),
    "curl": _BASE + ("torso_lean_max", "torso_sway", "elbow_drift_max"),
    "press": _BASE + ("torso_lean_max", "torso_sway", "hip_line_min"),
    "lunge": _BASE + ("torso_lean_max", "torso_sway", "valgus_min", "depth"),
}

# Neutral value used when a feature does not apply or could not be measured
# (e.g. valgus needs a frontal view). Mirrors NEUTRAL in frontend/src/pose/rules.ts.
NEUTRAL = {
    "min_angle": 90.0, "max_angle": 170.0, "rom": 80.0, "ecc_s": 1.2, "con_s": 1.0,
    "torso_lean_max": 10.0, "torso_sway": 3.0, "hip_line_min": 175.0, "valgus_min": 1.0,
    "elbow_drift_max": 10.0, "depth": 0.0, "heel_rise": 0.0,
}

# Rule penalties (points off 100) — kept in sync with frontend/src/pose/rules.ts. Used by the
# seed script to give demo reps the same rule score the browser would compute.
FAULT_PENALTY = {
    "shallow_depth": 30, "forward_lean": 20, "knee_valgus": 25, "heel_lift": 15,
    "rushed_descent": 10, "no_lockout": 15, "hip_sag": 30, "hip_pike": 20,
    "shallow_pushup": 30, "elbow_drift": 25, "torso_swing": 25, "partial_extension": 20,
    "partial_curl": 20, "back_arch": 30, "short_press": 20, "torso_lean_lunge": 20,
    "shallow_lunge": 30,
}

FAULT_LABEL = {
    "shallow_depth": "Not deep enough", "forward_lean": "Chest dropping", "knee_valgus": "Knees caving",
    "heel_lift": "Heels lifting", "rushed_descent": "Rushed descent", "no_lockout": "No lockout",
    "hip_sag": "Hips sagging", "hip_pike": "Hips piking", "shallow_pushup": "Shallow push-up",
    "elbow_drift": "Elbows drifting", "torso_swing": "Swinging torso", "partial_extension": "Partial extension",
    "partial_curl": "Partial curl", "back_arch": "Lower back arching", "short_press": "Press not locked out",
    "torso_lean_lunge": "Leaning forward", "shallow_lunge": "Lunge too shallow",
}


def vector(feats: dict, keys=FEATURES) -> list[float]:
    out = []
    for k in keys:
        v = feats.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = NEUTRAL[k]
        if v != v:  # NaN
            v = NEUTRAL[k]
        out.append(v)
    return out
