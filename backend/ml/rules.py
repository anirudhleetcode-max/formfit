"""Python mirror of the rep-level form rules in frontend/src/pose/rules.ts (evaluateRep).

Used for two things only: (1) as the rule-based baseline the classifier is compared against in
train.py, and (2) by the seed script to give demo reps a rule score. The browser is the source of
truth for live sessions. Thresholds must stay in sync with rules.ts (a test checks a few).

Known difference: the model features have no signed hip offset, so a broken push-up hip line is
always reported here as `hip_sag` (penalty 30). The browser splits it into sag vs pike
(penalty 20), so for piked push-ups this baseline's score is 10 points lower than the app's.
"""
from .features import FAULT_PENALTY

RULES = {
    "squat": [
        ("shallow_depth", lambda f: f["depth"] < -0.15),
        ("forward_lean", lambda f: f["torso_lean_max"] > 45),
        ("knee_valgus", lambda f: f["valgus_min"] < 0.8),
        ("heel_lift", lambda f: f["heel_rise"] > 0.2),
        ("rushed_descent", lambda f: f["ecc_s"] < 0.5),
        ("no_lockout", lambda f: f["max_angle"] < 160),
    ],
    "pushup": [
        ("hip_sag", lambda f: f["hip_line_min"] < 160),
        ("shallow_pushup", lambda f: f["min_angle"] > 95),
        ("no_lockout", lambda f: f["max_angle"] < 155),
    ],
    "curl": [
        ("elbow_drift", lambda f: f["elbow_drift_max"] > 28),
        ("torso_swing", lambda f: f["torso_sway"] > 11),
        ("partial_extension", lambda f: f["max_angle"] < 150),
        ("partial_curl", lambda f: f["min_angle"] > 70),
    ],
    "press": [
        ("short_press", lambda f: f["max_angle"] < 158),
        ("back_arch", lambda f: f["torso_lean_max"] > 13),
    ],
    "lunge": [
        ("shallow_lunge", lambda f: f["min_angle"] > 108),
        ("torso_lean_lunge", lambda f: f["torso_lean_max"] > 22),
        ("knee_valgus", lambda f: f["valgus_min"] < 0.8),
    ],
}


def evaluate(exercise: str, feats: dict) -> tuple[int, list[str]]:
    faults = [code for code, test in RULES[exercise] if test(feats)]
    score = max(0, 100 - sum(FAULT_PENALTY[c] for c in faults))
    return score, faults
