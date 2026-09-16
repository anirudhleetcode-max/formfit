import re
from pathlib import Path

import numpy as np

from app.services.analytics import change_point, detect_fatigue, fatigue_index, trend_per_week
from app.services.form_model import get_model
from ml import rules
from ml.features import FEATURES, NEUTRAL, vector
from ml.simulate import generate, simulate_rep


def test_vector_fills_missing_and_nan():
    v = vector({"min_angle": 80, "rom": float("nan"), "depth": "x"})
    assert len(v) == len(FEATURES)
    assert v[0] == 80 and v[FEATURES.index("rom")] == NEUTRAL["rom"] and v[FEATURES.index("depth")] == NEUTRAL["depth"]


def test_simulator_is_deterministic_and_balanced():
    X1, y1 = generate("squat", 300, seed=5)
    X2, y2 = generate("squat", 300, seed=5)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    for ex in ["squat", "pushup", "curl", "press", "lunge"]:
        _, y = generate(ex, 400, seed=1)
        assert 0.1 < y.mean() < 0.8, ex


def test_simulated_shallow_squat_is_labelled_faulty():
    rng = np.random.default_rng(0)
    bad = [simulate_rep("squat", rng, fault_rate=0.9)[1] for _ in range(200)]
    good = [simulate_rep("squat", rng, fault_rate=0.0)[1] for _ in range(200)]
    assert np.mean(good) > np.mean(bad) + 0.3


def test_model_separates_good_and_faulty():
    m = get_model()
    assert m.ready
    rng = np.random.default_rng(99)
    for ex in ["squat", "curl", "pushup"]:
        feats, labels = [], []
        for _ in range(300):
            f, lab, _ = simulate_rep(ex, rng)
            feats.append(f)
            labels.append(lab)
        s = np.array(m.score(ex, feats), float)
        labels = np.array(labels)
        assert s[labels == 1].mean() > s[labels == 0].mean() + 30, ex
    assert m.score("unknown", [{}]) == [None]


def test_fatigue_detected_on_clear_slowdown():
    reps = [{"con_s": 0.8, "rom": 110, "score": 95} for _ in range(6)]
    reps += [{"con_s": 1.6 + 0.05 * i, "rom": 85, "score": 65} for i in range(6)]
    for i, r in enumerate(reps):  # small jitter so variance is non-zero
        r["con_s"] += 0.02 * ((i % 3) - 1)
    out = detect_fatigue(reps)
    assert out["detected"] and out["onset_rep"] == 7
    assert out["slopes"]["con_s"] > 0 and out["slopes"]["rom"] < 0


def test_no_fatigue_on_steady_set_or_short_set():
    rng = np.random.default_rng(1)
    steady = [{"con_s": 0.9 + rng.normal(0, 0.05), "rom": 100 + rng.normal(0, 2), "score": 90} for _ in range(12)]
    assert not detect_fatigue(steady)["detected"]
    short = [{"con_s": 0.8, "rom": 100, "score": 90}, {"con_s": 3, "rom": 40, "score": 20}]
    out = detect_fatigue(short)
    assert not out["detected"] and len(out["index"]) == 2
    assert detect_fatigue([])["index"] == []


def test_fatigue_index_and_change_point():
    idx = fatigue_index([1, 1, 1, 2, 2, 2], [100] * 6, [90] * 6)
    assert np.allclose(idx[:3], 0) and np.all(idx[3:] > 0.3)
    k, shift, t = change_point(np.array([0, 0.01, 0, 0.5, 0.51, 0.5]))
    assert k == 3 and shift > 0.4 and t > 3


def test_trend_per_week():
    assert trend_per_week([0, 7, 14], [70, 75, 80]) == 5.0
    assert trend_per_week([0, 1], [1, 2]) is None


def test_python_rules_match_frontend_thresholds():
    ts = (Path(__file__).resolve().parents[2] / "frontend/src/pose/rules.ts").read_text()
    for needle in ["depth < -0.15", "torsoLeanMax > 45", "valgusMin < 0.8", "heelRise > 0.2",
                   "eccS < 0.5", "hipLineMin < 160", "elbowDriftMax > 28", "torsoSway > 11",
                   "maxAngle < 158", "torsoLeanMax > 13", "minAngle > 108", "torsoLeanMax > 22"]:
        assert needle in ts, needle
    score, faults = rules.evaluate("squat", {**NEUTRAL, "depth": -0.5, "torso_lean_max": 60})
    assert faults == ["shallow_depth", "forward_lean"] and score == 50
    for code, pen in re.findall(r'(\w+): (\d+),', ts.split("PENALTY")[1].split("}")[0]):
        from ml.features import FAULT_PENALTY
        assert FAULT_PENALTY[code] == int(pen), code
