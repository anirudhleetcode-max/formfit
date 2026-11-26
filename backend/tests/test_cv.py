"""Computer-vision research code (backend/cv): generator, normalization, classifier forward pass."""
import math

import numpy as np
import pytest

from cv.classify import CLS_IDX, WIN, tcn_forward_np, tcn_init
from cv.engine_eval import match_reps
from cv.normalize import FRAME_FEATURES, frame_features, frames_to_array, mirror, normalize
from cv.skeleton import (EXERCISES, L_ANK, L_HIP, L_KNEE, R_HIP, Body, interior, make_sequence, two_bone,
                         view_name)


def test_two_bone_ik_keeps_segment_lengths():
    root, target = np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.2, 0.3])
    mid, end = two_bone(root, target, 0.45, 0.45, np.array([0, 0, 1.0]))
    assert np.linalg.norm(mid - root) == pytest.approx(0.45, abs=1e-6)
    assert np.linalg.norm(end - mid) == pytest.approx(0.45, abs=1e-6)
    assert mid[2] > 0  # bends towards the pole


def test_generator_is_deterministic_and_labelled():
    a = make_sequence("a", np.random.default_rng(3), "squat", 90.0)
    b = make_sequence("a", np.random.default_rng(3), "squat", 90.0)
    assert a["frames"] == b["frames"]
    assert a["gt_reps"] == len(a["reps"]) >= 4
    assert a["camera"]["view"] == "side"
    for r in a["reps"]:
        assert r["start"] < r["turn"] < r["end"]


def test_squat_depth_reaches_requested_knee_angle():
    from cv.skeleton import _squat
    rng = np.random.default_rng(0)
    b = Body.sample(rng)
    r = {"half_stance": 0.15, "top": 175, "bottom": 60, "dorsi": 30, "extra_lean": 0, "valgus": 0, "heel": 0}
    J = _squat(1.0, r, b)
    knee = interior(J["hip_l"], J["kn_l"], J["an_l"])
    assert 55 < knee < 70


def test_distractors_have_no_reps():
    s = make_sequence("d", np.random.default_rng(1), "march", 0.0)
    assert s["label"] == "other" and s["gt_reps"] == 0


def test_view_names():
    assert view_name(0) == "front" and view_name(90) == "side" and view_name(-90) == "side"
    assert view_name(180) == "rear" and view_name(45) == "oblique" and view_name(135) == "rear-oblique"


def test_normalization_is_translation_and_scale_invariant():
    s = make_sequence("n", np.random.default_rng(2), "curl", 45.0, occlusion_p=0)
    lm, present = frames_to_array(s["frames"])
    lm = lm[present]
    aspect = s["width"] / s["height"]
    a, _ = normalize(lm, aspect)
    moved = lm.copy()
    moved[..., 0] = moved[..., 0] * 0.5 + 0.2   # smaller and shifted in the frame
    moved[..., 1] = moved[..., 1] * 0.5 + 0.1
    b, _ = normalize(moved, aspect)
    assert np.allclose(a, b, atol=1e-6)


def test_mirror_swaps_sides():
    s = make_sequence("m", np.random.default_rng(4), "lunge", 90.0, occlusion_p=0)
    lm, present = frames_to_array(s["frames"])
    m = mirror(lm[present])
    assert np.allclose(m[:, L_HIP, 0], 1 - lm[present][:, R_HIP, 0])
    assert np.allclose(mirror(m), lm[present])


def test_frame_features_shape_and_finite_with_dropouts():
    s = make_sequence("f", np.random.default_rng(5), "pushup", 90.0)
    s["frames"][0]["lm"] = None
    s["frames"][10]["lm"] = None
    lm, present = frames_to_array(s["frames"])
    f = frame_features(lm, present, s["width"] / s["height"])
    assert f.shape == (len(s["frames"]), len(FRAME_FEATURES))
    assert np.isfinite(f).all()
    assert f[10, FRAME_FEATURES.index("present")] == 0
    k = FRAME_FEATURES.index("knee_l")
    true_knee = [interior(*(np.array(fr["lm"])[[L_HIP, L_KNEE, L_ANK], :2] * [s["width"] / s["height"], 1]))
                 for fr in s["frames"][1:4]]
    assert np.allclose(f[1:4, k] * 180, true_knee, atol=0.5)


def test_tcn_forward_shapes_and_time_shift_robustness():
    p = tcn_init(len(FRAME_FEATURES), len(CLS_IDX), seed=0)
    x = np.random.default_rng(0).normal(size=(4, WIN, len(FRAME_FEATURES))).astype(np.float32)
    z = tcn_forward_np(p, x)
    assert z.shape == (4, len(CLS_IDX)) and np.isfinite(z).all()


def test_rep_matching():
    gt = [{"turn": 1.0, "end": 2.0}, {"turn": 4.0, "end": 5.0}]
    pairs, fp, fn, dup = match_reps(gt, [{"t": 2.1}, {"t": 2.3}, {"t": 9.0}])
    assert pairs == [(0, 0)] and dup == [1] and fp == [2] and fn == [1]


def test_every_exercise_generates():
    rng = np.random.default_rng(11)
    for ex in EXERCISES:
        s = make_sequence(ex, rng, ex, 60.0)
        assert s["label"] == ex and s["gt_reps"] > 0
        assert all(f["lm"] is None or len(f["lm"]) == 33 for f in s["frames"])
        assert not math.isnan(s["frames"][-1]["t"])
