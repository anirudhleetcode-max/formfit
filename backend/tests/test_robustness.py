"""Robustness: malformed / empty / huge / low-confidence input, missing or broken model, secrets."""
import uuid

import numpy as np
import pytest

from app.config import Settings, _check_secret
from app.services import form_model as fm
from app.services.analytics import detect_fatigue
from ml import rules
from ml.simulate import simulate_rep


def _user(client):
    r = client.post("/api/auth/register", json={"name": "R", "email": f"r{uuid.uuid4().hex[:8]}@example.com", "password": "secret123"})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _rep(**kw):
    base = {"set": 1, "score": 90, "faults": [], "ecc_s": 1.2, "con_s": 0.9, "rom": 100, "t": 3.0,
            "features": {"min_angle": 60, "max_angle": 175, "rom": 115, "ecc_s": 1.2, "con_s": 0.9,
                         "torso_lean_max": 25, "torso_sway": 20, "valgus_min": 1.0, "depth": 0.1, "heel_rise": 0}}
    base.update(kw)
    return base


def _session(client, h, exercise="squat"):
    return client.post("/api/sessions", json={"exercise": exercise}, headers=h).json()["id"]


def test_empty_session_is_valid(client):
    h = _user(client)
    sid = _session(client, h)
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": [], "duration_s": 0}, headers=h)
    assert r.status_code == 200
    s = r.json()["summary"]
    assert s["total_reps"] == 0 and s["avg_score"] is None and s["avg_model_score"] is None
    assert r.json()["fatigue"]["detected"] is False


@pytest.mark.parametrize("payload", [
    None, "not json", {"reps": "x", "duration_s": 1}, {"duration_s": 1},
    {"reps": [_rep(ecc_s=-1)], "duration_s": 1},
    {"reps": [_rep(set=0)], "duration_s": 1},
    {"reps": [_rep(abstain_reason="x" * 500)], "duration_s": 1},
    {"reps": [_rep(scored=True, score=None)], "duration_s": 1},
    {"reps": [_rep()] * 501, "duration_s": 1},
])
def test_malformed_finish_is_422(client, payload):
    h = _user(client)
    sid = _session(client, h)
    kw = {"content": payload} if isinstance(payload, str) else {"json": payload}
    r = client.post(f"/api/sessions/{sid}/finish", headers={**h, "Content-Type": "application/json"}, **kw)
    assert r.status_code == 422, r.text
    assert "detail" in r.json()


def test_chunked_body_over_limit_is_rejected(client):
    h = _user(client)

    def gen():
        for _ in range(12):
            yield b" " * 100_000

    r = client.post("/api/sessions", content=gen(), headers={**h, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_non_finite_features_are_dropped(client):
    h = _user(client)
    sid = _session(client, h)
    body = '{"reps": [{"set": 1, "score": 80, "faults": [], "ecc_s": 1, "con_s": 1, "rom": 90, "t": 1, ' \
           '"features": {"min_angle": NaN, "rom": Infinity, "depth": 0.1}}], "duration_s": 5}'
    r = client.post(f"/api/sessions/{sid}/finish", content=body, headers={**h, "Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["reps"][0]["features"] == {"depth": 0.1}


def test_low_confidence_reps_are_counted_but_not_scored(client):
    h = _user(client)
    sid = _session(client, h)
    reps = [_rep(t=i, confidence=0.9, scored=True) for i in range(3)]
    reps += [_rep(t=10 + i, score=None, scored=False, confidence=0.2, faults=["shallow_depth"],
                  abstain_reason="Can't see your knees clearly") for i in range(2)]
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": reps, "duration_s": 30}, headers=h)
    assert r.status_code == 200, r.text
    d = r.json()
    s = d["summary"]
    assert s["total_reps"] == 5 and s["scored_reps"] == 3 and s["unscored_reps"] == 2
    assert s["avg_score"] == 90.0 and s["avg_confidence"] == pytest.approx(0.62)
    low = [x for x in d["reps"] if not x["scored"]]
    assert all(x["score"] is None and x["faults"] == [] and x["model_score"] is None
               and x["model_status"] == "not_scored" for x in low)
    assert low[0]["abstain_reason"] == "Can't see your knees clearly"
    assert d["sets"][0]["scored_reps"] == 3


def test_old_clients_without_confidence_fields_still_work(client):
    h = _user(client)
    sid = _session(client, h)
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": [_rep()], "duration_s": 3}, headers=h)
    assert r.status_code == 200
    rep = r.json()["reps"][0]
    assert rep["scored"] is True and rep["score"] == 90 and rep["model_status"] in ("ok", "uncertain", "out_of_distribution")


def test_out_of_distribution_rep_gets_no_model_score():
    m = fm.get_model()
    weird = {"min_angle": 5, "max_angle": 179, "rom": 174, "ecc_s": 25, "con_s": 25, "torso_lean_max": 89,
             "torso_sway": 89, "valgus_min": 3.0, "depth": 1.5, "heel_rise": 0.9}
    out = m.predict("squat", [weird])[0]
    assert out["model_status"] == "out_of_distribution" and out["model_score"] is None
    assert "ecc_s" in out["model_ood"]


def test_uncertain_predictions_are_flagged():
    m = fm.get_model()
    rng = np.random.default_rng(3)
    feats = [simulate_rep("squat", rng)[0] for _ in range(400)]
    out = m.predict("squat", feats)
    statuses = {o["model_status"] for o in out}
    assert "uncertain" in statuses and "ok" in statuses
    for o in out:
        if o["model_status"] == "uncertain":
            assert o["model_confidence"] < m.bundle["abstain_conf"]["squat"]


def test_missing_model_file_degrades(tmp_path):
    m = fm.FormModel(tmp_path / "nope.joblib")
    m.load()
    assert not m.ready and "Run `python -m ml.train`" in m.error
    assert m.predict("squat", [{}]) == [{"model_score": None, "model_confidence": None,
                                        "model_status": "unavailable", "model_ood": []}]
    assert m.info()["ready"] is False


def test_corrupt_model_file_degrades(tmp_path):
    bad = tmp_path / "form_model.joblib"
    bad.write_bytes(b"this is not a joblib file")
    m = fm.FormModel(bad)
    m.load()
    assert not m.ready and m.error


def test_api_degrades_without_model(client, monkeypatch, tmp_path):
    broken = fm.FormModel(tmp_path / "missing.joblib")
    broken.load()
    monkeypatch.setattr(fm, "_model", broken)
    health = client.get("/api/health").json()
    assert health["status"] == "degraded" and health["model"] is False and health["model_error"]
    h = _user(client)
    sid = _session(client, h)
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": [_rep()], "duration_s": 3}, headers=h)
    assert r.status_code == 200
    assert r.json()["reps"][0]["model_status"] == "unavailable"
    assert client.get("/api/model", headers=h).json()["ready"] is False


def test_login_rate_limit(client):
    email = f"rl{uuid.uuid4().hex[:6]}@example.com"
    client.post("/api/auth/register", json={"name": "L", "email": email, "password": "secret123"})
    codes = [client.post("/api/auth/login", json={"email": email, "password": "wrong"}).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429
    # even the right password is refused while blocked
    r = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_placeholder_jwt_secret_policy():
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        _check_secret(Settings(env="production", jwt_secret="change-me-in-production"))
    dev = _check_secret(Settings(env="development", jwt_secret=""))
    assert len(dev.jwt_secret) >= 48
    strong = "x" * 40
    assert _check_secret(Settings(env="production", jwt_secret=strong)).jwt_secret == strong


def test_stale_active_sessions_are_abandoned(client):
    import pymongo
    from datetime import datetime, timedelta, timezone
    from app.config import get_settings
    h = _user(client)
    sid = _session(client, h)
    db = pymongo.MongoClient(get_settings().mongo_uri)[get_settings().mongo_db]
    from bson import ObjectId
    db.sessions.update_one({"_id": ObjectId(sid)}, {"$set": {"started_at": datetime.now(timezone.utc) - timedelta(hours=7)}})
    _session(client, h)
    assert db.sessions.find_one({"_id": ObjectId(sid)})["status"] == "abandoned"


def test_fatigue_false_alarm_rate_on_simulated_steady_sets():
    """Held-out seed (not the one used when choosing the thresholds)."""
    rng = np.random.default_rng(2024)
    flags = 0
    trials = 120
    for _ in range(trials):
        reps = []
        for _ in range(12):
            f, _, _ = simulate_rep("squat", rng, fatigue=0.0, fault_rate=0.1, view="side")
            reps.append({"con_s": f["con_s"], "rom": f["rom"], "score": rules.evaluate("squat", f)[0]})
        flags += detect_fatigue(reps)["detected"]
    assert flags / trials <= 0.12


def test_fatigue_handles_unscored_reps():
    reps = [{"con_s": 0.8, "rom": 110, "score": None} for _ in range(6)]
    reps += [{"con_s": 1.7 + 0.03 * i, "rom": 80, "score": None} for i in range(6)]
    for i, r in enumerate(reps):
        r["con_s"] += 0.02 * ((i % 3) - 1)
    out = detect_fatigue(reps)
    assert out["detected"] and out["onset_rep"] == 7 and out["slopes"]["score"] == 0.0


def test_fatigue_eval_reports_detection_and_false_alarm_rates():
    """Small version of `python -m ml.eval_fatigue` across exercises (keys are prefixed)."""
    from ml.eval_fatigue import run
    res = run({"trials": 40, "exercises": ["squat", "curl"], "seed": 11})
    assert res["synthetic"] is True
    assert res["false_alarm_rate"] is not None and res["false_alarm_rate"] <= 0.15
    # a sudden, large slowdown must be caught most of the time, and the onset located near rep 9
    assert res["detection_rate"]["sudden_after_rep_8"] >= 0.6
    assert res["detection_rate"]["sudden_after_rep_8"] > res["false_alarm_rate"]
    assert res["onset_error"]["squat/sudden_after_rep_8"]["mean_abs_reps"] <= 1.0
