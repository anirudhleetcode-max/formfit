import uuid

import numpy as np

from ml.simulate import simulate_rep


def _reps(n=10, sets=2, fatigue_late=False, exercise="squat"):
    rng = np.random.default_rng(0)
    out, t = [], 0.0
    for s in range(1, sets + 1):
        for i in range(n):
            fat = 0.95 if (fatigue_late and i >= n // 2) else 0.0
            f, good, faults = simulate_rep(exercise, rng, fatigue=fat, fault_rate=0.05, view="side")
            out.append({"set": s, "score": 100 if good else 70, "faults": [] if good else faults[:2],
                        "ecc_s": f["ecc_s"], "con_s": f["con_s"], "rom": f["rom"], "t": t, "features": f})
            t += 3
    return out


def _new_user(client):
    r = client.post("/api/auth/register", json={"name": "U", "email": f"u{uuid.uuid4().hex[:8]}@example.com", "password": "secret123"})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_session_lifecycle(client, auth_headers):
    r = client.post("/api/sessions", json={"exercise": "squat"}, headers=auth_headers)
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["status"] == "active"

    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": _reps(), "duration_s": 95.5}, headers=auth_headers)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["status"] == "done"
    assert doc["summary"]["total_reps"] == 20 and doc["summary"]["sets"] == 2
    assert [s["reps"] for s in doc["sets"]] == [10, 10]
    assert all(0 <= rep["model_score"] <= 100 for rep in doc["reps"])
    assert [rep["i"] for rep in doc["reps"]] == list(range(1, 21))
    assert "detected" in doc["fatigue"] and len(doc["fatigue"]["index"]) == 20

    # finishing twice is a conflict
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": [], "duration_s": 1}, headers=auth_headers)
    assert r.status_code == 409

    detail = client.get(f"/api/sessions/{sid}", headers=auth_headers).json()
    assert detail["id"] == sid and len(detail["reps"]) == 20


def test_validation_and_errors(client, auth_headers):
    assert client.post("/api/sessions", json={"exercise": "deadlift"}, headers=auth_headers).status_code == 422
    sid = client.post("/api/sessions", json={"exercise": "curl"}, headers=auth_headers).json()["id"]
    bad = _reps(2, 1, exercise="curl")
    bad[0]["faults"] = ["made_up_fault"]
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": bad, "duration_s": 5}, headers=auth_headers)
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)
    bad = _reps(2, 1, exercise="curl")
    bad[0]["score"] = 140
    assert client.post(f"/api/sessions/{sid}/finish", json={"reps": bad, "duration_s": 5}, headers=auth_headers).status_code == 422
    assert client.get("/api/sessions/not-an-id", headers=auth_headers).status_code == 404
    assert client.get("/api/sessions?cursor=garbage", headers=auth_headers).status_code == 400
    assert client.get("/api/sessions").status_code == 401
    # unknown feature keys are silently dropped
    ok = _reps(2, 1, exercise="curl")
    ok[0]["features"]["evil"] = 1.0
    r = client.post(f"/api/sessions/{sid}/finish", json={"reps": ok, "duration_s": 5}, headers=auth_headers)
    assert r.status_code == 200 and "evil" not in r.json()["reps"][0]["features"]


def test_body_limit(client, auth_headers):
    r = client.post("/api/sessions", content=b"x" * 1_100_000, headers={**auth_headers, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_users_are_isolated(client, auth_headers):
    sid = client.post("/api/sessions", json={"exercise": "press"}, headers=auth_headers).json()["id"]
    other = _new_user(client)
    assert client.get(f"/api/sessions/{sid}", headers=other).status_code == 404
    assert client.post(f"/api/sessions/{sid}/finish", json={"reps": [], "duration_s": 1}, headers=other).status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=other).status_code == 404
    assert client.get("/api/sessions", headers=other).json()["items"] == []


def test_pagination_and_filter(client):
    h = _new_user(client)
    ids = []
    for ex in ["squat", "curl", "squat", "pushup", "squat"]:
        sid = client.post("/api/sessions", json={"exercise": ex}, headers=h).json()["id"]
        client.post(f"/api/sessions/{sid}/finish", json={"reps": _reps(3, 1, exercise=ex), "duration_s": 10}, headers=h)
        ids.append(sid)
    # an unfinished session is not listed
    client.post("/api/sessions", json={"exercise": "squat"}, headers=h)

    seen, cursor = [], None
    while True:
        url = "/api/sessions?limit=2" + (f"&cursor={cursor}" if cursor else "")
        page = client.get(url, headers=h).json()
        assert len(page["items"]) <= 2
        assert all("reps" not in it for it in page["items"])
        seen += [it["id"] for it in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert seen == ids[::-1]
    squats = client.get("/api/sessions?exercise=squat", headers=h).json()["items"]
    assert len(squats) == 3

    assert client.delete(f"/api/sessions/{ids[0]}", headers=h).status_code == 204
    assert len(client.get("/api/sessions", headers=h).json()["items"]) == 4


def test_stats_overview(client):
    h = _new_user(client)
    empty = client.get("/api/stats/overview", headers=h).json()
    assert empty["totals"]["sessions"] == 0 and empty["weekly"] == []
    for ex in ["squat", "squat", "curl"]:
        sid = client.post("/api/sessions", json={"exercise": ex}, headers=h).json()["id"]
        client.post(f"/api/sessions/{sid}/finish", json={"reps": _reps(6, 2, exercise=ex), "duration_s": 60}, headers=h)
    s = client.get("/api/stats/overview?weeks=4&tz=Asia/Kolkata", headers=h).json()
    assert s["totals"]["sessions"] == 3 and s["totals"]["reps"] == 36
    assert sum(w["reps"] for w in s["weekly"]) == 36
    bests = {b["exercise"]: b for b in s["bests"]}
    assert bests["squat"]["total_reps"] == 24 and bests["squat"]["max_set_reps"] == 6
    assert all(f["count"] > 0 and f["label"] for f in s["faults"])
    assert client.get("/api/stats/overview?tz=Mars/Base", headers=h).status_code == 400


def test_model_info(client, auth_headers):
    info = client.get("/api/model", headers=auth_headers).json()
    assert info["ready"] is True
    assert set(info["metrics"]["exercises"]) == {"squat", "pushup", "curl", "press", "lunge"}
    assert info["synthetic"] is True and info["card"]["dataset"]["synthetic"] is True
    # key is always present (None until experiments/real_clips has been run)
    assert "real_clip_eval" in info
    if info["real_clip_eval"]:
        assert info["real_clip_eval"]["dataset"]["synthetic"] is False
