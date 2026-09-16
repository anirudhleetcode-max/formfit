"""Create the demo account with ~30 days of realistic training history.

    python -m scripts.seed

Reps come from the same biomechanical simulator the classifier is trained on (ml/simulate.py),
with fatigue rising through each set and form slowly improving over the month. Rule scores use
the Python mirror of the browser rules; model scores come from the trained classifier.
Re-running replaces the demo user's sessions only.
"""
import asyncio
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from app.db import close_client, ensure_indexes, get_db
from app.security import hash_password
from app.services.form_model import get_model
from app.services.sessions import build_finished
from ml import rules
from ml.simulate import simulate_rep

EMAIL, PASSWORD, NAME = "demo@formfit.app", "demo1234", "Demo Athlete"
PLAN = ["squat", "pushup", "curl", "squat", "press", "pushup", "lunge", "squat", "curl"]


def make_session(exercise: str, started: datetime, progress: float, rng: np.random.Generator):
    n_sets = int(rng.integers(2, 5))
    base_reps = {"squat": 10, "pushup": 12, "curl": 10, "press": 8, "lunge": 8}[exercise]
    fault_rate = 0.20 - 0.12 * progress            # form improves over the month
    view = "side" if rng.random() < 0.8 else "oblique"
    reps, t = [], 20.0
    for s in range(1, n_sets + 1):
        n = max(4, int(base_reps + rng.integers(-2, 4) + 3 * progress - s))
        for i in range(n):
            # fatigue climbs within a set; later sets start more tired
            fatigue = min(1.0, (i / n) ** 2 * 0.9 + 0.08 * (s - 1))
            feats, _, _ = simulate_rep(exercise, rng, fatigue=fatigue, fault_rate=fault_rate, view=view)
            score, faults = rules.evaluate(exercise, feats)
            reps.append({
                "set": s, "score": score, "faults": faults, "ecc_s": round(feats["ecc_s"], 2),
                "con_s": round(feats["con_s"], 2), "rom": round(feats["rom"], 1), "t": round(t, 1),
                "features": feats,
            })
            t += feats["ecc_s"] + feats["con_s"] + 0.6
        t += float(rng.uniform(60, 120))  # rest between sets
    scores = get_model().score(exercise, [r["features"] for r in reps])
    built = build_finished(reps, scores)
    return {
        "exercise": exercise, "source": "live" if rng.random() < 0.85 else "upload", "status": "done",
        "started_at": started, "finished_at": started + timedelta(seconds=t), "duration_s": round(t, 1),
        **built,
    }


async def main():
    await ensure_indexes()
    db = get_db()
    user = await db.users.find_one({"email": EMAIL})
    if not user:
        res = await db.users.insert_one({"name": NAME, "email": EMAIL, "password_hash": hash_password(PASSWORD),
                                         "created_at": datetime.now(timezone.utc)})
        uid = res.inserted_id
    else:
        uid = user["_id"]
    await db.sessions.delete_many({"user_id": uid})

    rng = np.random.default_rng(42)
    rnd = random.Random(42)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    docs, k = [], 0
    for day in range(30, 0, -1):
        if rnd.random() > 0.62:        # ~4 training days a week
            continue
        for _ in range(1 if rnd.random() < 0.7 else 2):
            ex = PLAN[k % len(PLAN)]
            k += 1
            started = (now - timedelta(days=day)).replace(hour=rnd.choice([7, 8, 18, 19]), minute=rnd.randint(0, 59))
            doc = make_session(ex, started + timedelta(minutes=30 * len(docs) % 90), (30 - day) / 30, rng)
            doc["user_id"] = uid
            docs.append(doc)
    await db.sessions.insert_many(docs)
    reps = sum(d["summary"]["total_reps"] for d in docs)
    print(f"seeded {len(docs)} sessions / {reps} reps for {EMAIL} (password {PASSWORD})")
    await close_client()


if __name__ == "__main__":
    asyncio.run(main())
