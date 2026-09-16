from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query

from ml.features import FAULT_LABEL

from ..db import get_db
from ..security import current_user
from ..services.analytics import trend_per_week
from ..services.form_model import get_model

router = APIRouter(prefix="/api", tags=["stats"])


def _tz(tz: str) -> str:
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(400, "Unknown time zone")
    return tz


@router.get("/stats/overview")
async def overview(
    user: dict = Depends(current_user),
    weeks: int = Query(8, ge=1, le=52),
    tz: str = Query("UTC", max_length=64),
):
    tz = _tz(tz)
    since = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    match = {"$match": {"user_id": user["_id"], "status": "done", "started_at": {"$gte": since}}}
    pipeline = [
        match,
        {"$facet": {
            "weekly": [
                {"$group": {
                    "_id": {"$dateTrunc": {"date": "$started_at", "unit": "week", "startOfWeek": "monday", "timezone": tz}},
                    "reps": {"$sum": "$summary.total_reps"},
                    "sessions": {"$sum": 1},
                    "minutes": {"$sum": {"$divide": [{"$ifNull": ["$duration_s", 0]}, 60]}},
                }},
                {"$sort": {"_id": 1}},
            ],
            "daily": [
                {"$group": {
                    "_id": {"day": {"$dateTrunc": {"date": "$started_at", "unit": "day", "timezone": tz}},
                            "exercise": "$exercise"},
                    "avg_score": {"$avg": "$summary.avg_score"},
                    "avg_model_score": {"$avg": "$summary.avg_model_score"},
                    "reps": {"$sum": "$summary.total_reps"},
                }},
                {"$sort": {"_id.day": 1}},
            ],
            "faults": [
                {"$unwind": "$reps"},
                {"$unwind": "$reps.faults"},
                {"$group": {"_id": {"code": "$reps.faults", "exercise": "$exercise"}, "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 12},
            ],
            "bests": [
                {"$group": {
                    "_id": "$exercise",
                    "sessions": {"$sum": 1},
                    "total_reps": {"$sum": "$summary.total_reps"},
                    "max_session_reps": {"$max": "$summary.total_reps"},
                    "max_set_reps": {"$max": "$summary.best_set_reps"},
                    "best_avg_score": {"$max": {"$cond": [{"$gte": ["$summary.total_reps", 5]}, "$summary.avg_score", None]}},
                    "last": {"$max": "$started_at"},
                }},
                {"$sort": {"total_reps": -1}},
            ],
            "totals": [
                {"$group": {
                    "_id": None,
                    "sessions": {"$sum": 1},
                    "reps": {"$sum": "$summary.total_reps"},
                    "clean_reps": {"$sum": "$summary.clean_reps"},
                    "avg_score": {"$avg": "$summary.avg_score"},
                    "fatigue_sessions": {"$sum": {"$cond": ["$fatigue.detected", 1, 0]}},
                    "demo_sessions": {"$sum": {"$cond": [{"$eq": ["$demo", True]}, 1, 0]}},
                }},
            ],
        }},
    ]
    res = (await get_db().sessions.aggregate(pipeline).to_list(1))[0]

    # Least-squares trend of daily average rule score per exercise (tiny arrays -> numpy is fine)
    per_ex: dict[str, tuple[list, list]] = {}
    for d in res["daily"]:
        if d["avg_score"] is None:
            continue
        xs, ys = per_ex.setdefault(d["_id"]["exercise"], ([], []))
        xs.append(d["_id"]["day"].timestamp() / 86400)
        ys.append(d["avg_score"])
    trends = {ex: trend_per_week(xs, ys) for ex, (xs, ys) in per_ex.items()}

    totals = res["totals"][0] if res["totals"] else {"sessions": 0, "reps": 0, "clean_reps": 0, "avg_score": None, "fatigue_sessions": 0, "demo_sessions": 0}
    totals.pop("_id", None)
    return {
        "weeks": weeks,
        "totals": totals,
        "weekly": [{"week": w["_id"], "reps": w["reps"], "sessions": w["sessions"], "minutes": round(w["minutes"], 1)} for w in res["weekly"]],
        "daily": [{"day": d["_id"]["day"], "exercise": d["_id"]["exercise"], "avg_score": round(d["avg_score"] or 0, 1),
                   "avg_model_score": round(d["avg_model_score"], 1) if d["avg_model_score"] is not None else None,
                   "reps": d["reps"]} for d in res["daily"]],
        "faults": [{"code": f["_id"]["code"], "exercise": f["_id"]["exercise"], "label": FAULT_LABEL.get(f["_id"]["code"], f["_id"]["code"]),
                    "count": f["count"]} for f in res["faults"]],
        "bests": [{"exercise": b["_id"], **{k: v for k, v in b.items() if k != "_id"},
                   "trend_per_week": trends.get(b["_id"])} for b in res["bests"]],
    }


@router.get("/model")
async def model_info(_: dict = Depends(current_user)):
    return get_model().info()
