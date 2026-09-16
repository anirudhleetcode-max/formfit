from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from starlette.concurrency import run_in_threadpool

from ..db import get_db
from ..schemas import Exercise, SessionCreate, SessionFinish
from ..security import current_user
from ..services.form_model import get_model
from ..services.sessions import build_finished

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(404, "Session not found")


def serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    doc.pop("user_id", None)
    return doc


def _encode_cursor(doc: dict) -> str:
    return f"{int(doc['started_at'].timestamp() * 1000)}.{doc['_id']}"


def _decode_cursor(cursor: str) -> tuple[datetime, ObjectId]:
    try:
        ms, oid = cursor.split(".", 1)
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc), ObjectId(oid)
    except Exception:
        raise HTTPException(400, "Invalid cursor")


@router.post("", status_code=201)
async def create_session(body: SessionCreate, user: dict = Depends(current_user)):
    doc = {
        "user_id": user["_id"],
        "exercise": body.exercise,
        "source": body.source,
        "status": "active",
        "started_at": datetime.now(timezone.utc),
    }
    res = await get_db().sessions.insert_one(doc)
    doc["_id"] = res.inserted_id
    return serialize(doc)


@router.post("/{session_id}/finish")
async def finish_session(session_id: str, body: SessionFinish, user: dict = Depends(current_user)):
    db = get_db()
    sess = await db.sessions.find_one({"_id": _oid(session_id), "user_id": user["_id"]})
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess["status"] == "done":
        raise HTTPException(409, "Session already finished")
    reps = [r.model_dump() for r in body.reps]
    model_scores = await run_in_threadpool(get_model().score, sess["exercise"], [r["features"] for r in reps])
    built = build_finished(reps, model_scores)
    update = {**built, "status": "done", "finished_at": datetime.now(timezone.utc),
              "duration_s": round(body.duration_s, 1)}
    doc = await db.sessions.find_one_and_update(
        {"_id": sess["_id"], "user_id": user["_id"], "status": "active"},
        {"$set": update}, return_document=True,
    )
    if not doc:
        raise HTTPException(409, "Session already finished")
    return serialize(doc)


@router.get("")
async def list_sessions(
    user: dict = Depends(current_user),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = None,
    exercise: Exercise | None = None,
):
    q: dict = {"user_id": user["_id"], "status": "done"}
    if exercise:
        q["exercise"] = exercise
    if cursor:
        ts, oid = _decode_cursor(cursor)
        q["$or"] = [{"started_at": {"$lt": ts}}, {"started_at": ts, "_id": {"$lt": oid}}]
    projection = {"reps": 0, "fatigue.index": 0}
    docs = await get_db().sessions.find(q, projection).sort([("started_at", -1), ("_id", -1)]).limit(limit + 1).to_list(limit + 1)
    more = len(docs) > limit
    docs = docs[:limit]
    return {
        "items": [serialize(d) for d in docs],
        "next_cursor": _encode_cursor(docs[-1]) if more else None,
    }


@router.get("/{session_id}")
async def get_session(session_id: str, user: dict = Depends(current_user)):
    doc = await get_db().sessions.find_one({"_id": _oid(session_id), "user_id": user["_id"]})
    if not doc:
        raise HTTPException(404, "Session not found")
    return serialize(doc)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, user: dict = Depends(current_user)):
    res = await get_db().sessions.delete_one({"_id": _oid(session_id), "user_id": user["_id"]})
    if not res.deleted_count:
        raise HTTPException(404, "Session not found")
    return Response(status_code=204)
