import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import auth
from .config import get_settings
from .db import close_client, ensure_indexes, get_db
from .routers import sessions, stats
from .services.form_model import get_model

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("formfit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    model = await run_in_threadpool(get_model)  # load the classifiers once
    if not model.ready:
        log.warning("starting in DEGRADED mode: %s", model.error)
    yield
    await close_client()


class BodyLimit:
    """Rejects request bodies above `limit` bytes, including chunked bodies without Content-Length.
    The body (at most `limit` bytes) is buffered and replayed to the app."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        too_large = JSONResponse({"detail": "Request body too large"}, status_code=413)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length and length.isdigit() and int(length) > self.limit:
            return await too_large(scope, receive, send)
        chunks, size = [], 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            size += len(chunks[-1])
            if size > self.limit:
                return await too_large(scope, receive, send)
            if not msg.get("more_body"):
                break
        body, sent = b"".join(chunks), False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app = FastAPI(title=settings.app_name, lifespan=lifespan,
              description="FormFit API: sessions, per-rep metrics, fatigue analytics and the form classifier "
                          "(trained on synthetic data). Not a medical device.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(BodyLimit, limit=settings.max_body_bytes)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    # method, path and status only: never bodies, tokens or query strings
    log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Something went wrong on our side"}, status_code=500)


app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(stats.router)


@app.get("/api/health")
async def health():
    m = get_model()
    try:
        await asyncio.wait_for(get_db().command("ping"), timeout=3)
        db_state = "connected"
    except Exception as e:  # noqa: BLE001 - report, don't crash
        log.warning("health: database unreachable (%s)", type(e).__name__)
        db_state = "unreachable"
    body = {
        "status": "ok" if m.ready and db_state == "connected" else "degraded",
        "db": db_state,
        "model": m.ready,
        "model_version": m.bundle.get("version") if m.ready else None,
        "model_error": m.error,
    }
    return body if db_state == "connected" else JSONResponse(body, status_code=503)
