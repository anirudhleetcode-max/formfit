import logging
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

log = logging.getLogger("formfit")
MAX_BODY = 1_000_000  # 1 MB: a 500-rep session is ~250 KB of JSON


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    await run_in_threadpool(get_model)  # load the classifiers once
    yield
    await close_client()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY:
        return JSONResponse({"detail": "Request body too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Something went wrong on our side"}, status_code=500)


app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(stats.router)


@app.get("/api/health")
async def health():
    await get_db().command("ping")
    return {"status": "ok", "db": "connected", "model": get_model().ready}
