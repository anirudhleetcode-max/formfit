from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(get_settings().mongo_uri, serverSelectionTimeoutMS=4000, tz_aware=True)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().mongo_db]


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def ensure_indexes() -> None:
    db = get_db()
    await db.users.create_index("email", unique=True)
    # list / cursor pagination and the stats $match all start with user_id + status + started_at
    await db.sessions.create_index([("user_id", 1), ("status", 1), ("started_at", -1), ("_id", -1)])
    await db.sessions.create_index([("user_id", 1), ("exercise", 1), ("started_at", -1)])
