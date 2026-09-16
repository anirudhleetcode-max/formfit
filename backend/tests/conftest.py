import os
import uuid

os.environ["MONGO_DB"] = f"test_{uuid.uuid4().hex[:8]}"

import pytest
from fastapi.testclient import TestClient

from app.db import get_client
from app.config import get_settings
from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
    import pymongo
    pymongo.MongoClient(get_settings().mongo_uri).drop_database(os.environ["MONGO_DB"])


@pytest.fixture(scope="session")
def auth_headers(client):
    r = client.post("/api/auth/register", json={"name": "Test", "email": f"t{uuid.uuid4().hex[:6]}@example.com", "password": "secret123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
