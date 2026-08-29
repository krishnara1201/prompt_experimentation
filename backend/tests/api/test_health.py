from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.main import app
from tests.conftest import db_test_engine, postgres_reachable


async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(db_test_engine) as session:
        yield session


@pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)")
def test_health_ok_when_db_reachable():
    app.dependency_overrides[get_session] = _override_get_session
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_503_when_db_query_fails():
    class _BrokenSession:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    async def _override_broken() -> AsyncGenerator[object, None]:
        yield _BrokenSession()

    app.dependency_overrides[get_session] = _override_broken
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unreachable"}
