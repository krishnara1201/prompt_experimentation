from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.main import app
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running")


async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
    # TestClient runs the app on its own portal loop; the production pooled
    # engine would leak asyncpg connections across loops. Use NullPool.
    async with AsyncSession(db_test_engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _use_test_engine_session():
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.pop(get_session, None)


def test_get_tasks_lists_financial_sentiment_active_by_default():
    resp = TestClient(app).get("/tasks")
    assert resp.status_code == 200
    by_name = {t["name"]: t for t in resp.json()}
    assert "financial_sentiment" in by_name
    fs = by_name["financial_sentiment"]
    assert fs["active"] is True
    assert set(fs["labels"]) == {"positive", "negative", "neutral"}
    assert fs["seeded_count"] >= 0


def test_get_tasks_skips_a_broken_pack_instead_of_500ing(monkeypatch):
    import app.api.routes.tasks as tasks_route

    monkeypatch.setattr(
        tasks_route, "list_tasks", lambda: ["financial_sentiment", "does_not_exist"]
    )
    resp = TestClient(app).get("/tasks")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()]
    assert names == ["financial_sentiment"]


def test_get_tasks_reports_multiple_packs_with_one_active():
    rows = TestClient(app).get("/tasks").json()
    # More than just the default pack is configured (ag_news ships too).
    assert len(rows) >= 2
    assert [r["name"] for r in rows if r["active"]] == ["financial_sentiment"]
    for r in rows:
        assert set(r) == {"name", "description", "labels", "active", "seeded_count"}
