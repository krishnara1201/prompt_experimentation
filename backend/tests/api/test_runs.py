import asyncio
import socket
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run
from app.db.session import DATABASE_URL, engine
from app.main import app


def _postgres_reachable() -> bool:
    if not DATABASE_URL:
        return False
    parts = urlsplit(DATABASE_URL.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

FAKE_ARMS = {"fake-arm": object()}


def _insert_example() -> int:
    async def _run():
        async with AsyncSession(engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)
            return example.id

    return asyncio.run(_run())


def _delete_example(example_id: int) -> None:
    async def _run():
        async with AsyncSession(engine) as session:
            obj = await session.get(EvalExample, example_id)
            if obj:
                await session.delete(obj)
                await session.commit()

    asyncio.run(_run())


def _delete_run(run_id: int) -> None:
    async def _run():
        async with AsyncSession(engine) as session:
            obj = await session.get(Run, run_id)
            if obj:
                await session.delete(obj)
                await session.commit()

    asyncio.run(_run())


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_enqueues_expected_number_of_calls(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        response = TestClient(app).post("/runs", json={"repeats": 2, "sample_size": 1, "seed": 1})
        assert response.status_code == 200
        body = response.json()
        run_id = body["run_id"]
        assert body["status"] == "pending"
        assert body["total_calls"] == 2  # 1 example x 1 arm x 2 repeats
        assert mock_task.delay.call_count == 2
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_rejects_unknown_arm(mock_load_arms, mock_task):
    response = TestClient(app).post("/runs", json={"arms": ["nonexistent-arm"]})
    assert response.status_code == 400


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_get_run_status_is_pending_before_any_results(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        create_response = TestClient(app).post("/runs", json={"repeats": 1, "sample_size": 1, "seed": 1})
        run_id = create_response.json()["run_id"]

        status_response = TestClient(app).get(f"/runs/{run_id}")
        assert status_response.status_code == 200
        body = status_response.json()
        assert body["status"] == "pending"
        assert body["total_calls"] == 1
        assert body["completed"] == 0
        assert body["failed"] == 0
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_status_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999")
    assert response.status_code == 404
