import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.main import app
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

FAKE_ARMS = {"fake-arm": object()}


async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
    """Make the TestClient-driven app use the NullPool test engine.

    TestClient runs the app on its own portal event loop, separate from the
    asyncio.run() loops used by the helpers below. The production pooled
    engine would leak asyncpg connections across those loops.
    """
    async with AsyncSession(db_test_engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _use_test_engine_session():
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.pop(get_session, None)


def _insert_example() -> int:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)
            return example.id

    return asyncio.run(_run())


def _delete_example(example_id: int) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            obj = await session.get(EvalExample, example_id)
            if obj:
                await session.delete(obj)
                await session.commit()

    asyncio.run(_run())


def _delete_run(run_id: int) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
            obj = await session.get(Run, run_id)
            if obj:
                await session.delete(obj)
            await session.commit()

    asyncio.run(_run())


def _latest_run_id() -> int | None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            result = await session.execute(select(Run.id).order_by(Run.id.desc()).limit(1))
            return result.scalars().first()

    return asyncio.run(_run())


def _insert_run(total_calls: int) -> int:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(
                arm_names=["fake-arm"],
                sample_size=None,
                repeats=1,
                seed=None,
                total_calls=total_calls,
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    return asyncio.run(_run())


def _insert_results(run_id: int, example_id: int, statuses: list[str]) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            for index, status in enumerate(statuses):
                session.add(
                    RunResult(
                        run_id=run_id,
                        example_id=example_id,
                        arm_name="fake-arm",
                        repeat_index=index,
                        status=status,
                        output_text="positive" if status == "completed" else None,
                        error_message=None if status == "completed" else "boom",
                    )
                )
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
def test_create_run_rejects_empty_arms_list(mock_load_arms, mock_task):
    # An explicit [] would create a run with total_calls == 0, which can
    # never resolve past "pending". Reject it at creation time.
    response = TestClient(app).post("/runs", json={"arms": []})
    assert response.status_code == 422


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_deletes_run_row_when_enqueue_fails(mock_load_arms, mock_task):
    mock_task.delay.side_effect = RuntimeError("redis is down")
    example_id = _insert_example()
    run_id_before = _latest_run_id()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/runs", json={"repeats": 1, "sample_size": 1, "seed": 1})
        assert response.status_code >= 500

        # No orphaned Run row survives the failed enqueue.
        assert _latest_run_id() == run_id_before
    finally:
        _delete_example(example_id)


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


def test_get_run_status_is_running_when_partially_done():
    example_id = _insert_example()
    run_id = _insert_run(total_calls=3)
    try:
        _insert_results(run_id, example_id, ["completed"])

        body = TestClient(app).get(f"/runs/{run_id}").json()
        assert body["status"] == "running"
        assert body["total_calls"] == 3
        assert body["completed"] == 1
        assert body["failed"] == 0
        assert body["pending"] == 2
    finally:
        _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_status_is_completed_when_all_succeed():
    example_id = _insert_example()
    run_id = _insert_run(total_calls=2)
    try:
        _insert_results(run_id, example_id, ["completed", "completed"])

        body = TestClient(app).get(f"/runs/{run_id}").json()
        assert body["status"] == "completed"
        assert body["completed"] == 2
        assert body["failed"] == 0
        assert body["pending"] == 0
    finally:
        _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_status_is_completed_with_errors_when_some_fail():
    example_id = _insert_example()
    run_id = _insert_run(total_calls=3)
    try:
        _insert_results(run_id, example_id, ["completed", "failed", "completed"])

        body = TestClient(app).get(f"/runs/{run_id}").json()
        assert body["status"] == "completed_with_errors"
        assert body["completed"] == 2
        assert body["failed"] == 1
        assert body["pending"] == 0
    finally:
        _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_results_returns_rows_ordered_by_id():
    example_id = _insert_example()
    run_id = _insert_run(total_calls=3)
    try:
        _insert_results(run_id, example_id, ["completed", "failed", "completed"])

        rows = TestClient(app).get(f"/runs/{run_id}/results").json()
        assert len(rows) == 3
        assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)

        page = TestClient(app).get(f"/runs/{run_id}/results?limit=2&offset=0").json()
        next_page = TestClient(app).get(f"/runs/{run_id}/results?limit=2&offset=2").json()
        assert len(page) == 2
        assert len(next_page) == 1
        # Pages must not overlap — only guaranteed with a deterministic ORDER BY.
        assert {row["id"] for row in page}.isdisjoint({row["id"] for row in next_page})
    finally:
        _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_status_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999")
    assert response.status_code == 404
