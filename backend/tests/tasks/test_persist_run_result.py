"""Execution coverage for the real _persist_run_result.

Every test in test_execute_call.py monkeypatches this function away, so the
ModelResponse -> RunResult field mapping never actually runs there. These
tests call it for real against Postgres.
"""
import asyncio

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.base import ModelResponse
from app.db.models import EvalExample, Run, RunResult
from app.tasks.worker import _persist_run_result
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


@pytest.fixture
def run_and_example():
    """Create the FK parents, yield their ids, then clean everything up."""

    async def _setup():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(
                text="Profits rose sharply.", gold_label="positive", source="test"
            )
            session.add(example)
            run = Run(
                arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1
            )
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)
            return run.id, example.id

    async def _teardown(run_id: int, example_id: int):
        async with AsyncSession(db_test_engine) as session:
            await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            example = await session.get(EvalExample, example_id)
            if example:
                await session.delete(example)
            await session.commit()

    run_id, example_id = asyncio.run(_setup())
    yield run_id, example_id
    asyncio.run(_teardown(run_id, example_id))


def _fetch_rows(run_id: int) -> list[RunResult]:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            result = await session.execute(
                select(RunResult).where(RunResult.run_id == run_id).order_by(RunResult.id)
            )
            return result.scalars().all()

    return asyncio.run(_run())


def test_persists_success_response_field_by_field(run_and_example):
    run_id, example_id = run_and_example
    response = ModelResponse(
        text="positive",
        latency_ms=123.5,
        prompt_tokens=17,
        completion_tokens=3,
        cost_estimate_usd=0.000123,
        finish_reason="stop",
    )

    result_id = asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name="fake-arm",
            repeat_index=2,
            celery_task_id="task-abc",
            status="completed",
            response=response,
        )
    )

    rows = _fetch_rows(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert result_id == row.id
    assert row.run_id == run_id
    assert row.example_id == example_id
    assert row.arm_name == "fake-arm"
    assert row.repeat_index == 2
    assert row.celery_task_id == "task-abc"
    assert row.status == "completed"
    assert row.output_text == "positive"
    assert row.latency_ms == pytest.approx(123.5)
    assert row.prompt_tokens == 17
    assert row.completion_tokens == 3
    assert row.cost_estimate_usd == pytest.approx(0.000123)
    assert row.error_message is None
    assert row.judge_score is None
    assert row.created_at is not None


def test_persists_failure_with_all_response_fields_null(run_and_example):
    run_id, example_id = run_and_example

    asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name="fake-arm",
            repeat_index=0,
            celery_task_id=None,
            status="failed",
            response=None,
            error_message="boom: upstream returned 500",
        )
    )

    rows = _fetch_rows(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "failed"
    assert row.error_message == "boom: upstream returned 500"
    assert row.celery_task_id is None
    assert row.output_text is None
    assert row.latency_ms is None
    assert row.prompt_tokens is None
    assert row.completion_tokens is None
    assert row.cost_estimate_usd is None
