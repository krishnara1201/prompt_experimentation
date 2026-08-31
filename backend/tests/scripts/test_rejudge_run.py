import asyncio

import pytest
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

TEST_SOURCE = "rejudge_test"


async def _seed_run(task: str) -> tuple[int, int]:
    async with AsyncSession(db_test_engine, expire_on_commit=False) as session:
        example = EvalExample(text="x", gold_label="World", source=TEST_SOURCE)
        run = Run(arm_names=["a"], repeats=1, total_calls=2, task=task)
        session.add(example)
        session.add(run)
        await session.commit()
        example_id, run_id = example.id, run.id

        for i in range(2):
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name="a",
                    repeat_index=i,
                    status="completed",
                    output_text="World",
                    judge_status="completed",
                    judge_score=3,
                )
            )
        await session.commit()
        return run_id, example_id


async def _cleanup(run_id: int, example_id: int) -> None:
    async with AsyncSession(db_test_engine) as session:
        await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
        await session.execute(delete(Run).where(Run.id == run_id))
        await session.execute(delete(EvalExample).where(EvalExample.source == TEST_SOURCE))
        await session.commit()


def test_rejudge_threads_the_runs_task_name(monkeypatch):
    monkeypatch.setattr("scripts.rejudge_run.engine", db_test_engine)
    calls: list[dict] = []
    monkeypatch.setattr(
        "scripts.rejudge_run.run_judge_call.apply_async",
        lambda **kw: calls.append(kw),
    )

    run_id, example_id = asyncio.run(_seed_run("ag_news"))
    try:
        from scripts.rejudge_run import main

        main(run_id)

        assert len(calls) == 2
        for call in calls:
            assert call["kwargs"]["task_name"] == "ag_news"
    finally:
        asyncio.run(_cleanup(run_id, example_id))
