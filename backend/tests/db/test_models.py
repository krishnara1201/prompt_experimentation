import asyncio

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


def test_round_trip_insert_and_read():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)

            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            fetched = await session.get(RunResult, result.id)
            assert fetched is not None
            assert fetched.output_text == "positive"
            assert fetched.judge_score is None

            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def test_run_result_judge_columns_default_correctly():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)

            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            assert result.judge_status == "pending"
            assert result.judge_score is None
            assert result.judge_rationale is None
            assert result.judge_error_message is None
            assert result.judge_celery_task_id is None

            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def test_judge_calibration_label_round_trip():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            result = RunResult(
                run_id=run.id, example_id=example.id, arm_name="fake-arm",
                repeat_index=0, status="completed", judge_score=4, judge_status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            label = JudgeCalibrationLabel(
                run_result_id=result.id, human_score=4, labeled_by="you@example.com", notes="agrees"
            )
            session.add(label)
            await session.commit()
            await session.refresh(label)

            fetched = await session.get(JudgeCalibrationLabel, label.id)
            assert fetched is not None
            assert fetched.human_score == 4
            assert fetched.labeled_by == "you@example.com"
            assert fetched.labeled_at is not None

            await session.delete(label)
            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())
