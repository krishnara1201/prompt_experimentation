import asyncio

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
from scripts.import_calibration_labels import CalibrationImportError, import_labels
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


@pytest.fixture
def run_result_id():
    async def _setup():
        # expire_on_commit=False: this coroutine returns ids read off ORM
        # objects after a commit; the default would lazy-reload them as sync
        # IO outside the greenlet and raise MissingGreenlet.
        async with AsyncSession(db_test_engine, expire_on_commit=False) as session:
            example = EvalExample(text="Profits rose.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
                judge_score=5,
                judge_status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)
            return result.id, run.id, example.id

    result_id, run_id, example_id = asyncio.run(_setup())
    yield result_id

    async def _teardown():
        async with AsyncSession(db_test_engine) as session:
            await session.execute(delete(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == result_id))
            rr = await session.get(RunResult, result_id)
            if rr:
                await session.delete(rr)
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            example = await session.get(EvalExample, example_id)
            if example:
                await session.delete(example)
            await session.commit()

    asyncio.run(_teardown())


def test_import_is_idempotent_upsert(monkeypatch, run_result_id):
    monkeypatch.setattr("scripts.import_calibration_labels.engine", db_test_engine)
    rows = [{"run_result_id": run_result_id, "human_score": 4, "notes": "seems right"}]

    first_count = asyncio.run(import_labels(rows, "you@example.com"))
    rows[0]["human_score"] = 5
    second_count = asyncio.run(import_labels(rows, "you@example.com"))

    assert first_count == 1
    assert second_count == 1

    async def _fetch():
        async with AsyncSession(db_test_engine) as session:
            result = await session.execute(
                select(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == run_result_id)
            )
            return result.scalars().all()

    labels = asyncio.run(_fetch())
    assert len(labels) == 1
    assert labels[0].human_score == 5


def test_rejects_missing_human_score(run_result_id):
    rows = [{"run_result_id": run_result_id, "human_score": None}]
    with pytest.raises(CalibrationImportError):
        asyncio.run(import_labels(rows, "you@example.com"))


def test_rejects_out_of_range_human_score(run_result_id):
    rows = [{"run_result_id": run_result_id, "human_score": 7}]
    with pytest.raises(CalibrationImportError):
        asyncio.run(import_labels(rows, "you@example.com"))
