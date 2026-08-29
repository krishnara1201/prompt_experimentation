import asyncio

import pytest
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
from scripts.calibration_report import build_report
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


def test_build_report_joins_judge_and_human_scores(monkeypatch):
    monkeypatch.setattr("scripts.calibration_report.engine", db_test_engine)

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
                judge_status="completed",
                judge_score=4,
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            label = JudgeCalibrationLabel(run_result_id=result.id, human_score=4, labeled_by="you@example.com")
            session.add(label)
            await session.commit()
            return run.id, example.id, result.id

    run_id, example_id, result_id = asyncio.run(_setup())

    try:
        report = asyncio.run(build_report(run_id))
        assert report["n"] == 1
        assert report["mean_abs_diff"] == pytest.approx(0.0)
    finally:
        async def _teardown():
            async with AsyncSession(db_test_engine) as session:
                await session.execute(
                    delete(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == result_id)
                )
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
