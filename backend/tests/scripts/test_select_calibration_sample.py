from scripts.select_calibration_sample import stratified_sample


def _row(run_result_id, arm_name, gold_label):
    return {
        "run_result_id": run_result_id,
        "arm_name": arm_name,
        "gold_label": gold_label,
        "input_text": f"text {run_result_id}",
        "model_output": f"output {run_result_id}",
        "judge_score": 4,
        "judge_rationale": "ok",
        "human_score": None,
    }


def test_returns_everything_when_n_exceeds_available():
    rows = [_row(1, "arm-a", "positive"), _row(2, "arm-b", "negative")]
    sample = stratified_sample(rows, n=10)
    assert [r["run_result_id"] for r in sample] == [1, 2]


def test_samples_across_all_strata():
    rows = (
        [_row(i, "arm-a", "positive") for i in range(1, 11)]
        + [_row(i, "arm-a", "negative") for i in range(11, 21)]
        + [_row(i, "arm-b", "positive") for i in range(21, 31)]
        + [_row(i, "arm-b", "negative") for i in range(31, 41)]
    )
    sample = stratified_sample(rows, n=8, seed=42)

    assert len(sample) == 8
    strata = {(r["arm_name"], r["gold_label"]) for r in sample}
    assert strata == {("arm-a", "positive"), ("arm-a", "negative"), ("arm-b", "positive"), ("arm-b", "negative")}


def test_is_deterministic_given_a_seed():
    rows = [_row(i, "arm-a", "positive") for i in range(1, 21)]
    first = stratified_sample(rows, n=5, seed=7)
    second = stratified_sample(rows, n=5, seed=7)
    assert [r["run_result_id"] for r in first] == [r["run_result_id"] for r in second]


import asyncio

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from scripts.select_calibration_sample import _fetch_judged_rows
from tests.conftest import db_test_engine, postgres_reachable


@pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)")
def test_fetch_judged_rows_only_includes_judge_completed(monkeypatch):
    monkeypatch.setattr("scripts.select_calibration_sample.engine", db_test_engine)

    async def _setup():
        # expire_on_commit=False: this coroutine returns ids read off ORM
        # objects after a commit; the default would lazy-reload them as sync
        # IO outside the greenlet and raise MissingGreenlet.
        async with AsyncSession(db_test_engine, expire_on_commit=False) as session:
            example = EvalExample(text="Profits rose.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=2)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            judged = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
                judge_status="completed",
                judge_score=5,
                judge_rationale="Good.",
            )
            pending = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=1,
                output_text="positive",
                status="completed",
                judge_status="pending",
            )
            session.add(judged)
            session.add(pending)
            await session.commit()
            await session.refresh(judged)
            await session.refresh(pending)
            return run.id, example.id, judged.id, pending.id

    run_id, example_id, judged_id, pending_id = asyncio.run(_setup())

    try:
        rows = asyncio.run(_fetch_judged_rows(run_id))
        assert [r["run_result_id"] for r in rows] == [judged_id]
        assert rows[0]["gold_label"] == "positive"
        assert rows[0]["judge_score"] == 5
    finally:
        async def _teardown():
            async with AsyncSession(db_test_engine) as session:
                for rid in (judged_id, pending_id):
                    obj = await session.get(RunResult, rid)
                    if obj:
                        await session.delete(obj)
                run = await session.get(Run, run_id)
                if run:
                    await session.delete(run)
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
                await session.commit()

        asyncio.run(_teardown())
