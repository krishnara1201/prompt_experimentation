import asyncio

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.stats.aggregation import load_metric_by_example, summarize_arms
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


def _insert_examples(n: int) -> list[int]:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            examples = [EvalExample(text=f"text {i}", gold_label="positive", source="test") for i in range(n)]
            session.add_all(examples)
            await session.commit()
            for example in examples:
                await session.refresh(example)
            return [e.id for e in examples]

    return asyncio.run(_run())


def _insert_run(arm_names: list[str]) -> int:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=arm_names, sample_size=None, repeats=1, seed=None, total_calls=0)
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    return asyncio.run(_run())


def _insert_result(
    run_id: int,
    example_id: int,
    arm_name: str,
    repeat_index: int,
    status: str = "completed",
    judge_status: str = "completed",
    judge_score: float | None = None,
    latency_ms: float | None = None,
) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    status=status,
                    judge_status=judge_status,
                    judge_score=judge_score,
                    latency_ms=latency_ms,
                    output_text="x" if status == "completed" else None,
                    error_message=None if status == "completed" else "boom",
                )
            )
            await session.commit()

    asyncio.run(_run())


def _cleanup(run_id: int, example_ids: list[int]) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            from sqlmodel import delete

            await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            for example_id in example_ids:
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def test_groups_completed_results_by_example_and_arm():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=100.0)
        _insert_result(run_id, example_ids[0], "arm-a", 1, latency_ms=120.0)
        _insert_result(run_id, example_ids[0], "arm-b", 0, latency_ms=200.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a", "arm-b"])

        result = asyncio.run(_query())
        assert result[(example_ids[0], "arm-a")] == [100.0, 120.0]
        assert result[(example_ids[0], "arm-b")] == [200.0]
    finally:
        _cleanup(run_id, example_ids)


def test_excludes_non_completed_status_rows():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, status="failed", latency_ms=999.0)
        _insert_result(run_id, example_ids[0], "arm-a", 1, status="completed", latency_ms=50.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        result = asyncio.run(_query())
        assert result[(example_ids[0], "arm-a")] == [50.0]
    finally:
        _cleanup(run_id, example_ids)


def test_judge_score_metric_requires_completed_judge_status():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(
            run_id, example_ids[0], "arm-a", 0,
            status="completed", judge_status="pending", judge_score=None, latency_ms=10.0,
        )
        _insert_result(
            run_id, example_ids[0], "arm-a", 1,
            status="completed", judge_status="completed", judge_score=4.0, latency_ms=20.0,
        )

        async def _query_judge():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "judge_score", ["arm-a"])

        async def _query_latency():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        judge_result = asyncio.run(_query_judge())
        latency_result = asyncio.run(_query_latency())
        assert judge_result[(example_ids[0], "arm-a")] == [4.0]
        assert latency_result[(example_ids[0], "arm-a")] == [10.0, 20.0]
    finally:
        _cleanup(run_id, example_ids)


def test_only_includes_requested_arms():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=1.0)
        _insert_result(run_id, example_ids[0], "arm-b", 0, latency_ms=2.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        result = asyncio.run(_query())
        assert list(result.keys()) == [(example_ids[0], "arm-a")]
    finally:
        _cleanup(run_id, example_ids)


def test_rejects_unknown_metric():
    async def _query():
        async with AsyncSession(db_test_engine) as session:
            return await load_metric_by_example(session, 1, "not_a_real_column", ["arm-a"])

    with pytest.raises(ValueError):
        asyncio.run(_query())


def test_summarize_arms_computes_means_and_completed_count():
    example_ids = _insert_examples(2)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, judge_score=4.0, latency_ms=100.0)
        _insert_result(run_id, example_ids[1], "arm-a", 0, judge_score=2.0, latency_ms=200.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a"])

        result = asyncio.run(_query())
        assert len(result) == 1
        summary = result[0]
        assert summary.arm_name == "arm-a"
        assert summary.n == 2
        assert summary.mean_judge_score == pytest.approx(3.0)
        assert summary.mean_latency_ms == pytest.approx(150.0)
        assert summary.mean_cost_estimate_usd is None
    finally:
        _cleanup(run_id, example_ids)


def test_summarize_arms_excludes_non_completed_rows():
    example_ids = _insert_examples(2)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, status="failed", latency_ms=999.0)
        _insert_result(run_id, example_ids[1], "arm-a", 0, status="completed", latency_ms=50.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a"])

        result = asyncio.run(_query())
        summary = result[0]
        assert summary.n == 1
        assert summary.mean_latency_ms == pytest.approx(50.0)
    finally:
        _cleanup(run_id, example_ids)


def test_summarize_arms_returns_arms_in_requested_order_with_zero_n_when_absent():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=10.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a", "arm-b"])

        result = asyncio.run(_query())
        assert [s.arm_name for s in result] == ["arm-a", "arm-b"]
        assert result[1].n == 0
        assert result[1].mean_latency_ms is None
    finally:
        _cleanup(run_id, example_ids)
