import asyncio
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.main import app
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(db_test_engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _use_test_engine_session():
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.pop(get_session, None)


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
    latency_ms: float | None = None,
    judge_score: float | None = None,
    judge_status: str = "completed",
) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=0,
                    status="completed",
                    judge_status=judge_status,
                    judge_score=judge_score,
                    latency_ms=latency_ms,
                    output_text="x",
                )
            )
            await session.commit()

    asyncio.run(_run())


def _cleanup(run_id: int, example_ids: list[int]) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
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


def _seed_two_arm_run(n_examples: int = 6, offset: float = 2.0) -> tuple[int, list[int]]:
    example_ids = _insert_examples(n_examples)
    run_id = _insert_run(["arm-a", "arm-b"])
    for i, example_id in enumerate(example_ids):
        _insert_result(run_id, example_id, "arm-a", latency_ms=float(i) + offset)
        _insert_result(run_id, example_id, "arm-b", latency_ms=float(i))
    return run_id, example_ids


def _seed_two_arm_run_judge_score(n_examples: int = 10, offset: float = 0.5) -> tuple[int, list[int]]:
    example_ids = _insert_examples(n_examples)
    run_id = _insert_run(["arm-a", "arm-b"])
    for i, example_id in enumerate(example_ids):
        _insert_result(run_id, example_id, "arm-a", judge_score=float(i) + offset)
        _insert_result(run_id, example_id, "arm-b", judge_score=float(i))
    return run_id, example_ids


def test_compare_arms_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999/compare?metric=latency_ms")
    assert response.status_code == 404


def test_compare_arms_422_for_unknown_metric():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=not_a_metric")
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_400_for_unknown_arm():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=not-an-arm"
        )
        assert response.status_code == 400
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_returns_paired_result_for_explicit_pair():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=arm-b&bootstrap_samples=200"
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["n_examples"] == 6
        assert body[0]["mean_diff"] == pytest.approx(2.0)
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_all_pairs_with_holm_correction_for_three_arms():
    example_ids = _insert_examples(6)
    run_id = _insert_run(["arm-a", "arm-b", "arm-c"])
    try:
        for i, example_id in enumerate(example_ids):
            _insert_result(run_id, example_id, "arm-a", latency_ms=float(i) + 3.0)
            _insert_result(run_id, example_id, "arm-b", latency_ms=float(i))
            _insert_result(run_id, example_id, "arm-c", latency_ms=float(i) + 1.0)

        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=latency_ms&bootstrap_samples=200")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3
        for row in body:
            assert row["p_value_corrected"] >= row["p_value"]
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_422_when_insufficient_paired_examples():
    run_id, example_ids = _seed_two_arm_run(n_examples=3)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=arm-b")
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)


def test_equivalence_returns_probability_between_zero_and_one():
    run_id, example_ids = _seed_two_arm_run_judge_score(n_examples=10, offset=0.1)
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/equivalence?metric=judge_score&arm_local=arm-a&arm_api=arm-b&epsilon=1.0"
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["p_equivalent"] <= 1.0
        assert body["ci_lower"] <= body["posterior_mean"] <= body["ci_upper"]
        # Fix 5: n_excluded must be reported, not silently dropped.
        assert body["n_excluded"] == 0
    finally:
        _cleanup(run_id, example_ids)


def test_equivalence_422_for_non_judge_score_metric():
    # Fix 4: equivalence's p_equivalent = P(mu >= -epsilon) is directionally
    # correct only for judge_score (higher-is-better); the other allowed
    # metrics are lower-is-better, so the endpoint now rejects them outright
    # rather than silently reporting the inverted question.
    run_id, example_ids = _seed_two_arm_run(n_examples=10, offset=0.1)
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/equivalence?metric=latency_ms&arm_local=arm-a&arm_api=arm-b&epsilon=1.0"
        )
        assert response.status_code == 422
        assert "judge_score" in response.json()["detail"]
    finally:
        _cleanup(run_id, example_ids)


def test_equivalence_reports_n_excluded_for_asymmetric_arm_coverage():
    # Fix 5: n_excluded must be reported, not silently dropped. Seed 10
    # paired examples plus one extra example scored only for arm-a, so it's
    # excluded from the paired diffs.
    run_id, example_ids = _seed_two_arm_run_judge_score(n_examples=10, offset=0.1)
    extra_example_id = _insert_examples(1)[0]
    _insert_result(run_id, extra_example_id, "arm-a", judge_score=5.0)
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/equivalence?metric=judge_score&arm_local=arm-a&arm_api=arm-b&epsilon=1.0"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_excluded"] == 1
        assert body["n_examples"] == 10
    finally:
        _cleanup(run_id, example_ids + [extra_example_id])


def test_power_returns_required_n_and_achieved_power():
    run_id, example_ids = _seed_two_arm_run(n_examples=10, offset=2.0)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/power?metric=latency_ms&arm_a=arm-a&arm_b=arm-b")
        assert response.status_code == 200
        body = response.json()
        assert body["required_n"] > 0
        assert 0.0 <= body["achieved_power"] <= 1.0
        # Fix 5: n_excluded must be reported, not silently dropped.
        assert body["n_excluded"] == 0
    finally:
        _cleanup(run_id, example_ids)


def test_power_returns_422_for_zero_effect_size():
    # Fix 1: estimate_sample_size raises a plain ValueError (not the
    # InsufficientDataError subclass) when the effective effect size is
    # zero -- reachable via effect_size=0 in the query string. The handler
    # must map this to a 422, not let it escape as an unhandled 500.
    run_id, example_ids = _seed_two_arm_run(n_examples=10, offset=2.0)
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/power?metric=latency_ms&arm_a=arm-a&arm_b=arm-b&effect_size=0"
        )
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_422_for_bootstrap_samples_over_cap():
    # Fix 3: bootstrap_samples has no upper bound, so a client could request
    # an arbitrarily large bootstrap and tie up a worker thread for hours.
    # FastAPI's own Query(le=200_000) validation should reject this before
    # the handler runs.
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=arm-b&bootstrap_samples=200001"
        )
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)
