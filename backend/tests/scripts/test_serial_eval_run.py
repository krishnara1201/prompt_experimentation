import asyncio

import pytest
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.base import ModelResponse
from app.adapters.claude_code_cli import UsageLimitError
from app.config.arms import Arm
from app.db.models import EvalExample, Run, RunResult
from tests.conftest import db_test_engine, postgres_reachable


# --- plan_cells: pure, no DB -----------------------------------------------

def test_plan_cells_orders_non_cli_arms_before_cli_arms():
    from scripts.serial_eval_run import plan_cells

    chosen = [(1, "a"), (2, "b")]
    cells = plan_cells(
        chosen, ["local", "cli"], repeats=1, cli_arms={"cli"}, completed=set()
    )

    arm_order = [c.arm_name for c in cells]
    assert arm_order == ["local", "local", "cli", "cli"]


def test_plan_cells_skips_completed_cells():
    from scripts.serial_eval_run import plan_cells

    chosen = [(1, "a"), (2, "b")]
    cells = plan_cells(
        chosen,
        ["local"],
        repeats=1,
        cli_arms=set(),
        completed={(1, "local", 0)},
    )

    assert [(c.example_id, c.arm_name, c.repeat_index) for c in cells] == [(2, "local", 0)]


def test_plan_cells_expands_repeats():
    from scripts.serial_eval_run import plan_cells

    cells = plan_cells([(1, "a")], ["local"], repeats=3, cli_arms=set(), completed=set())

    assert [c.repeat_index for c in cells] == [0, 1, 2]


# --- run_cells: DB + fake adapters ---------------------------------------

pytestmark_db = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

TEST_SOURCE = "serial_eval_test"


class _ScriptedAdapter:
    """Returns a canned ModelResponse per call, or raises the exception at
    that position when the scripted value is an Exception."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def generate(self, prompt: str) -> ModelResponse:
        item = self._script[self.calls] if self.calls < len(self._script) else self._script[-1]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return ModelResponse(
            text=item, latency_ms=1.0, prompt_tokens=1, completion_tokens=1
        )


async def _seed_run(n_examples: int, arm_names: list[str]) -> tuple[int, list[int]]:
    async with AsyncSession(db_test_engine, expire_on_commit=False) as session:
        example_ids = []
        for i in range(n_examples):
            ex = EvalExample(text=f"sentence {i}", gold_label="neutral", source=TEST_SOURCE)
            session.add(ex)
            await session.flush()
            example_ids.append(ex.id)
        run = Run(
            arm_names=arm_names,
            repeats=1,
            total_calls=n_examples * len(arm_names),
            task="financial_sentiment",
            sample_size=n_examples,
            seed=1,
        )
        session.add(run)
        await session.commit()
        return run.id, example_ids


async def _cleanup(run_id: int) -> None:
    async with AsyncSession(db_test_engine) as session:
        await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
        await session.execute(delete(Run).where(Run.id == run_id))
        await session.execute(delete(EvalExample).where(EvalExample.source == TEST_SOURCE))
        await session.commit()


async def _results(run_id: int) -> list[RunResult]:
    async with AsyncSession(db_test_engine) as session:
        rows = (
            await session.execute(
                RunResult.__table__.select().where(RunResult.run_id == run_id)
            )
        ).all()
        return rows


@pytestmark_db
def test_run_cells_pauses_on_usage_limit_without_writing_a_row(monkeypatch):
    from scripts import serial_eval_run

    monkeypatch.setattr(serial_eval_run, "engine", db_test_engine)
    run_id, example_ids = asyncio.run(_seed_run(3, ["cli"]))
    chosen = [(eid, f"sentence {i}") for i, eid in enumerate(example_ids)]
    try:
        # 1st CLI call ok, 2nd hits the limit.
        adapter = _ScriptedAdapter(["neutral", UsageLimitError("limit"), "neutral"])
        arms = {"cli": Arm(name="cli", adapter=adapter)}

        outcome = asyncio.run(
            serial_eval_run.run_cells(
                run_id=run_id, chosen=chosen, arms=arms, cli_arms={"cli"},
                task_name="financial_sentiment",
            )
        )

        assert outcome.paused is True
        rows = asyncio.run(_results(run_id))
        assert len(rows) == 1  # only the first cell persisted
        assert adapter.calls == 2  # stopped at the limit, did not try cell 3
    finally:
        asyncio.run(_cleanup(run_id))


@pytestmark_db
def test_run_cells_resume_completes_remaining_cells_without_duplicates(monkeypatch):
    from scripts import serial_eval_run

    monkeypatch.setattr(serial_eval_run, "engine", db_test_engine)
    run_id, example_ids = asyncio.run(_seed_run(3, ["cli"]))
    chosen = [(eid, f"sentence {i}") for i, eid in enumerate(example_ids)]
    try:
        first = _ScriptedAdapter(["neutral", UsageLimitError("limit")])
        asyncio.run(
            serial_eval_run.run_cells(
                run_id=run_id,
                chosen=chosen,
                arms={"cli": Arm(name="cli", adapter=first)},
                cli_arms={"cli"},
                task_name="financial_sentiment",
            )
        )

        second = _ScriptedAdapter(["neutral", "neutral", "neutral"])
        outcome = asyncio.run(
            serial_eval_run.run_cells(
                run_id=run_id,
                chosen=chosen,
                arms={"cli": Arm(name="cli", adapter=second)},
                cli_arms={"cli"},
                task_name="financial_sentiment",
            )
        )

        assert outcome.paused is False
        rows = asyncio.run(_results(run_id))
        assert len(rows) == 3  # 1 from first pass + 2 from resume, no dupes
        assert second.calls == 2  # only the 2 outstanding cells
    finally:
        asyncio.run(_cleanup(run_id))


@pytestmark_db
def test_run_cells_records_non_limit_error_as_failed_and_continues(monkeypatch):
    from scripts import serial_eval_run

    monkeypatch.setattr(serial_eval_run, "engine", db_test_engine)
    run_id, example_ids = asyncio.run(_seed_run(2, ["cli"]))
    chosen = [(eid, f"sentence {i}") for i, eid in enumerate(example_ids)]
    try:
        # boom on both the call and its one retry -> cell fails; then a
        # clean "neutral" for the second cell.
        adapter = _ScriptedAdapter([RuntimeError("boom"), RuntimeError("boom"), "neutral"])
        outcome = asyncio.run(
            serial_eval_run.run_cells(
                run_id=run_id,
                chosen=chosen,
                arms={"cli": Arm(name="cli", adapter=adapter)},
                cli_arms={"cli"},
                task_name="financial_sentiment",
            )
        )

        assert outcome.paused is False
        rows = asyncio.run(_results(run_id))
        statuses = sorted(r._mapping["status"] for r in rows)
        assert statuses == ["completed", "failed"]
    finally:
        asyncio.run(_cleanup(run_id))
