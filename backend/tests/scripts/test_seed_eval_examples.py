import asyncio
import sys
from pathlib import Path

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.tasks import DEFAULT_TASK
from app.db.models import EvalExample
from scripts.seed_eval_examples import seed
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

# The fixture task pack from Task 1: a 3-row jsonl news-topic pack whose
# task.yaml declares source: task_sample.
FIXTURE_TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
TEST_TASK = "task_sample"
# Deliberately NOT the production source: the test deletes before and cleans
# up after, and must never touch rows seeded from the real dataset.
TEST_SOURCE = "task_sample"


async def _cleanup():
    async with AsyncSession(db_test_engine) as session:
        await session.execute(delete(EvalExample).where(EvalExample.source == TEST_SOURCE))
        await session.commit()


def test_seed_task_is_idempotent_and_writes_under_task_source(monkeypatch):
    # seed() is invoked from two separate asyncio.run() calls below; the
    # production pooled engine would hand loop 2 a connection created on
    # loop 1. Use the NullPool test engine instead.
    monkeypatch.setattr("scripts.seed_eval_examples.engine", db_test_engine)
    asyncio.run(_cleanup())

    try:
        first_count, first_source = asyncio.run(
            seed(task_name=TEST_TASK, tasks_dir=FIXTURE_TASKS_DIR)
        )
        second_count, second_source = asyncio.run(
            seed(task_name=TEST_TASK, tasks_dir=FIXTURE_TASKS_DIR)
        )

        assert first_count == 3
        assert second_count == 0
        assert first_source == TEST_SOURCE
        assert second_source == TEST_SOURCE

        async def _check():
            async with AsyncSession(db_test_engine) as session:
                result = await session.execute(
                    select(EvalExample).where(EvalExample.source == TEST_SOURCE)
                )
                return result.scalars().all()

        rows = asyncio.run(_check())
        assert len(rows) == 3
        assert {r.gold_label for r in rows} == {"Sports", "Business", "SciTech"}
    finally:
        asyncio.run(_cleanup())


def test_main_defaults_to_financial_sentiment(monkeypatch, capsys):
    captured = {}

    async def fake_seed(task_name=DEFAULT_TASK, tasks_dir=None):
        captured["task_name"] = task_name
        return 0, "financial_phrasebank_allagree"

    monkeypatch.setattr("scripts.seed_eval_examples.seed", fake_seed)
    monkeypatch.setattr(sys, "argv", ["seed_eval_examples"])

    from scripts.seed_eval_examples import main

    main()

    assert captured["task_name"] == DEFAULT_TASK
    out = capsys.readouterr().out
    assert "task=financial_sentiment" in out
    assert "source=financial_phrasebank_allagree" in out
