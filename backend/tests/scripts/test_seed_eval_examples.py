import asyncio
from pathlib import Path

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample
from scripts.seed_eval_examples import seed
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_phrasebank.txt"

# Deliberately NOT the production SOURCE constant: the test both deletes
# before and cleans up after, and must never touch rows seeded from the real
# Financial PhraseBank dataset.
TEST_SOURCE = "test_phrasebank_fixture"


async def _cleanup():
    async with AsyncSession(db_test_engine) as session:
        await session.execute(delete(EvalExample).where(EvalExample.source == TEST_SOURCE))
        await session.commit()


def test_seed_is_idempotent(monkeypatch):
    monkeypatch.setattr("scripts.seed_eval_examples.DATA_PATH", FIXTURE_PATH)
    monkeypatch.setattr("scripts.seed_eval_examples.SOURCE", TEST_SOURCE)
    # seed() is invoked from two separate asyncio.run() calls below; the
    # production pooled engine would hand loop 2 a connection created on
    # loop 1. Use the NullPool test engine instead.
    monkeypatch.setattr("scripts.seed_eval_examples.engine", db_test_engine)
    asyncio.run(_cleanup())

    try:
        first_count = asyncio.run(seed())
        second_count = asyncio.run(seed())

        assert first_count == 3
        assert second_count == 0

        async def _check():
            async with AsyncSession(db_test_engine) as session:
                result = await session.execute(
                    select(EvalExample).where(EvalExample.source == TEST_SOURCE)
                )
                return result.scalars().all()

        rows = asyncio.run(_check())
        assert len(rows) == 3
    finally:
        asyncio.run(_cleanup())
