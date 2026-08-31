import asyncio

import pytest
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import Run
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running")


def test_run_row_defaults_task_to_financial_sentiment():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=["a"], repeats=1, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)
            rid = run.id
        async with AsyncSession(db_test_engine) as session:
            fetched = (await session.execute(select(Run).where(Run.id == rid))).scalar_one()
            assert fetched.task == "financial_sentiment"
            await session.delete(fetched)
            await session.commit()

    asyncio.run(_run())


def test_run_row_persists_explicit_task():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=["a"], repeats=1, total_calls=1, task="ag_news")
            session.add(run)
            await session.commit()
            await session.refresh(run)
            assert run.task == "ag_news"
            await session.delete(run)
            await session.commit()

    asyncio.run(_run())
