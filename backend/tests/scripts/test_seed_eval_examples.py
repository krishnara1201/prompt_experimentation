import asyncio
import socket
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample
from app.db.session import DATABASE_URL, engine
from scripts.seed_eval_examples import SOURCE, seed


def _postgres_reachable() -> bool:
    if not DATABASE_URL:
        return False
    parts = urlsplit(DATABASE_URL.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_phrasebank.txt"


async def _cleanup():
    async with AsyncSession(engine) as session:
        await session.execute(delete(EvalExample).where(EvalExample.source == SOURCE))
        await session.commit()


def test_seed_is_idempotent(monkeypatch):
    monkeypatch.setattr("scripts.seed_eval_examples.DATA_PATH", FIXTURE_PATH)
    asyncio.run(_cleanup())

    try:
        first_count = asyncio.run(seed())
        second_count = asyncio.run(seed())

        assert first_count == 3
        assert second_count == 0

        async def _check():
            async with AsyncSession(engine) as session:
                result = await session.execute(select(EvalExample).where(EvalExample.source == SOURCE))
                return result.scalars().all()

        rows = asyncio.run(_check())
        assert len(rows) == 3
    finally:
        asyncio.run(_cleanup())
