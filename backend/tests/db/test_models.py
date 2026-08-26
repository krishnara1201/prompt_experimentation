import asyncio
import socket
from urllib.parse import urlsplit

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.db.session import DATABASE_URL, engine


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


def test_round_trip_insert_and_read():
    async def _run():
        async with AsyncSession(engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)

            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            fetched = await session.get(RunResult, result.id)
            assert fetched is not None
            assert fetched.output_text == "positive"
            assert fetched.judge_score is None

            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())
