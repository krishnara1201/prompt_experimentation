# Orchestration Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Celery + Redis to run the Financial PhraseBank eval set x arms x N repeats as background jobs, persisting every call's output/latency/tokens/cost to Postgres, with a minimal FastAPI layer to trigger and poll runs.

**Architecture:** A `POST /runs` endpoint samples eval examples and enqueues one Celery task per (example, arm, repeat) call; each task calls the existing Phase 1 `ModelAdapter.generate()`, retries on failure, and persists exactly one `RunResult` row per call via a fresh async engine (mirroring the `experimentation_copilot` worker pattern). `GET /runs/{id}` derives run status by comparing `RunResult` counts against a `total_calls` value stored on the `Run` row — there is no independently-written status field.

**Tech Stack:** Python 3.12, FastAPI, SQLModel, Celery + Redis, PostgreSQL + Alembic, `uv`, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-25-orchestration-layer-design.md`

## Global Constraints

- Eval set: Financial PhraseBank, 100%-agreement subset (`sentences_allagree`, ~2,264 sentences), vendored into the repo — not fetched at runtime.
- Async jobs: Celery + Redis, one task per (example, arm, repeat) call — not one task per arm.
- `Run` has no stored `status` column; status is always derived at request time from `RunResult` counts vs. `Run.total_calls`.
- `RunResult.judge_score` exists now (nullable) even though Phase 3 (not this plan) populates it.
- Per-call failures retry up to 3 times with exponential backoff before the row is persisted as `failed` — one bad call never blocks the rest of a run.
- Reuse the `experimentation_copilot` async-job pattern: sync Celery task body, persistence via `asyncio.run(...)` against a freshly created async engine with `NullPool` (asyncpg connections are bound to the event loop that created them).
- DB-touching tests require a real Postgres (asyncpg has no SQLite equivalent) and must skip — not fail — when it's unreachable, matching the existing `test_ollama_e2e.py` pattern (`socket.create_connection` reachability check + `@pytest.mark.skipif`).
- All new backend commands run from the `backend/` directory (matches the existing `uv run python -m app.demo` convention).

---

## Task 1: Docker Compose infrastructure (Postgres + Redis)

**Files:**
- Create: `docker-compose.yml` (repo root)
- Create: `backend/Dockerfile`
- Create: `.env.example` (repo root)
- Modify: `backend/.env.example`

**Interfaces:**
- Produces: a `postgres` service reachable at `localhost:${POSTGRES_PORT:-5432}` and a `redis` service at `localhost:${REDIS_PORT:-6379}`, both used by every later task's tests and by the `migrate`/`worker`/`api` services added to this same compose file (not started until their code exists in later tasks).

- [ ] **Step 1: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env, copied from .env.example}"
      POSTGRES_DB: ${POSTGRES_DB:-prompt_experimentation}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "127.0.0.1:${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-prompt_experimentation}"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:${REDIS_PORT:-6379}:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  migrate:
    build: ./backend
    command: ["uv", "run", "alembic", "upgrade", "head"]
    environment: &backend-env
      DATABASE_URL: "postgresql+asyncpg://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env, copied from .env.example}@postgres:5432/${POSTGRES_DB:-prompt_experimentation}"
      REDIS_URL: redis://redis:6379/0
      REDIS_BACKEND_URL: redis://redis:6379/1
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    depends_on:
      postgres:
        condition: service_healthy

  api:
    build: ./backend
    command: ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
    environment: *backend-env
    ports:
      - "127.0.0.1:${API_PORT:-8000}:8000"
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy

  worker:
    build: ./backend
    command: ["uv", "run", "celery", "-A", "app.tasks.worker.celery_app", "worker", "--loglevel=info"]
    environment: *backend-env
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy

volumes:
  postgres_data:
```

- [ ] **Step 3: Write root `.env.example`**

```
# Copy to .env before running `docker compose up` -- POSTGRES_PASSWORD is
# required (compose fails fast with a clear error if it's missing).

POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace-with-your-own-password
POSTGRES_DB=prompt_experimentation

# Host ports. Change any of these if something else on your machine already
# uses the default (e.g. a native Postgres install already on 5432).
POSTGRES_PORT=5432
REDIS_PORT=6379
API_PORT=8000

# API keys for hosted arms configured in backend/arms.yaml.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

- [ ] **Step 4: Update `backend/.env.example`**

Add these lines (keep the existing `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` lines and comment as-is):

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/prompt_experimentation
REDIS_URL=redis://localhost:6379/0
REDIS_BACKEND_URL=redis://localhost:6379/1
```

- [ ] **Step 5: Bring up Postgres + Redis and verify health**

```bash
cp .env.example .env   # edit POSTGRES_PASSWORD to something real
cp backend/.env.example backend/.env   # update DATABASE_URL's password to match
docker compose up -d postgres redis
docker compose ps
```

Expected: both `postgres` and `redis` show `healthy` within ~10-15 seconds.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml backend/Dockerfile .env.example backend/.env.example
git commit -m "chore: add docker-compose infra for Postgres and Redis"
```

---

## Task 2: Financial PhraseBank dataset — vendoring + parser

**Files:**
- Create: `backend/scripts/fetch_financial_phrasebank.py`
- Create: `backend/data/financial_phrasebank/sentences_allagree.txt` (generated by the script above)
- Create: `backend/app/data/__init__.py`
- Create: `backend/app/data/financial_phrasebank.py`
- Create: `backend/tests/data/__init__.py`
- Create: `backend/tests/data/fixtures/sample_phrasebank.txt`
- Test: `backend/tests/data/test_financial_phrasebank.py`

**Interfaces:**
- Produces: `PhrasebankExample` dataclass (`text: str`, `label: str`) and `load_examples(path: str | Path) -> list[PhrasebankExample]`, used by Task 4's seed script.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/data/fixtures/sample_phrasebank.txt`:

```
# comment header, should be skipped
Profits rose to $5.2@million in Q3.@positive
The company reported a loss.@negative
Nothing changed this quarter.@neutral
```

Create `backend/tests/data/test_financial_phrasebank.py`:

```python
from pathlib import Path

import pytest

from app.data.financial_phrasebank import load_examples

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_phrasebank.txt"
VENDORED_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "financial_phrasebank"
    / "sentences_allagree.txt"
)


def test_load_examples_parses_fixture_and_skips_comments():
    examples = load_examples(FIXTURE_PATH)

    assert len(examples) == 3
    assert examples[0].text == "Profits rose to $5.2@million in Q3."
    assert examples[0].label == "positive"
    assert examples[1].label == "negative"
    assert examples[2].label == "neutral"


def test_load_examples_raises_on_malformed_line(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("no delimiter here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_examples(bad_file)


def test_vendored_file_has_expected_shape():
    examples = load_examples(VENDORED_PATH)

    assert len(examples) == 2264
    assert all(e.label in {"positive", "negative", "neutral"} for e in examples)
    assert all(e.text for e in examples)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/data/test_financial_phrasebank.py -v`
Expected: FAIL — `app.data.financial_phrasebank` doesn't exist yet, and the vendored file doesn't exist yet.

- [ ] **Step 3: Write the parser**

Create `backend/app/data/__init__.py` (empty file).

Create `backend/app/data/financial_phrasebank.py`:

```python
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = {"positive", "negative", "neutral"}


@dataclass
class PhrasebankExample:
    text: str
    label: str


def load_examples(path: str | Path) -> list[PhrasebankExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            text, _, label = line.rpartition("@")
            if not text or label not in VALID_LABELS:
                raise ValueError(f"Malformed line in {path}: {line!r}")
            examples.append(PhrasebankExample(text=text, label=label))
    return examples
```

- [ ] **Step 4: Write and run the vendoring script**

Create `backend/scripts/fetch_financial_phrasebank.py`:

```python
"""One-time vendoring script -- downloads the Financial PhraseBank
100%-agreement subset and writes it to
backend/data/financial_phrasebank/sentences_allagree.txt in
`sentence@label` format.

Run once, from inside backend/:

    uv run --with datasets --with pandas python scripts/fetch_financial_phrasebank.py

Source: gtfintechlab/financial_phrasebank_sentences_allagree on Hugging
Face, a mirror of Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and
Takala, P. (2014), "Good debt or bad debt: Detecting semantic
orientations in economic texts." Licensed CC-BY-NC-SA-3.0 (non-commercial
use); see http://creativecommons.org/licenses/by-nc-sa/3.0/.
"""

from pathlib import Path

from datasets import load_dataset

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

LICENSE_NOTE = (
    "# Financial PhraseBank, 100%-agreement subset (2264 sentences).\n"
    "# Source: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P.\n"
    '# (2014). "Good debt or bad debt: Detecting semantic orientations in\n'
    '# economic texts." Mirrored via gtfintechlab/financial_phrasebank_sentences_allagree\n'
    "# on Hugging Face. Licensed CC-BY-NC-SA-3.0 (non-commercial use);\n"
    "# see http://creativecommons.org/licenses/by-nc-sa/3.0/.\n"
    "# Format: one sentence per line, '<sentence>@<label>', UTF-8.\n"
)


def main() -> None:
    dataset = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree")
    split = dataset["train"]

    out_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "financial_phrasebank"
        / "sentences_allagree.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(LICENSE_NOTE)
        for row in split:
            sentence = row["sentence"].replace("\n", " ").strip()
            label = LABEL_NAMES[row["label"]]
            f.write(f"{sentence}@{label}\n")

    print(f"Wrote {len(split)} sentences to {out_path}")


if __name__ == "__main__":
    main()
```

Run: `cd backend && uv run --with datasets --with pandas python scripts/fetch_financial_phrasebank.py`
Expected: `Wrote 2264 sentences to .../sentences_allagree.txt`

This script is a one-time tool — it stays in the repo for provenance/reproducibility, but `datasets`/`pandas` are deliberately *not* added to `pyproject.toml` (they're only ever needed for this one-off run, via `uv run --with`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/data/test_financial_phrasebank.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/data backend/tests/data backend/scripts/fetch_financial_phrasebank.py backend/data/financial_phrasebank/sentences_allagree.txt
git commit -m "feat: vendor Financial PhraseBank dataset and add parser"
```

---

## Task 3: Database models + Alembic migration

**Files:**
- Modify: `backend/pyproject.toml` (add `sqlmodel`, `asyncpg`, `alembic`)
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/models.py`
- Create: `backend/app/db/session.py`
- Create: `backend/alembic.ini`
- Create: `backend/migrations/env.py`, `backend/migrations/script.py.mako`, `backend/migrations/README`
- Create: `backend/migrations/versions/0001_create_initial_tables.py`
- Create: `backend/tests/db/__init__.py`
- Test: `backend/tests/db/test_models.py`

**Interfaces:**
- Consumes: none from earlier tasks.
- Produces: `EvalExample`, `Run`, `RunResult` SQLModel tables (`backend/app/db/models.py`); `engine`, `get_session()`, `lifespan()`, `DATABASE_URL` (`backend/app/db/session.py`) — used by Tasks 4, 5, 6.

- [ ] **Step 1: Add dependencies**

In `backend/pyproject.toml`, add to `dependencies`:

```toml
    "sqlmodel>=0.0.39",
    "asyncpg>=0.31.0",
    "alembic>=1.18.5",
```

Run: `cd backend && uv sync`
Expected: exits 0, `uv.lock` updated.

- [ ] **Step 2: Write the failing test**

Create `backend/tests/db/__init__.py` (empty file).

Create `backend/tests/db/test_models.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: FAIL — `app.db.models` and `app.db.session` don't exist yet.

- [ ] **Step 4: Write the models and session**

Create `backend/app/db/__init__.py` (empty file).

Create `backend/app/db/models.py`:

```python
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvalExample(SQLModel, table=True):
    __tablename__ = "eval_example"

    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    gold_label: str
    source: str


class Run(SQLModel, table=True):
    __tablename__ = "run"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    arm_names: list[str] = Field(sa_column=Column(JSON))
    sample_size: Optional[int] = Field(default=None)
    repeats: int
    seed: Optional[int] = Field(default=None)
    total_calls: int


class RunResult(SQLModel, table=True):
    __tablename__ = "run_result"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    example_id: int = Field(foreign_key="eval_example.id")
    arm_name: str
    repeat_index: int
    output_text: Optional[str] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None)
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    cost_estimate_usd: Optional[float] = Field(default=None)
    judge_score: Optional[float] = Field(default=None)
    status: str
    error_message: Optional[str] = Field(default=None)
    celery_task_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
```

Create `backend/app/db/session.py`:

```python
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine) as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()
```

- [ ] **Step 5: Set up Alembic and write the migration**

```bash
cd backend && uv run alembic init migrations
```

Edit `backend/alembic.ini`: set `script_location = %(here)s/migrations` (it should already be this after `init`; confirm it).

Replace `backend/migrations/env.py` entirely with:

```python
import asyncio
from logging.config import fileConfig
import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

from app.db.models import EvalExample, Run, RunResult  # noqa: F401 -- registers metadata

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def do_run_migrations(connection):
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()

    async def run_async_migrations():
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
        await connectable.dispose()

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `backend/migrations/versions/0001_create_initial_tables.py`:

```python
"""create eval_example, run, run_result tables

Revision ID: 0001_create_initial_tables
Revises:
Create Date: 2026-08-25

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0001_create_initial_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_example",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("gold_label", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("source", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("arm_names", sa.JSON(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=True),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("total_calls", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run_result",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("example_id", sa.Integer(), nullable=False),
        sa.Column("arm_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("repeat_index", sa.Integer(), nullable=False),
        sa.Column("output_text", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_estimate_usd", sa.Float(), nullable=True),
        sa.Column("judge_score", sa.Float(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("celery_task_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["run.id"]),
        sa.ForeignKeyConstraint(["example_id"], ["eval_example.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("run_result")
    op.drop_table("run")
    op.drop_table("eval_example")
```

Run: `cd backend && uv run alembic upgrade head`
Expected: `Running upgrade  -> 0001_create_initial_tables, create eval_example, run, run_result tables`

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/db backend/alembic.ini backend/migrations backend/tests/db
git commit -m "feat: add EvalExample/Run/RunResult models and initial migration"
```

---

## Task 4: Seed script

**Files:**
- Create: `backend/scripts/__init__.py`
- Create: `backend/scripts/seed_eval_examples.py`
- Create: `backend/tests/scripts/__init__.py`
- Test: `backend/tests/scripts/test_seed_eval_examples.py`

**Interfaces:**
- Consumes: `load_examples` (Task 2), `EvalExample` + `engine` (Task 3).
- Produces: `seed() -> int` (async, returns count of newly-inserted rows), `SOURCE: str`, `DATA_PATH: Path` — module-level names in `scripts.seed_eval_examples`, referenced by tests via `monkeypatch`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/scripts/__init__.py` (empty file).

Create `backend/tests/scripts/test_seed_eval_examples.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/scripts/test_seed_eval_examples.py -v`
Expected: FAIL — `scripts.seed_eval_examples` doesn't exist yet.

- [ ] **Step 3: Write the seed script**

Create `backend/scripts/__init__.py` (empty file).

Create `backend/scripts/seed_eval_examples.py`:

```python
"""Loads the vendored Financial PhraseBank file into the eval_example
table. Safe to re-run: skips sentences that already exist for this source.

Run from inside backend/: uv run python -m scripts.seed_eval_examples
"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.data.financial_phrasebank import load_examples
from app.db.models import EvalExample
from app.db.session import engine

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "financial_phrasebank" / "sentences_allagree.txt"
SOURCE = "financial_phrasebank_allagree"


async def seed() -> int:
    examples = load_examples(DATA_PATH)
    inserted = 0
    async with AsyncSession(engine) as session:
        existing_result = await session.execute(
            select(EvalExample.text).where(EvalExample.source == SOURCE)
        )
        existing_texts = set(existing_result.scalars().all())

        for example in examples:
            if example.text in existing_texts:
                continue
            session.add(EvalExample(text=example.text, gold_label=example.label, source=SOURCE))
            inserted += 1
        await session.commit()
    return inserted


def main() -> None:
    inserted = asyncio.run(seed())
    print(f"Inserted {inserted} new eval examples (source={SOURCE}).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/scripts/test_seed_eval_examples.py -v`
Expected: PASS

- [ ] **Step 5: Seed the real dataset**

```bash
cd backend && uv run python -m scripts.seed_eval_examples
```

Expected: `Inserted 2264 new eval examples (source=financial_phrasebank_allagree).`

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/seed_eval_examples.py backend/tests/scripts
git commit -m "feat: add idempotent seed script for eval examples"
```

---

## Task 5: Celery worker + call execution logic

**Files:**
- Modify: `backend/pyproject.toml` (add `celery[redis]`)
- Create: `backend/app/tasks/__init__.py`
- Create: `backend/app/tasks/worker.py`
- Create: `backend/tests/tasks/__init__.py`
- Test: `backend/tests/tasks/test_execute_call.py`

**Interfaces:**
- Consumes: `load_arms` (`app.config.arms`), `ModelResponse` (`app.adapters.base`), `RunResult` + `engine`/`DATABASE_URL` (Task 3).
- Produces: `execute_call(*, run_id, example_id, example_text, arm_name, repeat_index, celery_task_id=None, max_retries=3, backoff_base_seconds=1.0) -> None`, `celery_app: Celery`, `run_single_call` (Celery task) — used by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/tasks/__init__.py` (empty file).

Create `backend/tests/tasks/test_execute_call.py`:

```python
from unittest.mock import AsyncMock

from app.adapters.base import ModelResponse
from app.tasks import worker

SUCCESS = ModelResponse(text="positive", latency_ms=10.0, prompt_tokens=5, completion_tokens=1, cost_estimate_usd=0.0001)


class FakeAdapter:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, prompt):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_succeeds_on_first_try(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1
    assert sleep_calls == []
    persist_mock.assert_awaited_once()
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
    assert kwargs["response"] is SUCCESS


def test_retries_then_succeeds(monkeypatch):
    adapter = FakeAdapter([RuntimeError("timeout"), RuntimeError("timeout"), SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 3
    assert sleep_calls == [1.0, 2.0]
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"


def test_persists_failure_after_exhausting_retries(monkeypatch):
    adapter = FakeAdapter([RuntimeError("boom")] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(
        run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0, max_retries=3
    )

    assert adapter.calls == 4  # initial attempt + 3 retries
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "boom" in kwargs["error_message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v`
Expected: FAIL — `app.tasks` doesn't exist yet.

- [ ] **Step 3: Add the Celery dependency**

In `backend/pyproject.toml`, add to `dependencies`:

```toml
    "celery[redis]>=5.6.3",
```

Run: `cd backend && uv sync`

- [ ] **Step 4: Write the worker module**

Create `backend/app/tasks/__init__.py` (empty file).

Create `backend/app/tasks/worker.py`:

```python
import asyncio
import os
from pathlib import Path
import time

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.base import ModelResponse
from app.config.arms import load_arms
from app.db.models import RunResult
from app.db.session import DATABASE_URL

load_dotenv()

ARMS_PATH = Path(__file__).resolve().parent.parent.parent / "arms.yaml"

celery_app = Celery(
    "worker",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_BACKEND_URL"),
    broker_connection_retry_on_startup=True,
)


async def _persist_run_result(
    *,
    run_id: int,
    example_id: int,
    arm_name: str,
    repeat_index: int,
    celery_task_id: str | None,
    status: str,
    response: ModelResponse | None = None,
    error_message: str | None = None,
) -> None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    celery_task_id=celery_task_id,
                    status=status,
                    output_text=response.text if response else None,
                    latency_ms=response.latency_ms if response else None,
                    prompt_tokens=response.prompt_tokens if response else None,
                    completion_tokens=response.completion_tokens if response else None,
                    cost_estimate_usd=response.cost_estimate_usd if response else None,
                    error_message=error_message,
                )
            )
            await session.commit()
    finally:
        await worker_engine.dispose()


def execute_call(
    *,
    run_id: int,
    example_id: int,
    example_text: str,
    arm_name: str,
    repeat_index: int,
    celery_task_id: str | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
) -> None:
    arms = load_arms(str(ARMS_PATH))
    adapter = arms[arm_name]

    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retries:
        try:
            response = adapter.generate(example_text)
            asyncio.run(
                _persist_run_result(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    celery_task_id=celery_task_id,
                    status="completed",
                    response=response,
                )
            )
            return
        except Exception as exc:
            last_exc = exc
            attempt += 1
            if attempt <= max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))

    asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name=arm_name,
            repeat_index=repeat_index,
            celery_task_id=celery_task_id,
            status="failed",
            error_message=str(last_exc),
        )
    )


@celery_app.task(bind=True)
def run_single_call(self, run_id: int, example_id: int, example_text: str, arm_name: str, repeat_index: int) -> None:
    execute_call(
        run_id=run_id,
        example_id=example_id,
        example_text=example_text,
        arm_name=arm_name,
        repeat_index=repeat_index,
        celery_task_id=self.request.id,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/tasks backend/tests/tasks
git commit -m "feat: add Celery worker with retrying call execution"
```

---

## Task 6: FastAPI app + run endpoints

**Files:**
- Modify: `backend/pyproject.toml` (add `fastapi[standard]`)
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes/__init__.py`
- Create: `backend/app/api/routes/runs.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/api/__init__.py`
- Test: `backend/tests/api/test_runs.py`

**Interfaces:**
- Consumes: `EvalExample`, `Run`, `RunResult`, `get_session`, `engine` (Task 3); `run_single_call` (Task 5); `load_arms` (`app.config.arms`).
- Produces: `POST /runs`, `GET /runs/{run_id}`, `GET /runs/{run_id}/results` — the API Phase 5's dashboard will call.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/api/__init__.py` (empty file).

Create `backend/tests/api/test_runs.py`:

```python
import asyncio
import socket
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run
from app.db.session import DATABASE_URL, engine
from app.main import app


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

FAKE_ARMS = {"fake-arm": object()}


def _insert_example() -> int:
    async def _run():
        async with AsyncSession(engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)
            return example.id

    return asyncio.run(_run())


def _delete_example(example_id: int) -> None:
    async def _run():
        async with AsyncSession(engine) as session:
            obj = await session.get(EvalExample, example_id)
            if obj:
                await session.delete(obj)
                await session.commit()

    asyncio.run(_run())


def _delete_run(run_id: int) -> None:
    async def _run():
        async with AsyncSession(engine) as session:
            obj = await session.get(Run, run_id)
            if obj:
                await session.delete(obj)
                await session.commit()

    asyncio.run(_run())


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_enqueues_expected_number_of_calls(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        response = TestClient(app).post("/runs", json={"repeats": 2, "sample_size": 1, "seed": 1})
        assert response.status_code == 200
        body = response.json()
        run_id = body["run_id"]
        assert body["status"] == "pending"
        assert body["total_calls"] == 2  # 1 example x 1 arm x 2 repeats
        assert mock_task.delay.call_count == 2
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_rejects_unknown_arm(mock_load_arms, mock_task):
    response = TestClient(app).post("/runs", json={"arms": ["nonexistent-arm"]})
    assert response.status_code == 400


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_get_run_status_is_pending_before_any_results(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        create_response = TestClient(app).post("/runs", json={"repeats": 1, "sample_size": 1, "seed": 1})
        run_id = create_response.json()["run_id"]

        status_response = TestClient(app).get(f"/runs/{run_id}")
        assert status_response.status_code == 200
        body = status_response.json()
        assert body["status"] == "pending"
        assert body["total_calls"] == 1
        assert body["completed"] == 0
        assert body["failed"] == 0
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)


def test_get_run_status_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: FAIL — `app.main` and `app.api.routes.runs` don't exist yet.

- [ ] **Step 3: Add the FastAPI dependency**

In `backend/pyproject.toml`, add to `dependencies`:

```toml
    "fastapi[standard]>=0.140.0",
```

Run: `cd backend && uv sync`

- [ ] **Step 4: Write the routes and app**

Create `backend/app/api/__init__.py` and `backend/app/api/routes/__init__.py` (both empty).

Create `backend/app/api/routes/runs.py`:

```python
from pathlib import Path
import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.arms import load_arms
from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.tasks.worker import run_single_call

router = APIRouter(prefix="/runs", tags=["runs"])

ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class RunCreateRequest(BaseModel):
    arms: list[str] | None = None
    sample_size: int | None = Field(default=None, gt=0)
    repeats: int = Field(default=1, ge=1)
    seed: int | None = None


class RunCreateResponse(BaseModel):
    run_id: int
    status: str
    total_calls: int


class RunStatusResponse(BaseModel):
    run_id: int
    status: str
    total_calls: int
    completed: int
    failed: int
    pending: int


@router.post("", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest, session: AsyncSession = Depends(get_session)):
    available_arms = load_arms(str(ARMS_PATH))
    arm_names = payload.arms if payload.arms is not None else list(available_arms.keys())
    unknown = [name for name in arm_names if name not in available_arms]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown arm(s): {', '.join(unknown)}")

    result = await session.execute(select(EvalExample.id, EvalExample.text))
    all_examples = result.all()
    if not all_examples:
        raise HTTPException(status_code=400, detail="No eval examples found; run the seed script first")

    seed = payload.seed
    if payload.sample_size is not None:
        seed = seed if seed is not None else random.randrange(2**31)
        chosen = random.Random(seed).sample(all_examples, min(payload.sample_size, len(all_examples)))
    else:
        chosen = all_examples

    total_calls = len(chosen) * len(arm_names) * payload.repeats
    run = Run(
        arm_names=arm_names,
        sample_size=payload.sample_size,
        repeats=payload.repeats,
        seed=seed,
        total_calls=total_calls,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    for example_id, example_text in chosen:
        for arm_name in arm_names:
            for repeat_index in range(payload.repeats):
                run_single_call.delay(
                    run_id=run.id,
                    example_id=example_id,
                    example_text=example_text,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                )

    return RunCreateResponse(run_id=run.id, status="pending", total_calls=total_calls)


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    counts_result = await session.execute(
        select(RunResult.status, func.count()).where(RunResult.run_id == run_id).group_by(RunResult.status)
    )
    counts = dict(counts_result.all())
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    done = completed + failed
    pending = run.total_calls - done

    if done == 0:
        status = "pending"
    elif done < run.total_calls:
        status = "running"
    elif failed == 0:
        status = "completed"
    else:
        status = "completed_with_errors"

    return RunStatusResponse(
        run_id=run.id,
        status=status,
        total_calls=run.total_calls,
        completed=completed,
        failed=failed,
        pending=pending,
    )


@router.get("/{run_id}/results", response_model=list[RunResult])
async def get_run_results(
    run_id: int, limit: int = 100, offset: int = 0, session: AsyncSession = Depends(get_session)
):
    result = await session.execute(
        select(RunResult).where(RunResult.run_id == run_id).offset(offset).limit(limit)
    )
    return result.scalars().all()
```

Create `backend/app/main.py`:

```python
from fastapi import FastAPI

from app.api.routes.runs import router as runs_router
from app.db.session import lifespan

app = FastAPI(title="Prompt Experimentation API", lifespan=lifespan)
app.include_router(runs_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Run the full backend test suite**

Run: `cd backend && uv run pytest -v`
Expected: all tests pass (Phase 1's suite plus everything added in this plan); DB-dependent tests only run if `docker compose up -d postgres redis` is active, otherwise they report `SKIPPED`, not `FAILED`.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/api backend/app/main.py backend/tests/api
git commit -m "feat: add FastAPI run endpoints (create, status, results)"
```

---

## Manual end-to-end check (not automated)

After Task 6, confirm the whole pipeline works against a live worker:

```bash
docker compose up -d postgres redis
cd backend && uv run alembic upgrade head && uv run python -m scripts.seed_eval_examples
uv run celery -A app.tasks.worker.celery_app worker --loglevel=info &
uv run fastapi run app/main.py &
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"sample_size": 3, "repeats": 1}'
# note the run_id from the response, then:
curl localhost:8000/runs/<run_id>
curl localhost:8000/runs/<run_id>/results
```

Expected: status moves from `pending` to `running` to `completed` (or `completed_with_errors` if an arm lacks a configured API key), and `/results` shows real adapter output, latency, and token counts for each configured arm.
