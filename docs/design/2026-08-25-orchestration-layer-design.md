# Orchestration Layer — Design Spec

Date: 2026-08-25
Status: Approved for implementation
Scope: part of the platform described in `docs/ARCHITECTURE.md` — wire Celery +
Redis to run (eval set) x (arms) x (N repeats) as async jobs, and persist raw
outputs to Postgres.

## Context

Phase 1 (see `docs/design/2026-08-25-model-adapter-layer-design.md`)
delivered a stable `ModelAdapter` protocol with two adapter implementations,
config-driven arms (`backend/arms.yaml`), and a demo script that runs a
handful of hardcoded prompts through every arm synchronously with no
persistence.

Phase 2 turns that into a real batch evaluation run: a configurable eval set,
run against a configurable subset of arms, with N repeats per (example, arm)
pair, executed as background jobs and persisted to Postgres so later phases
(judge calibration, stats, dashboard) have something to read.

This resolves two open decisions from `docs/ARCHITECTURE.md`:

- **Eval dataset**: Financial PhraseBank, 100%-annotator-agreement subset
  (~2,264 sentences). Chosen over FiQA and over the noisier 50/66/75%-agreement
  subsets because this project's core value proposition is statistical rigor
  (paired tests, judge calibration against human labels) — cleaner ground
  truth matters more here than raw sample count.
- **Async job approach**: Celery + Redis, reusing the pattern already proven
  in the sibling `experimentation_copilot` repo, rather than a hand-rolled
  DB-polling queue. On typical hosting (e.g. Render), a continuously-running
  worker process costs the same either way (~$7/mo for the one container),
  and Redis has a usable free tier there — so Celery's built-in retry and
  concurrency handling comes at no extra hosting cost over the DIY
  alternative.

## Architecture

```
[Financial PhraseBank (vendored, 100%-agreement)] --seed script--> [EvalExample table]

POST /runs {arms?, sample_size?, repeats, seed?}
   -> creates Run row, samples example_ids, enqueues N x M x R Celery tasks
   -> returns run_id immediately

Celery worker (Redis broker/backend)
   -> run_single_call(run_id, example_id, arm_name, repeat_index) x (N x M x R)
   -> loads arm from arms.yaml, calls existing Phase 1 adapter.generate()
   -> persists one RunResult row per call (fresh async engine + NullPool,
      per experimentation_copilot's worker pattern)
   -> autoretries (3x, exponential backoff) on failure; persists
      status=failed with the error message after retries are exhausted

GET /runs/{id}         -> status computed from RunResult counts
                           (completed/failed/pending vs total expected)
GET /runs/{id}/results -> paginated raw rows, for Phase 4 (stats) and
                           Phase 5 (dashboard) to consume later
```

## Data model

Three new tables (SQLModel, Postgres via Alembic migration):

- **`EvalExample`**
  - `id: int` (PK)
  - `text: str`
  - `gold_label: str` (positive/negative/neutral)
  - `source: str` (e.g. `"financial_phrasebank_allagree"`)
  - Seeded once from the vendored dataset file via a standalone script, not
    created per-run and not seeded by the Alembic migration itself (migrations
    stay schema-only; seeding is a separate, explicit step).

- **`Run`**
  - `id: int` (PK)
  - `created_at: datetime`
  - `arm_names: JSON` (list of arm names from `arms.yaml`; omitting this in
    the request means "all configured arms")
  - `sample_size: int | None` (None means the full eval set)
  - `repeats: int`
  - `seed: int | None` (for reproducible subset sampling when `sample_size`
    is set)
  - `total_calls: int` (`len(example_ids) * len(arm_names) * repeats`,
    computed once at creation — the stable denominator `GET /runs/{id}` uses
    to derive status; see Error handling)

- **`RunResult`**
  - `id: int` (PK)
  - `run_id: int` (FK -> Run)
  - `example_id: int` (FK -> EvalExample)
  - `arm_name: str`
  - `repeat_index: int`
  - `output_text: str | None`
  - `latency_ms: float | None`
  - `prompt_tokens: int | None`
  - `completion_tokens: int | None`
  - `cost_estimate_usd: float | None`
  - `judge_score: float | None` — nullable, populated by Phase 3. Included
    now (rather than added via a later migration) because it's named
    explicitly alongside latency/tokens/cost in `docs/ARCHITECTURE.md`'s results-store
    description, and Phase 3 will be writing into this same table's existing
    rows rather than creating a new one.
  - `status: str` (`completed` / `failed`)
  - `error_message: str | None`
  - `celery_task_id: str | None`
  - `created_at: datetime`

A `RunResult` row is inserted exactly once, when its task reaches a terminal
state — there is no `pending` value for `RunResult.status` because a call
that hasn't finished yet simply has no row. `Run` has no `status` column at
all: `GET /runs/{run_id}` derives it at request time from the count of
`RunResult` rows (by status) compared against `Run.total_calls` (see Error
handling).

## Components

- `backend/app/db/models.py` — the three SQLModel tables above.
- `backend/app/db/session.py` — async engine + `get_session` FastAPI
  dependency, mirroring `experimentation_copilot/backend/app/db/session.py`.
- `backend/app/tasks/worker.py` — Celery app (`REDIS_URL` broker,
  `REDIS_BACKEND_URL` backend) and the `run_single_call` task. The task is a
  plain sync function (Celery serializes args to JSON, so it takes primitive
  IDs/strings, not ORM objects), calls the Phase 1 adapter's `generate()`
  directly (already a sync HTTP call via `httpx`), and persists its result
  via `asyncio.run(...)` against a freshly created async engine with
  `NullPool` — reusing `experimentation_copilot`'s reasoning that asyncpg
  connections are bound to the event loop that created them, so a
  module-level shared engine breaks across Celery's forked workers /
  repeated `asyncio.run()` calls.
- `backend/app/api/routes/runs.py` + `backend/app/main.py` — FastAPI app
  exposing:
  - `POST /runs` — body `{arms?: list[str], sample_size?: int, repeats: int, seed?: int}`.
    Creates the `Run` row, samples `example_id`s (seeded `random.sample` when
    `sample_size` is given), enqueues one `run_single_call.delay(...)` per
    (example, arm, repeat), returns `{run_id, status, total_calls}`.
  - `GET /runs/{run_id}` — status + counts (`completed`/`failed`/`pending`/`total`)
    and aggregate cost/latency once available.
  - `GET /runs/{run_id}/results` — paginated raw `RunResult` rows.
- `backend/app/data/financial_phrasebank.py` — parser for the vendored
  dataset file (`sentence@label` per line, ISO-8859-1 encoded, per the
  original distribution format).
- `backend/scripts/seed_eval_examples.py` — one-off script that loads the
  vendored file via the parser above and inserts `EvalExample` rows,
  skipping rows that already exist (safe to re-run).
- `backend/data/financial_phrasebank/sentences_allagree.txt` — vendored
  dataset file, with a header comment citing Malo et al., 2014 and its
  CC-BY-NC-SA 3.0 license (non-commercial use — this project is a portfolio
  demonstration, not a commercial product, so this is compatible).
- `docker-compose.yml` (repo root) — `postgres`, `redis`, `migrate`, `worker`,
  `api` services, adapted from `experimentation_copilot/docker-compose.yml`
  (no `frontend`/auth services yet — those aren't part of this repo).
- `backend/alembic.ini` + `backend/alembic/versions/` — schema migration for
  the three new tables.

New `backend/pyproject.toml` dependencies: `celery[redis]`, `sqlmodel`,
`asyncpg`, `psycopg2-binary` (Alembic's sync driver), `alembic`, `fastapi`,
`uvicorn`. New env vars (documented in `backend/.env.example`):
`DATABASE_URL`, `REDIS_URL`, `REDIS_BACKEND_URL`.

## Data flow

1. Operator runs `uv run python -m backend.scripts.seed_eval_examples` once
   to populate `EvalExample` from the vendored file.
2. Client calls `POST /runs` with an optional arm subset, sample size, and
   repeat count.
3. The endpoint samples example IDs, creates the `Run` row (storing
   `total_calls = len(example_ids) * len(arm_names) * repeats`), enqueues
   that many Celery tasks, and returns `run_id`.
4. Each `run_single_call` task resolves its arm from `arms.yaml`, calls
   `adapter.generate(example.text)`, and writes one `RunResult` row with the
   response fields (or an error row on failure).
5. Client polls `GET /runs/{run_id}` until `completed` or
   `completed_with_errors`; `GET /runs/{run_id}/results` returns the raw rows
   for downstream consumption (Phase 3 judge scoring reads and updates these
   same rows; Phase 4/5 read them for stats and the dashboard).

## Error handling

- Per-call failures (timeout, rate limit, malformed response, etc.) trigger
  Celery's `autoretry_for=(Exception,)` with `retry_backoff=True` and
  `max_retries=3`. After retries are exhausted, the task persists the
  `RunResult` row with `status=failed` and the exception message in
  `error_message` — one bad call never blocks or fails the rest of the run.
- `Run` has no stored status; `GET /runs/{run_id}` computes it each time from
  `RunResult` counts against `Run.total_calls`: `pending` when zero rows
  exist yet, `running` when `0 < count(RunResult) < total_calls`,
  `completed` when `count(RunResult) == total_calls` and none are `failed`,
  `completed_with_errors` when `count(RunResult) == total_calls` and at least
  one is `failed`.
- Concurrency/rate limiting is handled by Celery worker concurrency
  (`--concurrency` flag) rather than per-arm throttling logic; per-arm rate
  limiting is deferred until it's actually observed to be a problem.

## Testing

- Unit tests for the dataset parser (`financial_phrasebank.py`) and for the
  example-sampling logic (pure functions, no I/O).
- Integration test for `run_single_call` against a real Postgres instance —
  skipped (not failed) if unreachable, matching Phase 1's Ollama e2e test
  pattern, since asyncpg needs a real Postgres rather than SQLite.
- API tests for `POST /runs` and `GET /runs/{run_id}` using FastAPI's test
  client, with Celery's `.delay()` mocked so these tests don't require a live
  worker or Redis.

## Out of scope

- Judge-as-LLM scoring and calibration (Phase 3) — `RunResult.judge_score`
  exists as a column now but stays `NULL` until Phase 3 populates it.
- Statistical analysis (paired bootstrap/Wilcoxon, Bayesian equivalence,
  sample-size calculator) — Phase 4.
- Dashboard UI — Phase 5. `GET /runs/{id}` and `GET /runs/{id}/results` are
  built now so Phase 5 has an API to call without rework, but no frontend
  code is part of this phase.
- Per-arm rate limiting, run cancellation, and auth/multi-tenancy on the API
  — none of these are needed for a single-user portfolio project; add only
  if they become an actual blocker.
