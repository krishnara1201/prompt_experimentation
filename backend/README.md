# Backend

Phase 1 (model adapter layer) and Phase 2 (orchestration layer) of the LLM
prompt-experimentation platform. Designs live at the repo root in
`docs/superpowers/specs/2026-08-25-model-adapter-layer-design.md` and
`docs/superpowers/specs/2026-08-25-orchestration-layer-design.md`.

## Phase 1: Model adapter layer

A unified adapter interface so local (Ollama) and hosted API models are
interchangeable behind one code path.

## Setup

All commands in this README assume the current directory is `backend/`.

```bash
cd backend && uv sync
cp .env.example .env   # fill in whichever API keys you have; local Ollama needs none
```

Local model: `ollama pull qwen3:8b` (requires [Ollama](https://ollama.com) running).

## Run the demo

```bash
uv run python -m app.demo
```

Runs every arm in `arms.yaml` over a handful of prompts and prints
text/latency/tokens/cost side by side. Arms with no configured API key fail
gracefully (printed as `FAILED`) rather than crashing the run.

## Add or swap an arm

Edit `arms.yaml` — no code changes needed. `adapter: openai_compatible`
works for Ollama and any provider using the OpenAI chat-completions schema
(OpenAI, OpenRouter, Groq, Together, etc.); `adapter: anthropic` is for
Claude models.

## Subscription-seat CLI arms (Claude Code, Codex)

`adapter: claude_code_cli` and `adapter: codex_cli` drive the `claude` and
`codex` CLIs directly, non-interactively, instead of calling a metered API.
Unlike every other arm type, these have no per-call price — `arms.yaml`
should not set `price_per_1m_input`/`price_per_1m_output` on them, and
`cost_estimate_usd` on their results is always `null`.

**Precondition**: the machine running the Celery worker must already have
an authenticated CLI session under your subscription — run `claude login`
or `codex login` yourself first. Neither adapter reads or stores
credentials; they only shell out to whatever session already exists.

**Tool use stays on.** Each call still runs from a fresh, empty scratch
directory (created and torn down per call), so neither CLI can read this
repo's `CLAUDE.md`/`AGENTS.md` or touch real files, but within that empty
directory the CLI is free to use tools exactly as it would interactively.

**Run a second, low-concurrency worker for these arms** — a CLI subprocess
call is heavier than an HTTP call, and a subscription session may not
tolerate the same parallelism as the API arms:

```bash
uv run celery -A app.tasks.worker.celery_app worker -Q subscription_cli --concurrency=1 --loglevel=info
```

Keep your existing worker command (`-Q celery`, or no `-Q` flag, which
defaults to the same queue) running alongside it — one worker per queue,
both pointed at the same Redis broker.

## Phase 2: Orchestration

Runs (eval set) × (arms) × (N repeats) as async Celery jobs and persists
every call to Postgres.

### 1. Start Postgres and Redis

From the repo root (`docker-compose.yml` lives there):

```bash
docker compose up -d postgres redis
```

Set `POSTGRES_PASSWORD` in the repo-root `.env` first — compose refuses to
start without it.

### 2. Apply migrations

```bash
uv run alembic upgrade head
```

### 3. Seed the eval dataset

Loads the vendored Financial PhraseBank all-agree sentences into
`eval_example`. Idempotent — safe to re-run.

```bash
uv run python -m scripts.seed_eval_examples
```

If you are running everything through Docker Compose instead, seed with a
one-off container against the same image (seeding is a one-time idempotent
operation, not a service, so there is no `seed` service in the compose file):

```bash
docker compose run --rm migrate uv run python -m scripts.seed_eval_examples
```

### 4. Start the Celery worker

```bash
uv run celery -A app.tasks.worker.celery_app worker --loglevel=info
```

### 5. Start the API

```bash
uv run fastapi run app/main.py
```

Or bring the whole stack up at once: `docker compose up -d` (starts
postgres, redis, migrations, the API on `:8000`, and the worker).

### Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /runs` | Create a run: picks the examples (optionally a seeded `sample_size` sample), fans out one Celery task per example × arm × repeat, returns `run_id` and `total_calls`. Body: `arms` (defaults to every arm in `arms.yaml`), `sample_size`, `repeats`, `seed`. |
| `GET /runs/{run_id}` | Run status — derived from persisted results: `pending`, `running`, `completed`, or `completed_with_errors`, plus completed/failed/pending counts. |
| `GET /runs/{run_id}/results` | Per-call rows for a run (output text, latency, tokens, cost, error), paginated with `limit` and `offset`. |

Example:

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"sample_size": 20, "repeats": 3, "seed": 42}'
curl localhost:8000/runs/1
curl 'localhost:8000/runs/1/results?limit=50'
```

## Phase 3: Judge layer + calibration

Every successfully completed `RunResult` is automatically scored 1-5 by a
rubric-based LLM judge (configured via the separate `judge:` key in
`arms.yaml`, never as an eval arm). Judge scores land on `judge_score` /
`judge_rationale`; `judge_status` tracks `pending` / `completed` / `failed`
independently of the generation call's own `status`.

**Before trusting `judge_score` on a full run**, run the calibration
workflow — CLAUDE.md's differentiator is that judge calibration is
reported, not assumed:

### 1. Select a stratified sample to hand-label

```bash
uv run python -m scripts.select_calibration_sample --run-id 1 --n 40 --out calibration_sample.json
```

Stratifies by `(arm_name, gold_label)` so every arm and sentiment class is
represented. Open the file and fill in each row's `human_score` (1-5) by
hand.

### 2. Import your labels

```bash
uv run python -m scripts.import_calibration_labels --in calibration_sample.json --labeled-by you@example.com
```

Idempotent — re-running with updated scores upserts rather than duplicates.

### 3. Read the calibration report

```bash
uv run python -m scripts.calibration_report --run-id 1
```

Prints Spearman correlation and Cohen's kappa (score >= 4 treated as
"correct") between judge and human scores, plus mean absolute difference.

### Known limitations

- **Judge cost is not tracked.** The judge call is itself a billed API
  call, but its cost/latency/token counts are currently discarded — only
  the parsed `SCORE`/`RATIONALE` are persisted. Judging a run roughly
  doubles its API spend. A future phase should add judge-side
  cost/latency columns if the cost/latency/quality frontier needs to
  account for it.
- **Watch for judge/arm model overlap.** `arms.yaml`'s default `judge:`
  entry uses the same underlying model as the `claude-haiku` example eval
  arm. LLM-as-judge self-preference is a known validity risk — consider
  configuring a different (ideally stronger) model as the judge than any
  arm under comparison.

## Tests

```bash
uv run pytest -v
```

The Ollama end-to-end test skips automatically if Ollama isn't running
locally, and the database tests skip automatically if Postgres isn't
reachable.
