# LLM Evaluation & Prompt-Experimentation Platform

A platform that treats LLM prompts and models as experiment arms and
evaluates them with the same statistical rigor as an A/B test — paired
significance testing, judge calibration against human labels, and a
Bayesian equivalence test, not just a leaderboard score. Local (Ollama) and
hosted API models are interchangeable, first-class arms in the same
comparison, config-driven via `backend/arms.yaml`.

See `CLAUDE.md` for the full write-up: motivation, core differentiators,
architecture, and build-phase history.

## Architecture

```
[eval dataset] -> [model arms: local + API] -> [orchestrator + results store]
               -> [LLM-as-judge, calibrated] -> [stats analysis + dashboard]
```

- **Backend** — FastAPI + Celery/Redis orchestration, Postgres results
  store, Alembic migrations (`backend/`)
- **Model arms** — `OpenAICompatibleAdapter` (Ollama, OpenAI, OpenRouter,
  Groq, ...), `AnthropicAdapter`, plus two subscription-seat CLI adapters
  (`ClaudeCodeCLIAdapter`, `CodexCLIAdapter`) — declared in
  `backend/arms.yaml`, no code changes to add or swap a provider
- **Judge layer** — rubric-based LLM-as-judge, calibrated against a
  hand-labeled gold subset before scores are trusted
- **Stats layer** — paired bootstrap + Wilcoxon signed-rank +
  Holm-Bonferroni correction, PyMC Bayesian equivalence test, sample-size
  calculator (`backend/app/stats/`)
- **Dashboard** — React (`frontend/`): win-rate table, cost/latency/quality
  frontier, judge calibration report

## Run with Docker

Requires [Docker](https://www.docker.com/) and Docker Compose.

```bash
cp .env.example .env   # set POSTGRES_PASSWORD; fill in API keys you have
docker compose up --build
```

This brings up Postgres, Redis, a migration job, the FastAPI backend, the
Celery worker, and the dashboard frontend (nginx, serving the production
build and proxying `/api` to the backend). Once healthy:

- Dashboard: http://localhost:5173
- API: http://localhost:8000

Seed the eval dataset once (idempotent, safe to re-run):

```bash
docker compose run --rm migrate uv run python -m scripts.seed_eval_examples
```

Then kick off a run:

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"sample_size": 20, "repeats": 3, "seed": 42}'
```

**Local model (Ollama) arm** — Ollama itself is not containerized; it's
expected to already be running natively on the host (`ollama serve`,
`ollama pull qwen3:8b`), the same as a non-Docker setup. The `api` and
`worker` containers can reach it at `host.docker.internal:11434`, but
`backend/arms.yaml`'s `qwen3-8b-local` entry ships pointed at
`http://localhost:11434/v1` for the non-Docker default flow (see
`backend/README.md`). When running via `docker compose`, change that arm's
`base_url` to `http://host.docker.internal:11434/v1` — `arms.yaml` is
bind-mounted into the containers read-only, so the edit takes effect on
container restart, no rebuild needed.

**Subscription-seat CLI arms (Claude Code, Codex)** — also not
containerized. They drive an already-authenticated `claude`/`codex` CLI
session on the host, and mounting that host credential/session state into a
container isn't worth the security surface for what's already an opt-in
arm type. Run their dedicated `subscription_cli` worker natively on the
host instead; it can reach the Dockerized Postgres/Redis via the ports
published to `127.0.0.1`. See "Subscription-seat CLI arms" in
`backend/README.md`.

## Run without Docker

See `backend/README.md` and `frontend/README.md` for native setup
(`uv sync` / `npm install`, Ollama, migrations, `npm run dev`).

## Tech stack

Python, FastAPI, SQLModel, Celery, Redis, PostgreSQL, Alembic, Ollama,
scipy/statsmodels, PyMC, TypeScript, React, Vite, Docker.

## Tests

```bash
cd backend && uv run pytest -v
```

The Ollama end-to-end test skips automatically if Ollama isn't running
locally, and the database tests skip automatically if Postgres isn't
reachable.
