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
`ollama pull qwen3:8b`), the same as a non-Docker setup, and it must be
listening on `0.0.0.0`, not just `127.0.0.1` — an OS-managed Ollama
(systemd, launchd) usually defaults to loopback-only, which containers
can't reach regardless of the fix below. Set `OLLAMA_HOST=0.0.0.0:11434`
in its environment and restart it (on a systemd host: a
`/etc/systemd/system/ollama.service.d/override.conf` drop-in with
`Environment="OLLAMA_HOST=0.0.0.0:11434"`, then
`systemctl daemon-reload && systemctl restart ollama`).

`backend/arms.yaml`'s `qwen3-8b-local` entry ships pointed at
`http://localhost:11434/v1` for the non-Docker default flow (see
`backend/README.md`); that doesn't resolve to the host from inside a
container, so it needs to change for Docker use. Try
`http://host.docker.internal:11434/v1` first — it works on most Docker
Desktop setups. If calls still fail with `Connection refused` (seen on at
least one Docker Desktop + WSL2 setup, where `host.docker.internal`
reaches a gateway that refuses the connection even though Windows' own
`localhost:11434` forwarding works), fall back to the IP of the WSL
distro's `eth0` interface instead (`ip addr show eth0 | grep inet`) — that
address isn't guaranteed stable across a `wsl --shutdown`/reboot, so
re-check it if connectivity breaks later.

Don't commit whichever address you land on — `localhost` is correct for
the non-Docker flow and for anyone else's Docker setup, so a machine- or
platform-specific override belongs in your local working tree only.

`arms.yaml` is bind-mounted read-only into `api`/`worker`, so editing it
doesn't need an image rebuild — but on Docker Desktop + WSL2, editing the
file *while the containers are running* can leave them holding onto the
old content: many editors and CLI tools replace a file by writing a new
one and renaming over it, and the bind mount can end up pinned to the old,
now-unlinked inode instead of the current filename. If a container's
`/app/arms.yaml` doesn't reflect a change you made, don't rely on
`docker compose restart` — recreate it instead:

```bash
docker compose up -d --force-recreate api worker
```

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

## Agent-facing judge tool

The repo-root `.mcp.json` registers an MCP tool, `score_financial_sentiment`,
for any Claude Code session opened in this repo (standard project-scoped-
server approval prompt applies) — it lets a coding agent score one
candidate financial-sentiment response against a gold label directly,
without running a full eval. See "Phase 6" in `backend/README.md` for the
tool signature and how to use it from another MCP client.

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
