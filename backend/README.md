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

These two arms ship commented out in `arms.yaml`, so `POST /runs` with no
explicit `arms` filter (which defaults to every arm in the file) won't
include them. Uncomment them only after the preconditions below are met —
an authenticated CLI session and a running `subscription_cli` worker —
otherwise their tasks queue forever with nothing consuming them.

**Precondition**: the machine running the Celery worker must already have
an authenticated CLI session under your subscription — run `claude login`
or `codex login` yourself first. Neither adapter reads or stores
credentials; they only shell out to whatever session already exists.

**Tool use stays on.** Each call still runs from a fresh, empty scratch
directory (created and torn down per call), so neither CLI has this repo's
`CLAUDE.md`/`AGENTS.md` to discover by default, and there's nothing real in
that directory for a tool call to touch.

This is a context boundary, not a security sandbox, and the two CLIs differ
here: Codex's `--sandbox workspace-write` is a real OS-level sandbox scoped
to the scratch directory. Claude Code's `--dangerously-skip-permissions`
only disables the interactive approval prompt — it installs no OS sandbox,
so its Bash/Write tools can reach anything the worker process's user
account can reach, regardless of `cwd`. Only run the Claude Code arm on a
host where that's an acceptable risk. A user-global `~/.claude/CLAUDE.md`,
if the operator running the worker has one, also still loads regardless of
`cwd` — point `CLAUDE_CONFIG_DIR` at a throwaway directory first if you
need to rule that out too.

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

`GET /arms` lists the arms configured in `arms.yaml` (name / adapter type /
model) — used by the `pe` CLI and the dashboard's New Run form.

### The `pe` CLI

`pe` (console entrypoint, `uv run pe …` from `backend/`) wraps all of the
above plus the stack lifecycle and the calibration workflow. The raw
`curl` / `python -m scripts.*` commands still work — `pe` is the friendlier
path.

| Command | What it does |
| --- | --- |
| `pe up [--no-wait]` / `pe down [-v]` / `pe logs [svc] [-f]` | `docker compose` lifecycle |
| `pe seed` | seed the eval dataset via a one-off `migrate` container |
| `pe arms` | list configured arms |
| `pe run [--sample N] [--repeats N] [--seed N] [--arm A ...] [-q]` | `POST /runs`; `-q` prints only the run id |
| `pe status RUN_ID` / `pe watch RUN_ID` / `pe results RUN_ID` | run status, poll-to-done, per-call rows |
| `pe stats compare RUN_ID [-m METRIC]` | pairwise paired comparison |
| `pe stats equivalence RUN_ID --local A --api B [--eps E]` | Bayesian equivalence (judge_score) |
| `pe stats power RUN_ID --arm-a A --arm-b B` | required sample size for the observed effect |
| `pe calibrate select \| import \| report …` | the Phase 3 calibration scripts (run on the host — need a local `.env` with a `localhost` `DATABASE_URL`) |

`PE_API_URL` overrides the API base (default `http://localhost:8000`).
`uv run pe --help` documents every command.

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

## Phase 6: Agent-facing judge tool

Exposes the judge layer (Phase 3) as an MCP tool so a coding agent (e.g. a
Claude Code session iterating on a prompt) can score one candidate
financial-sentiment response against a gold label directly — no eval run,
no Celery, no Postgres. It calls the same rubric and `judge:` config as the
automated pipeline.

### 1. Make the tool available

For **Claude Code opened in this repo**, nothing to do — the repo-root
`.mcp.json` registers the server automatically (approve it at the
project-scoped-server prompt the first time). Check it loaded with
`/mcp`; the tool appears as
`mcp__financial-sentiment-judge__score_financial_sentiment`.

For **Codex CLI**, the repo-root `.mcp.json` is not picked up (that's a
Claude Code convention) — register the server in `~/.codex/config.toml`
with an **absolute** path to `backend`:

```toml
[mcp_servers.financial-sentiment-judge]
command = "uv"
args = ["run", "--directory", "/abs/path/to/prompt_experimentation/backend", "python", "-m", "app.mcp_judge_server"]
```

Codex launches the server from its own working directory, so the relative
`--directory backend` that works for Claude Code must be made absolute here.

For **any other MCP client**, run the server over stdio:

```bash
uv run --directory backend python -m app.mcp_judge_server
```

and point the client's own MCP config at that command, again using an
absolute path if the client starts outside the repo root.

**Prerequisite — the judge model must be reachable.** The server shells out
to whatever `arms.yaml`'s `judge:` block names (see step 3), regardless of
which client called the tool. The default `adapter: claude_code_cli` needs
an authenticated `claude` CLI on `PATH` (`claude login`); an
`openai_compatible` / `anthropic` judge needs its API key in `backend/.env`.
Calling the tool from a Codex session does **not** provide a `claude` CLI —
if you don't have one, switch the judge to `openai_compatible` +
`OPENAI_API_KEY`, the `codex_cli` adapter, or local Ollama (step 3). A
misconfigured judge comes back as a tool error to the caller, not a server
crash.

### 2. Call the tool

```
score_financial_sentiment(
    input_text:   str,   # the sentence that was classified
    gold_label:   str,   # "positive" | "negative" | "neutral"
    model_output: str,   # the candidate response being graded
) -> {"score": 1-5, "rationale": str, "judge_model": str}
```

Example call and response:

```jsonc
// request
{
  "input_text": "Shares tumbled 12% after the firm slashed its full-year guidance.",
  "gold_label": "negative",
  "model_output": "Negative — a guidance cut and a 12% drop signal a deteriorating outlook."
}
// response
{
  "score": 5,
  "rationale": "The response correctly identifies the sentiment as negative, states it directly, and grounds it in the guidance cut and share price drop.",
  "judge_model": "opus"
}
```

The 1-5 scale (reference-guided against `gold_label`):

| Score | Meaning |
|-------|---------|
| 5 | Correct sentiment, stated clearly and directly |
| 4 | Correct sentiment, minor clarity/formatting issues |
| 3 | Ambiguous, hedged, or only partially matches |
| 2 | Wrong sentiment, but otherwise coherent and on-topic |
| 1 | Wrong sentiment, off-topic, malformed, or non-responsive |

`judge_model` echoes back which model produced the score — useful because
the `judge:` config is reloaded per call and may change between calls.

### 3. Choose the judge model

Edit the top-level `judge:` block in `backend/arms.yaml`. It takes the same
fields as an eval arm; `adapter` and `model` are required. Changes take
effect on the **next call** — no server restart.

```yaml
# Claude via an authenticated CLI seat (default — no API key, no per-token cost)
judge:
  adapter: claude_code_cli
  model: opus

# Hosted OpenAI-schema provider (OpenAI, OpenRouter, Groq, …)
judge:
  adapter: openai_compatible
  base_url: https://api.openai.com/v1
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
  max_tokens: 1024

# Claude via the metered Anthropic API
judge:
  adapter: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY

# Local Ollama — free, offline, no key
judge:
  adapter: openai_compatible
  base_url: http://localhost:11434/v1
  model: qwen3:8b
  max_tokens: 1024
```

Two things to watch when switching:

- **Don't reuse a model that's also an eval arm** — LLM-as-judge
  self-preference is a real validity risk (see "Watch for judge/arm model
  overlap" above).
- **Re-run calibration.** The Spearman/kappa figures from
  `scripts/calibration_report.py` are specific to one judge; a new judge is
  uncalibrated until you re-run that workflow.

### 4. Interpreting a score

A single ad-hoc score is a quick signal, not a verdict. The tool response
is deliberately silent on calibration — before relying on judge scores at
scale, run the Phase 3 calibration workflow (`select_calibration_sample` →
`import_calibration_labels` → `calibration_report`) so you know how well
this judge agrees with human labels.

Ad-hoc calls are **not persisted** — this is for disposable checks during
iteration, not part of the auditable `RunResult` / judge-score history.

### Example session

Spot-checking three candidate outputs from a sentiment classifier while
iterating on its prompt (real responses from the default `opus` judge):

```
> score_financial_sentiment(
    input_text="Nokia's Q3 net sales rose 9% year-on-year to EUR 5.7bn, beating estimates.",
    gold_label="positive",
    model_output="This is positive sentiment. Sales grew 9% and exceeded analyst expectations.")

  { "score": 5,
    "rationale": "The response correctly and directly identifies the sentiment as
                  positive, citing the 9% sales growth and the earnings beat as
                  justification.",
    "judge_model": "opus" }

> score_financial_sentiment(
    input_text="The company said it will cut 1,200 jobs and close two factories as demand slumps.",
    gold_label="negative",
    model_output="Positive — restructuring will make the company leaner and more efficient.")

  { "score": 2,
    "rationale": "The response is coherent and on-topic but assigns positive
                  sentiment to clearly negative news about layoffs, factory
                  closures, and slumping demand.",
    "judge_model": "opus" }

> score_financial_sentiment(
    input_text="The company will hold its annual general meeting on 25 March in Helsinki.",
    gold_label="neutral",
    model_output="Neutral. This is a factual scheduling announcement with no financial implication.")

  { "score": 5,
    "rationale": "The model correctly identifies the sentiment as neutral and gives
                  a clear, concise justification that matches the factual,
                  non-financial nature of the announcement.",
    "judge_model": "opus" }
```

The middle case is the useful signal: the classifier confidently inverted a
layoffs headline, and the judge caught it with a 2 (wrong sentiment, but
coherent) rather than a 1.

### Dry run

`scripts/judge_tool_dryrun.py` exercises the whole path once — it spawns the
real MCP server over stdio (exactly as a client would), lists its tools,
scores a stratified handful of real Financial PhraseBank sentences, and
confirms the validation rejects. Nothing is written to Postgres.

```bash
# candidate answers generated by a configured eval arm (dataset -> model -> judge)
uv run python -m scripts.judge_tool_dryrun --arm qwen3-8b-local --n 6

# judge only — fabricate the candidate answers, no eval model needed
uv run python -m scripts.judge_tool_dryrun --synthetic --n 6
```

It prints each sentence, the candidate answer, and the judge's
`score` / `rationale` / `judge_model`, then a score distribution and a
`failures:` count. Exit code is non-zero if the tool is missing from the
server, a valid call errors, or a call that should have been rejected
succeeds — so it works as a smoke test after changing the `judge:` config
or the adapters.

### Errors

- `gold_label` outside `{"positive", "negative", "neutral"}`, or a blank
  `input_text` / `model_output`, is rejected before any judge call.
- A malformed judge response (`JudgeParseError`) or an adapter failure
  (missing key, network error, unauthenticated CLI) propagates as an MCP
  tool error — no retries; the calling agent decides whether to retry or
  rephrase.

## Tests

```bash
uv run pytest -v
```

The Ollama end-to-end test skips automatically if Ollama isn't running
locally, and the database tests skip automatically if Postgres isn't
reachable.

`.github/workflows/ci.yml` runs this suite on every push and PR against a
throwaway Postgres + Redis (migrations applied with `alembic upgrade head`
first), alongside a frontend `lint` + `build` job. The subscription-CLI and
Ollama e2e tests skip in CI since those binaries/services aren't present.

Where a database test holds ORM objects across commits and reads their
columns after `asyncio.run()` returns, its session is created with
`expire_on_commit=False` — the default would turn that read into a lazy
reload outside the async greenlet and raise `MissingGreenlet`.
