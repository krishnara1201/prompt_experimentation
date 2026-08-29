# LLM Evaluation & Prompt-Experimentation Platform

## What this is

A platform that treats LLM prompts and models as experiment arms and evaluates
them with the same statistical rigor as an A/B test — not just a leaderboard
score. Extends the experimentation/stats approach from `experimentation_copilot`
into the LLM space, and treats local and hosted models as interchangeable,
first-class arms in the same comparison.

## Why this project exists

Built to demonstrate LLM/GenAI engineering skills for ML engineering and data
science roles — specifically the parts most LLM portfolio projects skip:
paired statistical testing (not naive win-rates), judge calibration against
human labels, and an honest, documented cost/latency/quality tradeoff between
a local model and hosted API models.

## Core differentiators — do not lose these while building

1. **Paired comparisons, not independent samples.** Every arm sees the same
   prompts, so use a paired bootstrap or Wilcoxon signed-rank test on the
   per-example quality difference — not a two-sample t-test.
2. **Judge calibration is reported, not assumed.** Score a held-out,
   human-labeled subset with the LLM judge and report agreement (Cohen's
   kappa or correlation) *before* trusting judge scores on the full eval set.
3. **Bayesian equivalence test.** Answer "is the local model good enough" as
   P(quality_local ≥ quality_api − ε) for a chosen margin ε, reusing the
   Bayesian methodology already built in `experimentation_copilot`.
4. **Cost/latency/quality frontier, not a single number.** Report local
   compute cost/latency alongside API $/token — the deliverable is a tradeoff
   plot, not a leaderboard.
5. **Repeated runs per prompt per arm.** LLMs are non-deterministic — account
   for within-arm variance, not just between-arm differences.

## Architecture

```
[eval dataset] -> [model arms: local + API] -> [orchestrator + results store]
               -> [LLM-as-judge, calibrated] -> [stats analysis + dashboard]
```

- **Eval dataset** — a public financial sentiment benchmark (Financial
  PhraseBank or FiQA): labeled, sentence-level, expert-agreement sentiment.
  Avoids building/labeling a gold set from scratch.
- **Model arms** — four adapter implementations behind a shared `ModelAdapter`
  protocol, not one per provider: `OpenAICompatibleAdapter` (any provider
  speaking the OpenAI chat-completions schema — Ollama, OpenAI, OpenRouter,
  Groq, etc.), `AnthropicAdapter` (Claude's distinct schema), and two
  subscription-seat CLI adapters with no per-token price —
  `ClaudeCodeCLIAdapter` and `CodexCLIAdapter`, which drive the `claude`/
  `codex` CLIs directly under an already-authenticated Pro/Max or Plus/Pro
  seat instead of calling a metered API. Arms are declared in
  `backend/arms.yaml`, never hardcoded in code — model-agnostic,
  bring-your-own-key: adding or swapping a provider is a config edit. The
  first two were built in Phase 1 (see Build phases below); design doc at
  `docs/superpowers/specs/2026-08-25-model-adapter-layer-design.md`. The two
  subscription-CLI adapters were added afterward; design doc at
  `docs/superpowers/specs/2026-08-27-subscription-cli-adapters-design.md`.
  - Local: Ollama serving **Qwen3-8B** by default (confirmed: <16GB VRAM GPU).
  - API: model-agnostic via config — `arms.yaml` currently has example arms
    for GPT-4o-mini and Claude Haiku, but any OpenAI-schema or Anthropic
    provider works without code changes.
- **Orchestrator** — Celery + Redis, runs (eval set) × (arms) × (N repeats)
  as async jobs. Reuse the async job pattern from `experimentation_copilot`.
- **Results store** — Postgres: raw prompts, outputs, judge scores, latency,
  token counts, per-call cost estimate.
- **Judge layer** — LLM-as-judge with a fixed rubric prompt, calibrated
  against a hand-labeled gold subset before being trusted on the full run.
- **Stats layer** — paired significance tests (bootstrap CI, Wilcoxon
  signed-rank), Bayesian posterior comparison between arms, multiple-
  comparison correction across arm pairs, sample-size/power calculator.
- **Dashboard** — React: win-rate table with confidence intervals,
  cost/latency/quality frontier scatter, judge calibration report. Runs
  can be started from a "New run" form (backed by `GET /arms` +
  `POST /runs`), not only via `curl`.
- **`pe` CLI** — `backend/app/cli/`, console entrypoint (`uv run pe …`
  from `backend/`). One command over the whole loop: `docker compose`
  lifecycle (`up`/`down`/`logs`/`seed`), runs (`run`/`status`/`watch`/
  `results`/`arms`) and stats (`stats compare|equivalence|power`) over
  HTTP, and the host-side calibration scripts (`calibrate
  select|import|report`). `scripts/demo.sh` chains it end to end. Spec:
  `docs/superpowers/specs/2026-08-29-cli-and-dashboard-run-design.md`.

## Tech stack

- Backend: Python, FastAPI, SQLModel
- Async jobs: Celery, Redis
- Database: PostgreSQL, Alembic migrations
- Local model serving: Ollama (OpenAI-compatible endpoint at
  `localhost:11434/v1`); consider vLLM later for batching/throughput
- API models: model-agnostic, bring-your-own-key — any OpenAI-schema
  provider (OpenAI, OpenRouter, Groq, etc.) plus Anthropic (Claude), declared
  in `backend/arms.yaml`. Keys read from env vars named per-arm in that
  config; `backend/.env.example` documents the ones the example arms use.
  Package manager: `uv` (matches `experimentation_copilot`).
- Frontend: TypeScript, React, Vite
- Containerization: Docker
- Stats: scipy/statsmodels for frequentist tests; match whatever Bayesian
  approach (or library) `experimentation_copilot` already uses rather than
  introducing a second methodology

## Data

- Primary candidate: Financial PhraseBank (public, sentence-level financial
  sentiment, expert-agreement labels) or the FiQA sentiment task.
- Gold subset for judge calibration: a stratified sample (~30-50 examples)
  with human labels held back from judge-only scoring, used purely to check
  judge agreement.

## Build phases

1. **Model adapter layer** ✅ **Done.** Unified local (Ollama) and API models
   behind one interface (`backend/app/adapters/`), config-driven via
   `backend/arms.yaml` (`backend/app/config/arms.py`). Demo script
   (`backend/app/demo.py`) runs a handful of prompts through every configured
   arm end to end. Spec: `docs/superpowers/specs/2026-08-25-model-adapter-layer-design.md`;
   plan: `docs/superpowers/plans/2026-08-25-model-adapter-layer.md`. Extended
   afterward with two subscription-seat CLI adapters (`ClaudeCodeCLIAdapter`,
   `CodexCLIAdapter`) so Claude Code/Codex CLI arms can run under a
   subscription instead of a metered API key, routed to a dedicated
   low-concurrency Celery queue. Spec:
   `docs/superpowers/specs/2026-08-27-subscription-cli-adapters-design.md`;
   plan: `docs/superpowers/plans/2026-08-27-subscription-cli-adapters.md`.
2. **Orchestration** ✅ **Done.** Celery/Redis run eval set × arms × repeats;
   raw outputs persisted to Postgres. FastAPI run endpoints (create, status,
   results); idempotent seed script for eval examples.
3. **Judge layer + calibration** ✅ **Done.** Rubric-based LLM-as-judge
   (`backend/app/judge/`) auto-scores every completed `RunResult` via a
   chained Celery task. Calibration workflow
   (`backend/scripts/select_calibration_sample.py`,
   `import_calibration_labels.py`, `calibration_report.py`) reports
   Spearman correlation and Cohen's kappa between judge and human scores
   before judge scores are trusted on a full run. Spec:
   `docs/superpowers/specs/2026-08-27-judge-layer-calibration-design.md`.
4. **Stats layer** ✅ **Done.** `backend/app/stats/` — hierarchical paired
   bootstrap + Wilcoxon signed-rank + Holm-Bonferroni correction
   (`paired_tests.py`), PyMC-based Bayesian equivalence test
   (`bayesian.py`, restricted to `judge_score` — direction-sensitive for
   the other metrics), closed-form sample-size/power calculator
   (`power.py`), and `RunResult` aggregation (`aggregation.py`). Exposed
   via `GET /runs/{run_id}/compare|equivalence|power`
   (`backend/app/api/routes/stats.py`). Spec:
   `docs/superpowers/specs/2026-08-27-stats-layer-design.md`; plan:
   `docs/superpowers/plans/2026-08-27-stats-layer.md`.
5. **Dashboard** ✅ **Done.** React app (`frontend/`) — run list with live
   status polling, and a per-run tabbed view: win-rate table (pairwise
   quality diff + CI + corrected p-value), cost/latency/quality frontier
   scatter, and judge calibration report. Backed by three new read-only
   endpoints (`GET /runs`, `/runs/{run_id}/summary`,
   `/runs/{run_id}/calibration`). Spec:
   `docs/superpowers/specs/2026-08-27-dashboard-design.md`; plan:
   `docs/superpowers/plans/2026-08-27-dashboard.md`.
6. **Agent-facing judge tool** ✅ **Done.** MCP server
   (`backend/app/mcp_judge_server.py`) exposes a single
   `score_financial_sentiment` tool wrapping the existing
   `judge/scorer.py:score_output`, so a Claude Code session (or any other
   MCP client) can score a candidate response against a gold label
   directly, without running a full eval. Discovered automatically via the
   repo-root `.mcp.json`. The tool response also carries `judge_model` (call
   provenance, since the `judge:` config reloads per call), and rejects
   blank `input_text`/`model_output` or an out-of-domain `gold_label` before
   making a judge call. `backend/scripts/judge_tool_dryrun.py` exercises the
   whole path (real MCP stdio transport, real dataset rows, real arm output)
   as a smoke test. Spec:
   `docs/superpowers/specs/2026-08-29-agent-facing-judge-tool-design.md`
   (see the 2026-08-29 post-implementation amendment).
7. **Local fine-tune** 🚧 **Capability built; comparison run pending a GPU
   session.** `backend/app/training/` — QLoRA fine-tune of Qwen3-8B on the
   Financial PhraseBank *lower-agreement* subset (disjoint from the
   all-agree eval set, enforced by a leakage guard in
   `training/dataset.py`), then merge → GGUF → `ollama create` so the
   fine-tuned model is just another `openai_compatible` arm. Driven by
   `pe finetune prep|train|export|report` over `backend/training.yaml`.
   Training deps are an optional `training` extra, out of the core/CI
   path. Spec:
   `docs/superpowers/specs/2026-08-29-local-finetune-phase7-design.md`;
   plan: `docs/superpowers/plans/2026-08-29-local-finetune-phase7.md`.
   The executed fine-tuned-vs-base-vs-API comparison lands at
   `docs/superpowers/reports/2026-08-29-finetune-comparison.md`.

## Open decisions

- ~~Confirm hardware~~ **Resolved:** GPU with <16GB VRAM. Local model is
  Qwen3-8B via Ollama (Phase 1, done).
- ~~Confirm API arm approach~~ **Resolved:** model-agnostic, bring-your-own-key
  via config (Phase 1, done) rather than hardcoding Claude Haiku/GPT-4o-mini
  as fixed arms.
- ~~Confirm final task dataset (Financial PhraseBank vs. FiQA)~~ **Resolved:**
  Financial PhraseBank (Malo et al. 2014), 3-class sentence sentiment with
  expert-agreement labels. Licensed **CC BY-NC-SA 3.0** (verified on the
  `takala/financial_phrasebank` HF card) — non-commercial, share-alike,
  attribution. This repo is open source (permissive code license) but the
  dataset is **not vendored**: the seed script downloads it at runtime via
  `load_dataset("takala/financial_phrasebank", ...)`, so the dataset's NC
  terms bind the user who downloads it, not the repo. README must attribute
  Malo et al. 2014 and state the CC BY-NC-SA 3.0 / non-commercial
  restriction. The calibration gold subset stores row IDs + human labels
  (not redistributed source text) to avoid redistributing the licensed
  corpus. Keep the dataset a config choice so a commercial user can swap in
  a permissively-licensed sentiment set.
- ~~Confirm Bayesian library/approach~~ **Resolved:** PyMC (Phase 4, done).
  `experimentation_copilot/backend/app/stats/stat_analysis.py` was purely
  frequentist with no existing Bayesian code to reuse, so this was a fresh
  pick rather than a copy — a PyMC paired-difference model
  (`backend/app/stats/bayesian.py`), deliberately diverging from
  `experimentation_copilot`.

## Non-goals

- Not a general-purpose LLM chat UI.
- Not fine-tuning the API models — only the local model is a fine-tuning
  candidate.
- Not optimizing for maximum absolute model quality — optimizing for a
  rigorous, honest comparison between arms.
