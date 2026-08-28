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
  cost/latency/quality frontier scatter, judge calibration report.

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
6. **Agent-facing judge tool** — expose the judge layer (Phase 3) as a
   callable tool so local coding agents (e.g. Claude Code sessions) can use
   this platform's calibrated rubric judge directly, independent of the
   eval-run pipeline. Deliberately designed after Phase 3 lands, against its
   actual interface rather than a guess at one.
7. **Stretch** — LoRA fine-tune the local model on a subset of the task;
   compare fine-tuned local vs. base local vs. API arms.

## Open decisions

- ~~Confirm hardware~~ **Resolved:** GPU with <16GB VRAM. Local model is
  Qwen3-8B via Ollama (Phase 1, done).
- ~~Confirm API arm approach~~ **Resolved:** model-agnostic, bring-your-own-key
  via config (Phase 1, done) rather than hardcoding Claude Haiku/GPT-4o-mini
  as fixed arms.
- Confirm final task dataset (Financial PhraseBank vs. FiQA) once
  licensing/format is checked.
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
