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

- **Eval dataset** — one *task pack* among others (Phase 8). A task is a
  directory under `backend/tasks/<name>/`: a `task.yaml` (label set, default
  eval prompt, judge rubric, data-file pointer) plus a data file (vendored
  `.txt`/`.jsonl`). `arms.yaml`'s top-level `task:` key selects the active
  one; the default `financial_sentiment` pack is byte-identical to the
  former hardcoded behaviour — a public financial sentiment benchmark
  (Financial PhraseBank), labeled, sentence-level, expert-agreement. Bring
  your own eval = drop in a JSONL file + a `task.yaml`, no code change.
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
  - **Prompts are arms too.** An arm carries an optional `prompt_template`
    (must contain `{text}`; when unset it falls back to the active task's
    `eval_prompt`, itself defaulting to `app/eval_prompt.py`). Two arms with
    the same model but different templates A/B the prompts — the paired
    stats apply unchanged. `config/arms.py` wraps each adapter in an `Arm`
    (`name`, `adapter`, `prompt_template`); `GET /arms` reports the resolved
    template.
- **Orchestrator** — Celery + Redis, runs (eval set) × (arms) × (N repeats)
  as async jobs. Reuse the async job pattern from `experimentation_copilot`.
- **Results store** — Postgres: raw prompts, outputs, judge scores, latency,
  token counts, per-call cost estimate.
- **Judge layer** — LLM-as-judge with a fixed rubric prompt (now supplied
  per-task by the active task pack; still a fixed integer 1–5 score, so the
  stats/calibration layers are unchanged), calibrated against a hand-labeled
  gold subset before being trusted on the full run.
- **Stats layer** — paired significance tests (bootstrap CI, Wilcoxon
  signed-rank), Bayesian posterior comparison between arms, multiple-
  comparison correction across arm pairs, sample-size/power calculator.
- **Dashboard** — React: win-rate table with confidence intervals,
  cost/latency/quality frontier scatter, judge calibration report. Runs
  can be started from a "New run" form (backed by `GET /arms` +
  `POST /runs`), not only via `curl`.
- **`pe` CLI** — `backend/app/cli/`, console entrypoint (`uv run pe …`
  from `backend/`). One command over the whole loop: `docker compose`
  lifecycle (`up`/`down`/`logs`/`seed [--task <name>]`), task packs
  (`tasks` — list packs, active flag, seeded counts), runs
  (`run`/`status`/`watch`/`results`/`arms`) and stats
  (`stats compare|equivalence|power`) over HTTP, and the host-side
  calibration scripts (`calibrate select|import|report`). `scripts/demo.sh`
  chains it end to end. Spec:
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
7. **Local fine-tune** ✅ **Done (base-vs-fine-tune + local-vs-hosted
   executed; a metered per-token API arm still pending a paid key).**
   `backend/app/training/` — QLoRA fine-tune of Qwen3-8B on the
   Financial PhraseBank *lower-agreement* subset (disjoint from the
   all-agree eval set, enforced by a leakage guard in
   `training/dataset.py`), then merge → GGUF → `ollama create` so the
   fine-tuned model is just another `openai_compatible` arm. Driven by
   `pe finetune prep|train|export|report` over `backend/training.yaml`.
   Training deps are an optional `training` extra, out of the core/CI
   path. Spec:
   `docs/superpowers/specs/2026-08-29-local-finetune-phase7-design.md`;
   plan: `docs/superpowers/plans/2026-08-29-local-finetune-phase7.md`.
   The executed comparison (Deliverable 1) lands at
   `docs/superpowers/reports/2026-08-30-finetune-comparison.md` (run 1914; the
   thinner run-1731 writeup is archived under `reports/archive/`): with the full
   judge sample the fine-tune is
   a *significant* quality win over the base (90% vs 80% accuracy, paired
   Wilcoxon corrected p=0.031, Bayesian posterior +0.30 clearing zero) and
   ~1.8× faster / ~49× fewer output tokens.

   **Local-vs-hosted leg — DONE via a subscription-seat arm** (run 1058,
   `docs/superpowers/reports/2026-09-01-local-vs-cli-hosted.md`):
   `qwen3-8b-local` vs. `claude-code-sonnet` (the `claude` CLI under a Max
   seat, no per-token bill), 150 Financial PhraseBank sentences, 300 calls,
   0 failures. Quality is **not significantly different** (paired Wilcoxon
   corrected p=0.10; 87.3% vs 92.0% raw accuracy; Bayesian P(within ±0.5)
   =1.00, but underpowered — achieved power 0.39, ~414 examples needed for a
   tight claim). Median latency comparable (3.8s vs 2.4s) but the local arm
   had a severe tail (max 3.4h) under memory pressure on the 7.8GB box. Both
   costs `null`. Run via `backend/scripts/serial_eval_run.py`, an in-process
   runner that pauses on the Claude seat's usage limit and resumes after it
   resets (`--max-cli-calls` batches the CLI phase). A *metered* per-token
   API arm (real $/token on the frontier x-axis) still needs a paid key —
   `gemini-flash` free tier 429'd 130/150.

8. **Task-agnostic eval** ✅ **Done.** The eval loop is no longer hardwired
   to financial sentiment: a task is a pack under `backend/tasks/<name>/`
   (`task.yaml` + a `.txt`/`.jsonl` data file) declaring the label set, the
   default eval prompt, and the judge rubric (`backend/app/config/tasks.py`
   loads/validates it; `backend/app/data/loader.py` reads JSONL packs).
   `arms.yaml`'s `task:` key selects the active pack (default
   `financial_sentiment`, byte-identical to the old behaviour); `Run.task`
   (migration `0003`) records it per run and it is threaded through the
   worker's call + judge tasks. `POST /runs` takes an optional `task`,
   samples only that task's seeded examples, and 422s on an unknown one;
   `GET /tasks`, `pe tasks`, and `pe seed --task` expose packs; the New Run
   form has a task dropdown. The MCP judge server (now `rubric-judge`, tool
   `score_output_against_gold`) and the QLoRA training dataset builder both
   read the active/configured task. Score scale stays a fixed integer 1–5
   so the stats and calibration layers are untouched. Spec:
   `docs/superpowers/specs/2026-08-30-task-agnostic-eval-design.md`; plan:
   `docs/superpowers/plans/2026-08-30-task-agnostic-eval.md`. The AG News
   pack (`backend/tasks/ag_news/`, 4-class topic classification) is the
   second pack. **Deliverable 2 executed** (run 559,
   `docs/superpowers/reports/2026-08-31-prompt-ab-comparison.md`): a prompt
   A/B on AG News of a terse "just the label" instruction vs. a "reason step
   by step" one (`ag-news-terse` / `ag-news-cot` in `arms.yaml`, same
   `qwen3:8b`, `reasoning_effort: none`). Result — **quality is a wash**
   (77.5% vs 79.1% raw, paired Wilcoxon corrected p=0.37, Bayesian posterior
   +0.06 with 94% CI inside ±0.2, P(equivalent)=1.00 at ε=0.5) while CoT
   costs **~40× the output tokens** (p≈2e-21) and ~6× the uncontended
   latency; the paired test resolves a 1.6-pt gap an unpaired win-rate would
   misread. Judge calibration for AG News: κ=1.00 vs. hand labels (n=32).
   Enabled by two small additions: `OpenAICompatibleAdapter` `extra_body`
   passthrough and a `pe run --task` flag.

   **Judge calibration on financial sentiment with free-text outputs** (run
   707, `docs/superpowers/reports/2026-08-31-judge-calibration-financial.md`):
   a `qwen3-fin-explain` arm prompted for prose rationales, 50 rows
   hand-labeled blind to the judge. κ=1.00 on label-correctness (0 label
   disagreements / 50), but Spearman only 0.729 — the judge emits a
   2-value scale (5 or 2), never 3/4, so it agrees on *whether* the label is
   right but is not a calibrated 1–5 scorer. `scripts/serial_judge_run.py`
   added as a low-RAM in-process judge fallback (no Celery/Redis).
   `docs/RESULTS.md` is the external walk-through of all four experiments
   (fine-tune, prompt A/B, judge calibration, local-vs-hosted).

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
  attribution. The 100%-agreement subset (2264 sentences) **is vendored**
  at `backend/data/financial_phrasebank/sentences_allagree.txt` (with a
  license header); `backend/scripts/fetch_financial_phrasebank.py`
  regenerates it from the `gtfintechlab/financial_phrasebank_sentences_allagree`
  HF mirror, and the seed script reads the local file. So the repo does
  redistribute the corpus — CC BY-NC-SA 3.0 permits that with attribution,
  non-commercially, share-alike, and those terms bind the bundled copy and
  anything derived from it. The root README's "Data & license" section
  carries the Malo et al. 2014 citation and the CC BY-NC-SA 3.0 /
  non-commercial notice. The calibration gold subset stores row IDs +
  human labels (not redistributed source text). The dataset swap-in is now
  concrete (Phase 8): it is a task pack, not just an aspiration — a
  commercial user substitutes a permissively-licensed set by adding a
  `task.yaml` + JSONL under `backend/tasks/` and pointing `arms.yaml`
  `task:` at it, no code change.
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
