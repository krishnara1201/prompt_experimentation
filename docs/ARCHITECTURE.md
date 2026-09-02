# Architecture

## What this is

A platform that treats LLM prompts and models as experiment arms and
evaluates them with the same statistical rigor as an A/B test — not just a
leaderboard score. Local and hosted models are interchangeable, first-class
arms in the same comparison.

## What makes it different

1. **Paired comparisons, not independent samples.** Every arm sees the same
   prompts, so the analysis uses a paired bootstrap / Wilcoxon signed-rank
   test on the per-example quality difference — not a two-sample t-test.
2. **Judge calibration is reported, not assumed.** A held-out, human-labeled
   subset is scored with the LLM judge and agreement (Cohen's kappa /
   correlation) is reported before judge scores are trusted on the full
   eval set.
3. **Bayesian equivalence test.** "Is the local model good enough" is
   answered as P(quality_local ≥ quality_api − ε) for a chosen margin ε.
4. **Cost/latency/quality frontier, not a single number.** Local compute
   cost/latency is reported alongside API $/token — the deliverable is a
   tradeoff plot, not a leaderboard.
5. **Repeated runs per prompt per arm.** LLMs are non-deterministic, so
   within-arm variance is modeled, not just between-arm differences.

## Components

```
[eval dataset] -> [model arms: local + API] -> [orchestrator + results store]
               -> [LLM-as-judge, calibrated] -> [stats analysis + dashboard]
```

- **Eval dataset — task packs.** A task is a directory under
  `backend/tasks/<name>/`: a `task.yaml` (label set, default eval prompt,
  judge rubric, data-file pointer) plus a data file (vendored
  `.txt`/`.jsonl`). `arms.yaml`'s top-level `task:` key selects the active
  one. Ships with `financial_sentiment` (Financial PhraseBank) and
  `ag_news`. Bring your own eval = drop in a JSONL file + a `task.yaml`, no
  code change. Design doc:
  `docs/design/2026-08-30-task-agnostic-eval-design.md`.
- **Model arms.** Four adapter implementations behind a shared
  `ModelAdapter` protocol: `OpenAICompatibleAdapter` (any provider speaking
  the OpenAI chat-completions schema — Ollama, OpenAI, OpenRouter, Groq,
  …), `AnthropicAdapter` (Claude's distinct schema), and two
  subscription-seat CLI adapters with no per-token price —
  `ClaudeCodeCLIAdapter` and `CodexCLIAdapter`, which drive the `claude` /
  `codex` CLIs non-interactively under an already-authenticated
  subscription seat instead of calling a metered API. Arms are declared in
  `backend/arms.yaml`, never hardcoded in code — adding or swapping a
  provider is a config edit. Design docs:
  `docs/design/2026-08-25-model-adapter-layer-design.md`,
  `docs/design/2026-08-27-subscription-cli-adapters-design.md`.
  - Local: Ollama serving **Qwen3-8B** by default (fits a <16GB VRAM GPU).
  - API: model-agnostic via config — `arms.yaml` has example arms for
    GPT-4o-mini and Claude Haiku; any OpenAI-schema or Anthropic provider
    works without code changes.
  - **Prompts are arms too.** An arm carries an optional `prompt_template`
    (must contain `{text}`; when unset it falls back to the active task's
    `eval_prompt`). Two arms with the same model but different templates
    A/B the prompts — the paired stats apply unchanged. `config/arms.py`
    wraps each adapter in an `Arm` (`name`, `adapter`, `prompt_template`);
    `GET /arms` reports the resolved template.
- **Orchestrator.** Celery + Redis, runs (eval set) × (arms) × (N repeats)
  as async jobs. Design doc:
  `docs/design/2026-08-25-orchestration-layer-design.md`.
- **Results store.** Postgres: raw prompts, outputs, judge scores, latency,
  token counts, per-call cost estimate.
- **Judge layer.** LLM-as-judge with a fixed rubric prompt (supplied
  per-task by the active task pack; a fixed integer 1–5 score, so the
  stats/calibration layers are task-independent), calibrated against a
  hand-labeled gold subset before being trusted on the full run. Design
  doc: `docs/design/2026-08-27-judge-layer-calibration-design.md`.
- **Stats layer.** `backend/app/stats/` — hierarchical paired bootstrap +
  Wilcoxon signed-rank + Holm-Bonferroni correction (`paired_tests.py`),
  PyMC-based Bayesian equivalence test (`bayesian.py`), closed-form
  sample-size/power calculator (`power.py`), and `RunResult` aggregation
  (`aggregation.py`). Exposed via
  `GET /runs/{run_id}/compare|equivalence|power`. Design doc:
  `docs/design/2026-08-27-stats-layer-design.md`.
- **Dashboard.** React app (`frontend/`) — run list with live status
  polling, and a per-run tabbed view: win-rate table (pairwise quality diff
  + CI + corrected p-value), cost/latency/quality frontier scatter, and
  judge calibration report. Runs can be started from a "New run" form
  (backed by `GET /arms` + `POST /runs`), not only via `curl`. Design docs:
  `docs/design/2026-08-27-dashboard-design.md`,
  `docs/design/2026-08-29-cli-and-dashboard-run-design.md`.
- **`pe` CLI.** `backend/app/cli/`, console entrypoint (`uv run pe …` from
  `backend/`). One command over the whole loop: `docker compose` lifecycle
  (`up`/`down`/`logs`/`seed`), task packs (`tasks`), runs
  (`run`/`status`/`watch`/`results`/`arms`) and stats
  (`stats compare|equivalence|power`) over HTTP, plus the host-side
  calibration scripts (`calibrate select|import|report`). `scripts/demo.sh`
  chains it end to end.
- **Agent-facing judge tool.** MCP server
  (`backend/app/mcp_judge_server.py`, server `rubric-judge`) exposes a
  single `score_output_against_gold` tool wrapping the existing
  `judge/scorer.py:score_output`, so an MCP client can score a candidate
  response against a gold label directly, without running a full eval.
  Discovered automatically via the repo-root `.mcp.json`. Design doc:
  `docs/design/2026-08-29-agent-facing-judge-tool-design.md`.
- **Local fine-tune.** `backend/app/training/` — QLoRA fine-tune of
  Qwen3-8B on the Financial PhraseBank *lower-agreement* subset (disjoint
  from the all-agree eval set, enforced by a leakage guard in
  `training/dataset.py`), then merge → GGUF → `ollama create` so the
  fine-tuned model is just another `openai_compatible` arm. Driven by
  `pe finetune prep|train|export|report` over `backend/training.yaml`.
  Training deps are an optional `training` extra, out of the core/CI path.
  Design doc: `docs/design/2026-08-29-local-finetune-design.md`.

## Tech stack

- Backend: Python, FastAPI, SQLModel
- Async jobs: Celery, Redis
- Database: PostgreSQL, Alembic migrations
- Local model serving: Ollama (OpenAI-compatible endpoint at
  `localhost:11434/v1`)
- API models: model-agnostic, bring-your-own-key — any OpenAI-schema
  provider plus Anthropic (Claude), declared in `backend/arms.yaml`; keys
  read from per-arm env vars (`backend/.env.example` documents the ones the
  example arms use)
- Frontend: TypeScript, React, Vite
- Containerization: Docker
- Stats: scipy/statsmodels for frequentist tests; PyMC for the Bayesian
  paired-difference model
- Package manager: `uv`

## Data

- **Financial PhraseBank** (Malo et al. 2014), 100%-agreement subset —
  2264 sentence-level, 3-class financial-sentiment examples with expert
  agreement. Vendored at
  `backend/data/financial_phrasebank/sentences_allagree.txt` (regenerate
  with `backend/scripts/fetch_financial_phrasebank.py`). Licensed
  **CC BY-NC-SA 3.0** — attribution, non-commercial, share-alike; those
  terms bind the bundled copy and anything derived from it. A commercial
  user substitutes a permissively-licensed set by adding a task pack, no
  code change.
- **AG News** — 4-class news-topic classification, the secondary task pack
  (120-row stratified sample vendored, non-commercial research use).
- **Gold subset for judge calibration** — a stratified sample (~30–50
  examples) with human labels held back from judge-only scoring, used
  purely to check judge agreement. Stores row IDs + human labels, not
  redistributed source text.

## Non-goals

- Not a general-purpose LLM chat UI.
- Not fine-tuning the API models — only the local model is a fine-tuning
  candidate.
- Not optimizing for maximum absolute model quality — optimizing for a
  rigorous, honest comparison between arms.

## Experiment reports

Executed comparisons live in `docs/reports/`:

- `2026-08-30-finetune-comparison.md` — QLoRA fine-tune vs. base Qwen3-8B
- `2026-08-31-prompt-ab-comparison.md` — terse vs. chain-of-thought prompt
  A/B on AG News
- `2026-08-31-judge-calibration-financial.md` — judge calibration against
  hand labels
- `2026-09-01-local-vs-cli-hosted.md` — local vs. hosted (subscription-seat)
  arm

`docs/RESULTS.md` is the readable walk-through of all four.
