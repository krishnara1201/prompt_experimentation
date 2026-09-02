# Local LoRA fine-tune + fine-tuned-vs-base-vs-API comparison — Design Spec

Date: 2026-08-29
Status: Approved (design)
Scope: a stretch extension of the platform described in `docs/ARCHITECTURE.md`

## Goal

Fine-tune the local model (Qwen3-8B) with QLoRA on the financial-sentiment
task, serve the result as a first-class arm, and produce an honest
fine-tuned-local vs. base-local vs. API comparison using the platform's
existing paired-stats and frontier machinery.

The deliverable is the executed comparison and its written report — not
just the capability.

## Non-goals

- Not a hosted-model fine-tune (only the local model is a candidate — see
  `docs/ARCHITECTURE.md` non-goals).
- Not an HTTP/Celery-driven training service. Fine-tuning is a host-side
  workflow (`app/training/` + `pe finetune` + host scripts), mirroring the
  judge-calibration workflow. No `POST /finetune`.
- Not reasoning/thinking-mode fine-tuning. Financial PhraseBank has no
  reasoning traces; this is a single-label classification fine-tune with
  Qwen3 thinking disabled.
- No changes to the adapter layer, orchestrator, worker, stats layer, or
  judge. The fine-tuned model is served through Ollama and is just another
  `openai_compatible` arm.

## Context (current state)

- Local arm today: `qwen3-8b-local`, `adapter: openai_compatible`, pointed
  at Ollama's `qwen3:8b` (Q4_K_M GGUF) at the WSL eth0 IP.
- Eval set: all 2271 `sentences_allagree` rows are seeded into
  `eval_example` (`source = financial_phrasebank_allagree`).
- Judge-calibration rows (`judge_calibration_label`) reference
  `run_result` → `eval_example`, so every calibration example is an
  `allagree` sentence and a subset of the eval set.
- Eval prompt is fixed in `app/tasks/worker.py`:
  `EVAL_PROMPT_TEMPLATE` / `render_eval_prompt(text)`.
- Judge rubric (`app/judge/rubric.py`) grades a response 1–5 against the
  gold label; `judge_score` is the quality metric the stats layer uses.
- Stats endpoints already exist: `GET /runs/{id}/compare` (paired
  bootstrap + Wilcoxon + Holm–Bonferroni), `/equivalence` (PyMC Bayesian
  equivalence on `judge_score`), `/summary` (frontier data), `/power`.
- `pe` CLI: `app/cli/__init__.py` with `stats` and `calibrate` sub-typers;
  `calibrate` commands shell out to `backend/scripts/*` via
  `backend_script()` in `app/cli/_shell.py`.
- Hardware: RTX 4070, 12 GB VRAM. Ollama running locally.

## Design decisions (resolved during design)

| Decision | Choice | Rationale |
|---|---|---|
| Training data source | Financial PhraseBank lower-agreement subset (`sentences_75agree`, minus `allagree`) | Keeps 100% of `allagree` pristine as eval + calibration set; no eval-set shrinkage. ~1.1k usable rows. |
| Packaging | `app/training/` module + `pe finetune` CLI + host scripts; optional `training` dependency group | Fine-tuning is a first-class, structured platform module, but training deps stay out of the core/runtime/CI path. |
| Thinking mode | Disabled | Single-label classification; thinking adds latency the frontier plot penalizes and needs synthetic traces we don't have. |
| Fine-tune stack | Unsloth QLoRA | One dep covers train + merge + GGUF; ~9 GB VRAM fits the 4070; Qwen3 supported. |
| Serving | Merge LoRA → GGUF → `ollama create` → new `arms.yaml` entry | Zero adapter/orchestrator code; fair quant-to-quant comparison vs. the base local arm. |
| Comparison scope | Executed run + committed report | It is the portfolio deliverable. |

## Architecture

```
training.yaml ─┐
               ├─> app/training/dataset.py ──> {train,val}.jsonl        (leakage-guarded)
HF 75agree ────┘        │
                        ├─> app/training/train.py  ──> adapter/         (Unsloth QLoRA)
                        └─> app/training/export.py ──> gguf/ + Modelfile ──> `ollama create`
                                                          │
                                                          └─> arm_snippet.yaml  (paste into arms.yaml)

arms.yaml (+ ft arm) ──> `pe run` ──> RunResult + judge ──> GET /runs/{id}/compare|equivalence|summary
                                                                     │
                                          scripts/finetune_report.py ─┴─> docs/.../reports/2026-08-29-finetune-comparison.md (+ PNG)
```

## Components

### 0. Shared eval prompt (refactor)

Move `EVAL_PROMPT_TEMPLATE` and `render_eval_prompt` out of
`app/tasks/worker.py` into a new `app/eval_prompt.py`. `worker.py` imports
from there; `app/training/dataset.py` imports the same function.

- `tests/test_eval_prompt.py` pins the rendered string byte-for-byte
  (guards the refactor and the train/serve match).
- `worker.py` keeps a re-export or updates its single call site — pick
  whichever keeps the diff minimal; no behavior change.

### 1. `app/training/dataset.py`

`build_sft_dataset(cfg: TrainingConfig) -> DatasetBuildResult`

- Loads `cfg.source_dataset` / `cfg.source_config` via
  `datasets.load_dataset`. Concatenates all splits to recover the full
  subset (the HF configs are seed splits over one corpus — see
  `scripts/fetch_financial_phrasebank.py`).
- Normalizes each sentence: `casefold()` + collapse internal whitespace +
  strip. Same normalization applied to every `eval_example.text` pulled
  from the DB (async query, reuse `app/db/session.py`).
- **Leakage guard**:
  - Compute `overlap = train_norm ∩ eval_norm`.
  - Drop every overlapping row; record `dropped_count`.
  - `assert not overlap` is NOT the guard (overlap is expected since
    `allagree ⊂ 75agree`) — the guard is: after dropping, re-check the
    intersection is empty and raise `LeakageError` if not; and raise
    `LeakageError` if the surviving pool is < `cfg.min_pool_size`
    (default 500) — a signal the subset or normalization is wrong.
  - Calibration rows need no separate handling: all are `allagree`,
    already removed with the eval set.
- Formats survivors into chat records:
  `{"messages": [{"role": "user", "content": render_eval_prompt(sentence)},
  {"role": "assistant", "content": label}]}` where `label ∈
  {positive, negative, neutral}` (map from the dataset's int label).
- Optional `cfg.balance_neutral` (default `false`): downsample the
  `neutral` class to the size of the larger minority class, seeded.
- Deterministic seeded split: `cfg.val_fraction` (default 0.1) held out
  from the training pool for loss monitoring only. Val is never `allagree`.
- Writes `train.jsonl` / `val.jsonl` + `LICENSE.txt` (CC BY-NC-SA 3.0
  attribution, Malo et al. 2014) to
  `backend/training/artifacts/<cfg.run_name>/`.
- Returns `DatasetBuildResult`: paths, per-class counts (train/val),
  `dropped_count`, `pool_size`, `source` string.

`backend/training/artifacts/` is gitignored. The dataset is **not
committed** (same non-commercial posture as the eval set).

### 2. `app/training/config.py`

- `@dataclass TrainingConfig` with fields:
  `run_name: str`, `base_model: str` (default `unsloth/Qwen3-8B`),
  `source_dataset: str`, `source_config: str`,
  `max_seq_len: int` (default 512 — sentences are short),
  `lora_r: int` (16), `lora_alpha: int` (16), `lora_dropout: float` (0.0),
  `lora_target_modules: list[str]` (default the 7 attn+MLP proj names),
  `epochs: float` (3), `learning_rate: float` (2e-4),
  `batch_size: int` (8), `grad_accum: int` (2), `seed: int` (42),
  `val_fraction: float` (0.1), `balance_neutral: bool` (false),
  `min_pool_size: int` (500),
  `gguf_quant: str` (default `q5_k_m`),
  `ollama_tag: str` (default `ft-qwen3-8b`),
  `output_dir: str` (default `training/artifacts/<run_name>`).
- `load_training_config(path: str = "training.yaml") -> TrainingConfig`
  with `InvalidTrainingConfigError` on unknown/missing keys (mirror the
  style of `app/config/arms.py`).
- `backend/training.yaml` committed with the defaults above filled in
  explicitly — the single config surface, like `arms.yaml`.

### 3. `app/training/train.py`

`run_training(cfg: TrainingConfig, dataset: DatasetBuildResult) -> TrainingResult`

- Guarded import at call time:
  `try: from unsloth import FastLanguageModel` … `except ImportError:
  raise RuntimeError("Training deps missing — run `uv sync --extra
  training` (needs a CUDA GPU).")`.
- `FastLanguageModel.from_pretrained(cfg.base_model, load_in_4bit=True,
  max_seq_length=cfg.max_seq_len)`.
- `FastLanguageModel.get_peft_model(model, r=cfg.lora_r,
  lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
  target_modules=cfg.lora_target_modules, use_gradient_checkpointing=
  "unsloth", random_state=cfg.seed)`.
- Load `train.jsonl` / `val.jsonl`; render each record through the
  tokenizer chat template with `enable_thinking=False`; mask the prompt
  tokens from the loss (train on completion only).
- `trl.SFTTrainer` (Unsloth-patched) with `cfg` hyperparams,
  `eval_strategy="epoch"`, `report_to=[]`, seeded.
- Train; capture per-epoch train/eval loss.
- Save the LoRA adapter to `<output_dir>/adapter/`
  (`model.save_pretrained` + tokenizer). **No merge/GGUF here.**
- `TrainingResult`: adapter path, loss history, wall-clock seconds,
  `cfg.seed`, resolved `base_model` commit hash if available.
- Note in the docstring: training is seeded but not bit-reproducible.

### 4. `app/training/export.py`

`export_arm(cfg: TrainingConfig) -> ExportResult`

- Guarded import (same as train).
- Reload base + `<output_dir>/adapter/`;
  `model.save_pretrained_merged("<output_dir>/merged", tokenizer,
  save_method="merged_16bit")`.
- `model.save_pretrained_gguf("<output_dir>/gguf", tokenizer,
  quantization_method=cfg.gguf_quant)` → `<output_dir>/gguf/*.gguf`
  (Unsloth clones/builds llama.cpp on first run).
- Write `<output_dir>/Modelfile`:
  - `FROM ./gguf/<file>.gguf`
  - `PARAMETER num_ctx <max_seq_len>`
  - `PARAMETER temperature 0`
  - `PARAMETER stop "<|im_end|>"`
  - A `TEMPLATE` (or system directive) that pins Qwen3 to **non-thinking**
    so `render_eval_prompt` output is passed through unchanged and the arm
    emits a bare label. Exact incantation verified during implementation
    (see Risks).
- `subprocess.run(["ollama", "create", cfg.ollama_tag, "-f",
  "<output_dir>/Modelfile"])`; non-zero exit → `ExportError`.
- Write + print `<output_dir>/arm_snippet.yaml`:
  ```yaml
  - name: ft-qwen3-8b-local
    adapter: openai_compatible
    base_url: <same base_url as qwen3-8b-local>
    model: <cfg.ollama_tag>
    max_tokens: 1024
  ```
  `base_url` is read from the existing `qwen3-8b-local` arm in `arms.yaml`
  so it stays correct after a WSL IP change.
- `ExportResult`: gguf path, Modelfile path, ollama tag, snippet path.

The user pastes the snippet into `arms.yaml` manually — explicit and
reviewable, same as adding any arm. Nothing auto-edits `arms.yaml`.

### 5. Host scripts — `backend/scripts/`

Thin `argparse` wrappers over `app/training/` (mirror the calibration
scripts wrapping `app/judge/`):

- `finetune_prep.py` — `build_sft_dataset`; print class counts,
  `dropped_count`, `pool_size`, output paths.
- `finetune_train.py` — `build_sft_dataset` (or reuse existing artifacts
  via `--reuse-dataset`) then `run_training`; print loss history + wall
  clock. `--dry-run`: build dataset + instantiate config + render 3 sample
  formatted records, then stop before any model load (CI-safe, no GPU).
- `finetune_export.py` — `export_arm`; print the arm snippet.
- `finetune_report.py` — see §7.

### 6. `pe finetune` CLI

- New `finetune_app = typer.Typer(...)`; `app.add_typer(finetune_app,
  name="finetune")` in `app/cli/__init__.py`.
- Commands shell out via `backend_script()` (host-side; needs GPU, local
  `.env`, `training` extra):
  - `pe finetune prep` → `scripts.finetune_prep`
  - `pe finetune train [--dry-run] [--reuse-dataset]` → `scripts.finetune_train`
  - `pe finetune export` → `scripts.finetune_export`
  - `pe finetune report --run-id N --baseline ARM --candidate ARM
    [--epsilon E] [--out PATH]` → `scripts.finetune_report`
- Help text states the GPU / `training` extra precondition, like the
  `calibrate` sub-typer states its `.env` precondition.

### 7. Comparison run + report

Workflow (documented in `backend/README.md` + `docs/ARCHITECTURE.md`):

1. `pe finetune prep` → `train` → `export`; paste snippet into `arms.yaml`
   as `ft-qwen3-8b-local`.
2. `pe run --arms qwen3-8b-local,ft-qwen3-8b-local,gpt-4o-mini,claude-haiku
   --repeats N --sample-size M` (or the dashboard New Run form). Judge
   auto-scores every result.
3. `pe finetune report --run-id <id> --baseline qwen3-8b-local
   --candidate ft-qwen3-8b-local` (repeat `--candidate` per API arm, or
   loop internally).

`scripts/finetune_report.py`:

- Pulls `GET /runs/{id}/summary`, `/compare`, `/equivalence` from
  `$PE_API_URL` (reuse `app/cli/_api.py` helpers if importable, else
  `httpx`).
- Renders `docs/reports/2026-08-30-finetune-comparison.md`:
  - Run metadata (arms, N, sample size, seed, judge model).
  - **Win-rate / quality table**: for each pair (ft vs base; ft vs each
    API arm) — mean `judge_score` diff, bootstrap CI, Holm-corrected
    p-value, Wilcoxon result.
  - **Bayesian equivalence**: `P(judge_score_ft ≥ judge_score_api − ε)`
    for `--epsilon` (default 0.5 on the 1–5 scale), per API arm.
  - **Cost / latency / quality frontier**: a committed matplotlib PNG
    (`docs/reports/2026-08-30-finetune-frontier.png`) —
    x = mean latency, y = mean judge_score, marker size/label = cost —
    plus the underlying table.
  - **Training-cost accounting**: one-time cost = training wall-clock
    hours × a configurable local `$/GPU-hour` estimate (CLI flag,
    default stated in the report as an assumption). Reported *separately*
    from per-inference `cost_estimate_usd`, with a break-even line
    ("vs. `gpt-4o-mini` at $X/1k calls, the fine-tune amortizes after
    ~K calls").
  - **Honest read** (prose): did fine-tuning close the gap to the API
    arms on quality, and at what one-time cost / latency.
- `matplotlib` is in the `training` extra; the report script is run
  host-side like the other finetune scripts.

## Data flow

1. `training.yaml` → `TrainingConfig`.
2. HF `75agree` + `eval_example` texts → leakage-guarded, formatted
   `{train,val}.jsonl`.
3. JSONL + base model → Unsloth QLoRA → `adapter/`.
4. `adapter/` + base → merged 16-bit → GGUF → `Modelfile` →
   `ollama create ft-qwen3-8b` → `arm_snippet.yaml`.
5. Snippet pasted into `arms.yaml` → `pe run` over 4 arms → `RunResult`
   rows → chained judge task fills `judge_score`.
6. Stats endpoints → `finetune_report.py` → committed markdown + PNG.

## Error handling

- `LeakageError` (dataset.py) — overlap survives the drop, or pool below
  `min_pool_size`. Hard failure, no artifacts written.
- `InvalidTrainingConfigError` (config.py) — unknown/missing/mistyped
  `training.yaml` keys, following `app/config/arms.py` conventions.
- `RuntimeError` on missing training deps — guarded imports in
  `train.py` / `export.py` with an actionable message.
- `ExportError` — `ollama create` non-zero exit; surfaces ollama's stderr.
- HF dataset load failure (bad `source_config`, offline) — let
  `datasets`' exception propagate with the config values in the message.
- Report script: a non-terminal run, or a missing arm in the run, →
  clear error naming the run state / arm before hitting the endpoints.

## Testing

CI runs without the `training` extra, so all CI tests avoid `unsloth` /
`torch` / real GPU.

- `tests/test_eval_prompt.py` — rendered prompt is byte-identical to the
  pre-refactor string; `worker.py` still renders the same thing.
- `tests/training/test_dataset.py`:
  - HF loader monkeypatched to a small in-memory fixture; DB eval texts
    stubbed.
  - Planted overlap → dropped, `dropped_count` correct, survivors clean.
  - Overlap that cannot be normalized away → `LeakageError`.
  - Pool below `min_pool_size` → `LeakageError`.
  - Formatted record shape: user content `== render_eval_prompt(sentence)`,
    assistant content is the mapped label word.
  - Deterministic split given a seed; val disjoint from train.
  - `balance_neutral` downsamples as specified.
- `tests/training/test_config.py` — valid `training.yaml` parses to the
  expected dataclass; unknown key / wrong type / missing required →
  `InvalidTrainingConfigError`.
- `tests/scripts/test_finetune_prep.py` — `--dry-run`-style path: script
  calls `build_sft_dataset` and prints the summary (loader monkeypatched).
- `tests/cli/` (matching existing CLI tests) — `pe finetune *` shells out
  with the expected argv via the `_run` indirection; no subprocess.
- `train.py` / `export.py` — no CI unit tests; exercised manually and via
  `finetune_train.py --dry-run` (dataset + config only). Real runs are
  documented in `backend/README.md`.

## Dependencies

New optional group in `backend/pyproject.toml`:

```toml
[project.optional-dependencies]
training = [
    "unsloth",
    "trl",
    "peft",
    "transformers",
    "datasets",
    "bitsandbytes",
    "torch",
    "accelerate",
    "matplotlib",
]
```

- Core backend, Celery worker, Docker images, and CI are unchanged — the
  runtime serving path is Ollama, which needs none of these.
- Version pins decided at implementation time against a known-good
  Unsloth + Qwen3 + CUDA combination on the 4070; recorded in
  `backend/README.md`.

## Risks / verify during implementation

1. **HF mirror for `sentences_75agree`** — confirm a mirror (likely
   `gtfintechlab/financial_phrasebank_sentences_75agree`, matching the
   `_allagree` repo already used) loads without `trust_remote_code`.
   Fallbacks, in order: canonical `takala/financial_phrasebank` (loading
   script), then `sentences_66agree`, then `sentences_50agree`. The
   config is a `training.yaml` field so switching is a config edit.
2. **Unsloth GGUF export toolchain** — needs cmake/gcc for llama.cpp;
   Unsloth normally auto-clones+builds. Document the manual
   `convert_hf_to_gguf.py` + `llama-quantize` fallback in the README.
3. **Qwen3 non-thinking in an Ollama Modelfile** — pin the exact
   TEMPLATE / parameter form so the arm reliably emits a bare label with
   no `<think>` block. Verify against a handful of eval sentences before
   the full run.
4. **VRAM on the 4070 (12 GB)** — 4-bit QLoRA r=16 on 8B ≈ 8–10 GB. If
   OOM: lower `max_seq_len`, `batch_size=1`, raise `grad_accum`. All are
   `training.yaml` fields.
5. **Base-model quantization parity** — the base local arm serves
   `qwen3:8b` at Q4_K_M; the fine-tuned arm serves at `q5_k_m` by default.
   Either set `gguf_quant: q4_k_m` to match exactly, or note the
   difference in the report. Default: match at `q4_k_m` for a clean
   comparison; revisit if quality loss is visible.
6. **Judge / arm model overlap** — the judge runs on `claude-code-cli`
   (opus). The API arms include `claude-haiku`. Existing caveat in
   `backend/README.md` already covers this; the report restates it.

## Documentation updates

- `docs/ARCHITECTURE.md` — note the fine-tune workflow with a one-paragraph summary and spec
  link.
- `backend/README.md` — a "Local fine-tune" section: the
  `training` extra, `training.yaml`, the `pe finetune prep|train|export`
  flow, pasting the arm snippet, running the comparison, generating the
  report, and the toolchain/VRAM fallbacks.
- `.gitignore` — `backend/training/artifacts/`.
