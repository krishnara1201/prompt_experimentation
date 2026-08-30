# Task-agnostic eval — bring-your-own dataset + rubric

Date: 2026-08-30
Status: Approved (design)
Phase: 8 (extends "prompts as arms", per `CLAUDE.md`)

## Goal

Make the eval loop task-agnostic. Today the dataset (Financial PhraseBank),
the eval prompt, the judge rubric, and the valid label set
(`positive`/`negative`/`neutral`) are all hardwired to 3-class financial
sentiment. After this change, a new evaluation task is a config edit plus a
JSONL file — no code changes — the same way adding a model arm already is.

This is the other half of the "prompts as arms" story: an arm already
carries a `prompt_template`, but every arm is still answering the one baked-in
task. After this change, the task itself is swappable, and a prompt A/B can
run on any dataset.

The deliverable is the capability **and** an executed prompt-A/B comparison
on a non-financial dataset (AG News) with a written report — see
"Deliverable 2".

## Non-goals

- **Not** a configurable judge score scale. The judge still emits an integer
  1–5. The stats layer, the calibration binarization (`score >= 4` =
  "correct"), the dashboard, and the response parser all assume this, and a
  1–5 rubric already expresses arbitrary classification tasks. Only the
  rubric *text* and the label *set* become config.
- **Not** a multi-task run. One run targets exactly one task. Runs do not mix
  datasets.
- **Not** a dataset-download service. Task data files are vendored (or
  fetched by a task-specific script, like `fetch_financial_phrasebank.py`);
  the platform reads a local file.
- **Not** a change to the adapter layer, the Celery/Redis orchestration
  topology, the stats layer, or the calibration statistics. Those operate on
  a numeric `judge_score` and are already task-neutral.
- **Not** removing or relicensing the vendored Financial PhraseBank corpus.
  It stays exactly where it is; `financial_sentiment` becomes one task pack
  among others and remains the default.

## Context (current state)

- **Eval examples:** `EvalExample(text, gold_label, source)` in Postgres.
  Seeded by `backend/scripts/seed_eval_examples.py`, which reads the vendored
  `backend/data/financial_phrasebank/sentences_allagree.txt` (`sentence@label`
  lines) via `app/data/financial_phrasebank.py:load_examples`
  (`VALID_LABELS = {"positive","negative","neutral"}`), writing
  `source = "financial_phrasebank_allagree"`.
- **Run creation:** `POST /runs` samples **all** `EvalExample` rows ordered
  by `id` (no `source` filter), fans out `(examples × arms × repeats)` Celery
  tasks. `Run(arm_names, sample_size, repeats, seed, total_calls)` — no task
  field.
- **Eval prompt:** `app/eval_prompt.py:EVAL_PROMPT_TEMPLATE` /
  `render_eval_prompt(text)`. An `Arm` (`app/config/arms.py`) has
  `prompt_template: str = EVAL_PROMPT_TEMPLATE`; `Arm.render(text)` formats
  it with `{text}`. `load_arms` validates the template contains `{text}` and
  no other placeholder.
- **Judge:** `app/judge/rubric.py:RUBRIC_PROMPT_TEMPLATE` — a fixed
  financial-sentiment rubric, `{input_text}` / `{gold_label}` /
  `{model_output}` placeholders, "Score the response 1-5". `score_output`
  (`app/judge/scorer.py`) renders it, calls the judge adapter, parses
  `SCORE: <1-5>` / `RATIONALE: <text>`. The judge task
  (`app/tasks/worker.py:execute_judge_call`) loads
  `(EvalExample.text, EvalExample.gold_label, RunResult.output_text)` by
  `run_result_id` and calls `score_output` with the config-resolved judge
  adapter (`load_judge_arm`, reloaded per call from `arms.yaml`).
- **MCP judge server:** `app/mcp_judge_server.py` — server name
  `financial-sentiment-judge`, one tool `score_financial_sentiment` with
  `gold_label: Literal["positive","negative","neutral"]`, hardcoded
  `GOLD_LABELS`. Registered in the repo-root `.mcp.json`. Smoke-tested by
  `backend/scripts/judge_tool_dryrun.py`.
- **Training dataset builder:** `app/training/dataset.py` —
  `LABEL_NAMES = {0:"negative",1:"neutral",2:"positive"}`,
  `_VALID_LABELS`, `_write_jsonl` uses `render_eval_prompt`,
  `_balance_neutral` assumes a `neutral` class. Driven by
  `backend/training.yaml` (no task field). Leakage guard `fetch_eval_texts`
  reads **all** `EvalExample.text` — already task-neutral.
- **CLI:** `pe seed` shells `docker compose run --rm migrate uv run python -m
  scripts.seed_eval_examples`. `pe run` posts to `/runs`.
- **Frontend:** `NewRunForm.tsx` posts `{repeats, sample_size?, seed?,
  arms?}` to `/runs`. No task concept.
- **Stats / calibration:** `app/stats/*` and `app/judge/calibration.py`
  operate purely on numeric `judge_score` / `human_score`.
  `select_calibration_sample.py` stratifies by `gold_label` (any label set),
  `calibration_report.py` uses `CORRECT_THRESHOLD = 4`. **No changes
  needed.**

## Design decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Config unit | A **task pack** directory `backend/tasks/<name>/` with `task.yaml` + a data file | One coherent, swappable unit; keeps `arms.yaml` about arms. `EvalExample.source` already namespaces datasets in the DB. |
| Task selection | Top-level `task:` key in `arms.yaml`, default `financial_sentiment` | Mirrors how the `judge:` block already lives in `arms.yaml`. |
| Judge score scale | Fixed integer 1–5 | Stats, calibration threshold, dashboard, parser all assume it; a 1–5 rubric covers arbitrary classification. |
| Which run's rubric the judge uses | The task recorded on the run, passed through Celery kwargs | The global `arms.yaml` `task:` can change between a run starting and its judge tasks firing. The run must be self-consistent. |
| Data file format | JSONL (`{"text", "gold_label"}` per line) default; `phrasebank` format for the existing `@`-delimited file | New tasks use the obvious format; the vendored financial file stays untouched. |
| Blast radius | Orchestrator + seed + judge **and** MCP judge server **and** training dataset builder | User chose full generalization so a non-financial task is fine-tunable and agent-scoreable too. |
| Frontend | Minimal: a task dropdown in `NewRunForm` + task shown on the run header | Otherwise the dashboard can't drive a non-default task, breaking the "no curl needed" story in `CLAUDE.md`. |
| Backward compatibility | `financial_sentiment/task.yaml` carries the **current** eval prompt + rubric text **verbatim** | Existing behaviour byte-identical; existing judge-calibration numbers stay valid. |

## Architecture

### New: task pack

```
backend/tasks/
  financial_sentiment/
    task.yaml
    # data: points back at the existing vendored file, not a copy
  ag_news/
    task.yaml
    data.jsonl        # vendored stratified sample (Deliverable 2)
    fetch_ag_news.py  # regenerates data.jsonl from HF (Deliverable 2)
```

`task.yaml` schema:

```yaml
name: ag_news
description: a news-topic classification         # noun phrase, goes into the rubric
labels: [World, Sports, Business, Sci/Tech]
source: ag_news_sample                           # EvalExample.source tag
data: data.jsonl                                 # relative to the task dir
format: jsonl                                    # jsonl | phrasebank ; default jsonl
eval_prompt: |                                   # must contain {text}
  Classify the topic of the news snippet below as exactly one of:
  World, Sports, Business, Sci/Tech. Answer with only the label.

  Snippet: {text}
rubric: |                                        # must contain {input_text} {gold_label} {model_output}
  You are grading a {description} model's response.

  Input text: {input_text}
  Correct label: {gold_label}
  Model's response: {model_output}

  Score the response 1-5:
  5 = correctly identifies the label as {gold_label}, clearly and directly
  4 = correct label, minor clarity/formatting issues
  3 = ambiguous, hedged, or only partially matches
  2 = wrong label but otherwise coherent/on-topic
  1 = wrong label, off-topic, malformed, or non-responsive

  Respond in exactly this format:
  SCORE: <1-5>
  RATIONALE: <one sentence>
# label_names: [negative, neutral, positive]     # optional, int->word, training only
```

`financial_sentiment/task.yaml` uses `format: phrasebank`,
`data: ../../data/financial_phrasebank/sentences_allagree.txt`,
`source: financial_phrasebank_allagree`, and its `eval_prompt` / `rubric`
are the current strings copied verbatim.

### New: `app/config/tasks.py`

```python
@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    labels: tuple[str, ...]
    source: str
    data_path: Path            # resolved absolute
    data_format: str           # "jsonl" | "phrasebank"
    eval_prompt: str
    rubric: str
    label_names: tuple[str, ...] | None

TASKS_DIR = Path(__file__).../.."tasks"
DEFAULT_TASK = "financial_sentiment"

def load_task(name: str) -> TaskConfig: ...
def list_tasks() -> list[str]: ...             # directory names under tasks/
def active_task_name(arms_path: str) -> str:   # arms.yaml `task:` or DEFAULT_TASK
```

Validation on load (raise `InvalidTaskConfigError`):
- `task.yaml` exists and is a mapping with all required keys.
- `labels` non-empty list of non-empty strings.
- `eval_prompt` contains `{text}` and no other `{...}` field (reuse the
  check now in `arms.py:_validate_prompt_template`, lifted to a shared
  helper).
- `rubric` contains `{input_text}`, `{gold_label}`, `{model_output}`, and no
  other `{...}` field.
- `data_format in {"jsonl", "phrasebank"}`.
- `data` resolves to an existing file.
- `label_names`, if present, is a list covering the same set as `labels`.

### New: `app/data/loader.py`

```python
@dataclass
class TaskExample:
    text: str
    gold_label: str

def load_task_examples(task: TaskConfig) -> list[TaskExample]:
    if task.data_format == "phrasebank":
        rows = financial_phrasebank.load_examples(task.data_path)
        return [TaskExample(r.text, r.label) for r in rows]
    # jsonl
    ...
```

JSONL rules: one JSON object per line; blank lines and `#`-comment lines
skipped; `text` and `gold_label` required and non-empty; `gold_label` must be
in `task.labels`; any other keys ignored; malformed line → `ValueError` with
the line number. `app/data/financial_phrasebank.py` stays as-is (used by the
`phrasebank` branch and by the training builder).

### Changed: `app/eval_prompt.py`

Keep the constant name `EVAL_PROMPT_TEMPLATE` (it is the fallback when a task
omits `eval_prompt`, and `tests/test_eval_prompt.py` imports it).
`render_eval_prompt(text, template=EVAL_PROMPT_TEMPLATE)` gains the optional
template arg (used by the training builder). The module docstring is updated
to say the prompt is now task-scoped and per-arm-overridable.

### Changed: `app/config/arms.py`

- `Arm.prompt_template: str | None = None` (was `= EVAL_PROMPT_TEMPLATE`).
- `Arm.render(text)` — if `self.prompt_template` is `None`, the caller must
  supply the task default. Cleanest: `load_arms(config_path, *, task: TaskConfig)`
  resolves each arm's template to `entry.prompt_template or task.eval_prompt`
  at load time, so `Arm.prompt_template` is always concrete after loading and
  `render` is unchanged. `_validate_prompt_template` still runs on whichever
  string wins.
- `GET /arms` already reports the resolved template — it now reflects the
  task default for arms that don't override. `/arms` loads the active task
  to resolve.

### Changed: `Run` model + migration `0003`

```python
class Run(SQLModel, table=True):
    ...
    task: str = Field(default="financial_sentiment")
```

Migration `0003_add_run_task`: `op.add_column("run", sa.Column("task",
AutoString(), nullable=False, server_default="financial_sentiment"))`.
Downgrade drops the column. Existing rows backfill to the default via
`server_default`.

### Changed: `POST /runs` (`app/api/routes/runs.py`)

- `RunCreateRequest` gains `task: str | None = None`.
- Resolve: `task_name = payload.task or active_task_name(ARMS_PATH)`;
  `task_cfg = load_task(task_name)` (404/422 on unknown task).
- `load_arms(ARMS_PATH, task=task_cfg)`.
- Example query filters `WHERE EvalExample.source == task_cfg.source`
  (still `ORDER BY id`). Empty → 400 "No eval examples for task '<name>';
  run `pe seed --task <name>`".
- `Run(..., task=task_name)`.
- `_enqueue_all` passes `task_name` in each `run_single_call` kwargs.

### Changed: worker (`app/tasks/worker.py`)

- `run_single_call(self, run_id, example_id, example_text, arm_name,
  repeat_index, task_name)` — new trailing param.
- `execute_call(..., task_name)` — `load_task(task_name)`,
  `load_arms(ARMS_PATH, task=task_cfg)`, render as today. On
  `run_judge_call.apply_async`, pass `task_name` in kwargs.
- `run_judge_call(self, run_result_id, task_name)` /
  `execute_judge_call(..., task_name)` — `load_task(task_name)`, pass
  `task_cfg.rubric` into `score_output`.
- `score_output(adapter, input_text, gold_label, model_output, *,
  rubric_template)` — thread the template through to `render_prompt`.
- Backward-compat for in-flight tasks at deploy time: `task_name` defaults to
  `"financial_sentiment"` in both task signatures, so a message enqueued by
  old code still resolves.

### Changed: `app/judge/rubric.py`

Keep `RUBRIC_PROMPT_TEMPLATE` as the constant name (it is the default when a
task omits `rubric`). `render_prompt(input_text, gold_label, model_output,
template=RUBRIC_PROMPT_TEMPLATE)` gains the optional template arg;
`score_output` threads `rubric_template` through to it.

### Changed: MCP judge server (`app/mcp_judge_server.py`)

- Server name `financial-sentiment-judge` → `rubric-judge`.
- Tool `score_financial_sentiment` → `score_output_against_gold`.
  `gold_label` param becomes `str` (a `Literal` can't be built from runtime
  config); docstring says "against this platform's active evaluation task".
- Loads `active_task_name` + `load_task` per call (matches the existing
  per-call `load_judge_arm` reload). Validates `gold_label in task.labels`
  (error lists the allowed set), renders `task.rubric`.
- `ScoreResult` gains `task: str` alongside `judge_model`.
- `.mcp.json`: rename the server key to `rubric-judge`.
- `backend/scripts/judge_tool_dryrun.py`: rename `TOOL_NAME`, drop the
  hardcoded financial framing, drive it from `load_task(DEFAULT_TASK)` (still
  exercises real MCP stdio + real rows). The "out-of-domain gold_label" case
  becomes "label not in the active task's set".

### Changed: training dataset builder (`app/training/dataset.py`)

- `backend/training.yaml` + `app/training/config.py:TrainingConfig` gain
  `task: str = "financial_sentiment"`.
- `build_sft_dataset` calls `load_task(cfg.task)`; the int→word map comes
  from `task.label_names` (required if the HF source uses int labels — a
  validation error otherwise), valid-label set from `task.labels`,
  `_write_jsonl` renders `task.eval_prompt`.
- `_balance_neutral` → `_balance_majority(rows, seed, majority_label)`; skip
  entirely when the task has no obviously-dominant class. Keep current
  behaviour for `financial_sentiment` by setting a `balance_label: neutral`
  in `training.yaml` (default `None` → no balancing). *Out of scope to
  re-run any fine-tune; this is just keeping the builder honest.*
- Leakage guard unchanged (already reads all `EvalExample.text`).

### Changed: seed script + `pe seed`

- `seed_eval_examples.py`: `--task <name>` (default `financial_sentiment`).
  `load_task(name)` → `load_task_examples` → insert
  `EvalExample(text, gold_label, source=task.source)`. Idempotency check
  already keys on `source` — unchanged.
- `pe seed` gains `--task` passthrough. `pe` also gets `pe tasks` (list task
  pack names + which is active), mirroring `pe arms`.

### Changed: frontend (minimal)

- `GET /tasks` → `[{name, description, labels, active: bool, seeded_count:
  int}]`. `seeded_count` = `COUNT(EvalExample WHERE source = task.source)`.
- `NewRunForm.tsx`: a task `<select>` (defaults to the active task),
  included as `task` in the POST body. If `seeded_count == 0` for the
  chosen task, show "run `pe seed --task <name>` first" and disable submit.
- Run header (`RunDetail`): show the run's `task` (already returned once
  `Run.task` is in the `RunStatusResponse` / summary models — add the field).

## Data flow

```
task.yaml ──load_task──> TaskConfig ─┬─> seed_eval_examples --task ─> EvalExample(source=task.source)
                                     │
POST /runs {task} ──> resolve task ──┼─> filter EvalExample by source ─> sample ─> Run(task=name)
                                     │                                              │
                                     └─> load_arms(task=cfg) ─> Arm.render(eval_prompt)
                                                                                    │
                            Celery: run_single_call(..., task_name) ──> ModelResponse ──> RunResult
                                                                                    │
                            run_judge_call(run_result_id, task_name) ──> load_task ──> score_output(rubric=task.rubric)
                                                                                    │
                                                                              RunResult.judge_score (1-5)
                                                                                    │
                                          stats / calibration / dashboard (unchanged, numeric)
```

## Error handling

- Unknown task name (API): 422 with the list from `list_tasks()`.
- Malformed `task.yaml` / bad placeholders: `InvalidTaskConfigError` at load;
  in the worker this is caught by the existing "could not resolve arm"-style
  guard and persisted as a failed `RunResult` so the run can still reach
  `total_calls` (same pattern as a bad arm name today).
- Task has zero seeded examples: 400 at `POST /runs` (as today for the empty
  dataset), and the frontend disables submit.
- JSONL row with a `gold_label` not in `labels`: seed script aborts with the
  offending line number — no partial seed (the insert loop already commits
  once at the end).
- Judge given a `gold_label` outside the task's set (MCP path): `ValueError`
  before any judge call (matches today's `GOLD_LABELS` check).
- In-flight Celery messages across the deploy: defaulted `task_name` params
  keep them working.

## Testing

New:
- `tests/config/test_tasks.py` — load a good pack; each validation failure
  (missing key, bad placeholder, extra placeholder, unknown format, missing
  data file, label_names mismatch); `active_task_name` reads `arms.yaml`
  and falls back to default.
- `tests/data/test_loader.py` — JSONL happy path, blank/comment lines,
  missing field, empty field, out-of-set label, bad JSON → line number;
  `phrasebank` branch delegates correctly.
- `tests/judge/test_rubric.py` — extend: `render_prompt` with a custom
  template; default unchanged.
- `tests/api/test_tasks.py` — `GET /tasks` shape, `active` flag,
  `seeded_count`.
- `tests/api/test_runs.py` — extend: `POST /runs {task}` filters by source;
  unknown task → 422; zero-seeded task → 400; `Run.task` persisted;
  `task_name` in enqueued kwargs.
- `tests/tasks/test_execute_call.py` / `test_execute_judge_call.py` —
  extend: `task_name` threaded through; judge uses the task's rubric;
  defaulted `task_name` still resolves.
- `tests/test_mcp_judge_server.py` — rename; `score_output_against_gold`
  validates against the active task's labels; `task` in the result.
- `tests/scripts/test_seed_eval_examples.py` — `--task` seeds under the
  right source; idempotent per source.
- `tests/scripts/test_judge_tool_dryrun.py` — updated tool name / framing.

Regression (must stay green unchanged in behaviour):
- `tests/test_eval_prompt.py` — the default template string is unchanged.
- All `tests/stats/*`, `tests/judge/test_calibration.py`,
  `tests/scripts/test_calibration_report.py` — untouched.
- `financial_sentiment` end-to-end (seed → run → judge) produces the same
  prompt + rubric strings as before (a test asserts the resolved strings
  equal the pre-change constants).

Frontend:
- `NewRunForm` test: task select renders, defaults to active, posts `task`;
  zero-seeded task disables submit.

## Deliverable 2 — prompt A/B on AG News (separate bounded task, after this ships)

Executed as a follow-up once the capability is merged. Summary of intent so
the plan can scope it:

- **Task pack** `backend/tasks/ag_news/` — `fetch_ag_news.py` pulls
  `fancyzhx/ag_news` (or `sh0416/ag_news`) from HF, writes a **stratified
  ~120-row** `data.jsonl` (30/class), license/attribution header in a
  sibling `LICENSE.txt` and a note in the README "Data & license" section.
  AG News is derived from the AG corpus (ComeToMyHead); it is redistributed
  widely for research — vendor only the small sample, cite the source.
- **Arms** — two, same model (`qwen3:8b`, the only keyless arm), differing
  only in `prompt_template`:
  - `ag-news-terse`: one line, "Answer with only the label."
  - `ag-news-cot`: "Think step by step about what the snippet is about, then
    end with `Label: <one of ...>`." (rubric/parse tolerate trailing label.)
  - Both set Qwen3 `/no_think` in the prompt so the *arm* prompt, not the
    model's default thinking, is the manipulated variable.
- **Run** — `pe run --task ag_news --arm ag-news-terse --arm ag-news-cot -n
  120 -r 3` (720 calls, local, free).
- **Calibration** — hand-label a ~30-row stratified gold subset
  (`pe calibrate select|import|report`) and report judge/human agreement
  for AG News *before* trusting the judge scores, same as the financial
  task.
- **Analysis** — `pe stats compare` (paired bootstrap CI + Wilcoxon +
  Holm–Bonferroni), `pe stats equivalence` (Bayesian, ε on `judge_score`),
  `pe stats power`. Frontier scatter (latency/tokens vs judge score) —
  CoT should cost materially more tokens/latency for whatever quality it
  buys.
- **Report** — `docs/superpowers/reports/2026-08-31-prompt-ab-comparison.md`
  + `2026-08-31-prompt-ab-frontier.png`, same structure as
  `2026-08-30-finetune-comparison.md`. Honest result either way: the point
  is that the paired test resolves a small effect an unpaired win-rate
  wouldn't.
- **CLAUDE.md** — add a Phase 8 bullet and flip the "feature exists but no
  run has used it" gap.

## Rollout / sequencing

1. This spec → plan (`superpowers:writing-plans`).
2. Implement task-agnostic eval (all changes above), full test suite green,
   `financial_sentiment` regression proven byte-identical.
3. Commit; update `CLAUDE.md` architecture section (task packs) + the
   `backend/README.md` "Data & license" / dataset-swap notes.
4. Deliverable 2 as its own bounded task: build the AG News pack, run,
   calibrate, analyze, write the report.

## Open questions

None blocking. Deferred to Deliverable 2: exact AG News HF mirror, the final
CoT prompt wording, and whether 120×3 is enough power (the `pe stats power`
output will say).
