# Judge Layer & Calibration — Design Spec

Date: 2026-08-27
Status: Approved for implementation
Scope: Build phase 3 of the platform described in `CLAUDE.md` — a rubric-
based LLM-as-judge that scores every completed `RunResult`, plus a
calibration workflow that checks judge/human agreement on a held-out sample
before judge scores are trusted on a full run.

## Context

Phases 1 (model adapter layer) and 2 (Celery orchestration + FastAPI run
endpoints) are done. `RunResult` rows already carry a `judge_score: float |
None` column, anticipating this phase, but nothing populates it yet.

This is deliberately scoped to the judge itself. A separate, later spec will
cover exposing judging as a callable tool for external/local agents (see
`CLAUDE.md`'s "Build phases" — added there as a phase following this one) —
that interface should be designed against a judge that already exists, not
guessed at in parallel.

## Architecture

```
[RunResult, status=completed] -> [judge/scorer.py: adapter.generate(rubric prompt)]
                                -> [judge_score, judge_rationale on RunResult]

[stratified sample of judged RunResults] -> [hand-labeled JSON]
                                          -> [judge_calibration_label table]
                                          -> [calibration_report.py: Spearman + Cohen's kappa]
```

### Data model

New Alembic migration adds:

- On `run_result`: `judge_rationale` (text, nullable), `judge_status`
  (`pending` / `completed` / `failed`, default `pending`),
  `judge_error_message` (text, nullable), `judge_celery_task_id` (text,
  nullable). `judge_score` already exists.
- New table `judge_calibration_label`: `id` (PK), `run_result_id` (FK ->
  `run_result.id`), `human_score` (int, 1-5), `labeled_by` (text), `notes`
  (text, nullable), `labeled_at` (timestamp, default now).

Judging stays 1:1 with a `RunResult`, mirroring how generation itself is
tracked — no separate `judge_result` table, since a `RunResult` is never
judged more than once by the automated pipeline (calibration labels are a
distinct, human-sourced table, never conflated with `judge_score`).

### Judge model configuration

`arms.yaml` gets a new top-level `judge:` key, same field shape as an arm
entry (`adapter`, `model`, `api_key_env`, etc.), loaded via a new
`load_judge_arm(config_path) -> ModelAdapter` in `app/config/arms.py`,
alongside the existing `load_arms`. Kept out of the `arms:` list so it can
never be accidentally selected as an eval arm in a run, and so the judge
model can be swapped independently of what's under test.

### Judge module (`backend/app/judge/`)

- **`rubric.py`** — a fixed, reference-guided prompt template (fills
  `{input_text}`, `{gold_label}`, `{model_output}`):

  ```
  You are grading a financial-sentiment model's response.

  Input text: {input_text}
  Correct sentiment: {gold_label}
  Model's response: {model_output}

  Score the response 1-5:
  5 = correctly identifies the sentiment as {gold_label}, clearly and directly
  4 = correctly identifies the sentiment, but with minor clarity/formatting issues
  3 = ambiguous, hedged, or only partially matches the correct sentiment
  2 = identifies the wrong sentiment but the response is otherwise coherent/on-topic
  1 = wrong sentiment, off-topic, malformed, or non-responsive

  Respond in exactly this format:
  SCORE: <1-5>
  RATIONALE: <one sentence>
  ```

  Reference-guided (the judge is shown the correct answer) rather than
  reference-free, since Financial PhraseBank provides an objective gold
  label to grade against — more reliable than asking the judge to assess
  quality blind.

- **`scorer.py`** — `score_output(adapter: ModelAdapter, input_text: str,
  gold_label: str, model_output: str) -> JudgeResult` where `JudgeResult =
  (score: int, rationale: str)`. Calls `adapter.generate(prompt)` and parses
  the fixed `SCORE:`/`RATIONALE:` format. A response that doesn't match the
  format raises `JudgeParseError` — treated as non-retryable in the Celery
  task, since a fixed prompt producing malformed output won't fix itself on
  retry (same reasoning as the existing `is_retryable` logic for bad API
  keys).

No changes to the `ModelAdapter` protocol or existing adapters — the judge
is just another `generate(prompt)` call, reusing `ModelResponse` for
latency/token/cost tracking on the judge call itself (stored via the new
`judge_*` columns, not the generation columns).

### Task flow (`app/tasks/worker.py`)

New Celery task `run_judge_call(run_result_id: int)`. In `execute_call`,
immediately after a successful `_persist_run_result(status="completed",
...)`, the task enqueues `run_judge_call.delay(run_result_id)` — mirroring
the existing chain-of-tasks pattern. `run_judge_call`:

1. Loads the `RunResult` and its parent `EvalExample` (for `gold_label`).
2. Resolves the judge adapter via `load_judge_arm`.
3. Calls `score_output`, using the same retry/backoff loop and
   `is_retryable` classification already used for generation calls.
4. On success: sets `judge_score`, `judge_rationale`, `judge_status =
   "completed"`.
5. On failure: sets `judge_status = "failed"`, `judge_error_message`.

A failed judge call never changes the `RunResult.status` set by generation —
a bad judge call must not make a successful generation look failed, and vice
versa: judging is only ever attempted on `status = "completed"` results.

### Calibration workflow

Three scripts under `backend/scripts/`, following the existing
`seed_eval_examples.py` idiom (idempotent, CLI-driven):

1. **`select_calibration_sample.py --run-id X --n 40 --out FILE`** — pulls
   judged `RunResult`s for the run, stratifies by `(arm_name, gold_label)`
   so every arm and every sentiment class is represented, writes a JSON
   array of `{run_result_id, arm_name, input_text, gold_label, model_output,
   judge_score, judge_rationale, human_score: null}`.
2. **`import_calibration_labels.py --in FILE`** — validates every row has
   `human_score` filled in (int 1-5), upserts into `judge_calibration_label`
   keyed on `run_result_id` (re-running updates rather than duplicating).
3. **`calibration_report.py --run-id X`** — joins `run_result.judge_score`
   against `judge_calibration_label.human_score` for that run, computes:
   - **Spearman correlation** (ordinal 1-5 scores; via `scipy.stats
     .spearmanr`, already a project dependency per `CLAUDE.md`'s stats
     stack)
   - **Cohen's kappa** on binarized correct/incorrect (`score >= 4`),
     implemented as a small manual helper in `app/judge/calibration.py`
     rather than adding sklearn as a dependency for one metric
   - n, mean absolute difference

   Prints a report to stdout and returns a dict (for testing). No new table
   stores the computed report — it's cheap to recompute on demand from
   `judge_calibration_label` + `run_result`, and there's exactly one
   consumer (you, reading it before trusting the run's judge scores) until
   the dashboard phase.

This is the artifact that satisfies the CLAUDE.md differentiator "judge
calibration is reported, not assumed" — judge scores on a full run are only
trustworthy once this report has been run and reviewed for that run (or a
representative run using the same judge config).

## Error handling

- Judge parse failures are non-retryable (malformed rubric-format output
  won't self-correct on retry) — same `is_retryable`-style classification
  used for generation, extended to cover `JudgeParseError`.
- Judge HTTP/network errors (rate limits, 5xx, timeouts) reuse the existing
  retryable classification unchanged.
- `import_calibration_labels.py` fails loudly (non-zero exit, no partial
  import) if any row is missing `human_score` — a half-labeled calibration
  set would silently understate the sample size in the report.

## Testing

- `judge/rubric.py` — prompt renders all three placeholders correctly.
- `judge/scorer.py` — parses well-formed `SCORE:`/`RATIONALE:` output;
  raises `JudgeParseError` on malformed output (missing score, out-of-range
  score, missing rationale).
- `tasks/worker.py` — `run_judge_call` persists score/rationale on success;
  persists `judge_status="failed"` + error message on judge failure without
  touching `RunResult.status`; retries on retryable errors, doesn't on
  `JudgeParseError`.
- `config/arms.py` — `load_judge_arm` parses the `judge:` key; errors if
  absent or malformed, mirroring `load_arms`'s error handling.
- `judge/calibration.py` — Spearman correlation and Cohen's kappa computed
  correctly against known small fixtures (e.g. perfect agreement -> kappa
  1.0; anti-correlated scores -> negative correlation).
- `scripts/select_calibration_sample.py` — stratifies across
  `(arm_name, gold_label)` combinations present in a run; respects `--n`.
- `scripts/import_calibration_labels.py` — upserts (idempotent re-run);
  rejects a file with any missing `human_score`.

## Out of scope

- Exposing judge scoring as a tool other agents/processes can call directly
  (planned as a later phase in `CLAUDE.md`, designed once this judge exists).
- Any gating logic that blocks a run or its stats from proceeding based on
  calibration results — calibration is reported for you to read and decide,
  not automatically enforced.
- Multi-rater calibration (more than one human labeler, inter-rater
  agreement) — out of scope for a single-user portfolio project.
- Judge score persistence for the calibration *report itself* (recomputed
  on demand, not stored).
