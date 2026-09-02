# Resumable In-Process Eval Runner (subscription-CLI arm) — Design Spec

Date: 2026-08-31
Status: Approved for implementation
Scope: A host-side script that runs an eval set × arms paired comparison
in-process (no Celery/Redis), able to **pause on a Claude Code usage-limit
signal and resume later** once the subscription window resets. Plus the
adapter change that makes the limit distinguishable, and an `arms.yaml`
entry for the hosted arm.

## Why

`docs/ARCHITECTURE.md` differentiator #4 (local-vs-hosted cost/latency/quality
frontier) and Phase 7's Deliverable still has no hosted-API arm that has
ever completed a paired run — the metered `gemini-flash` leg 429'd 130/150
calls (`docs/RESULTS.md`, "The open gap"). The `claude_code_cli` adapter
(spec `2026-08-27-subscription-cli-adapters-design.md`) gives a
hosted-quality arm with **no per-token bill** under a Claude Pro/Max seat —
the cheapest path to that missing number.

The obstacle: a Claude subscription seat has a rolling usage limit
(~5-hour window, plus a weekly cap). A 150-example paired run against
`claude-code-sonnet` pays ~25k cache-creation tokens per call (measured
against the live CLI) and will very likely hit the limit mid-run. The
existing Celery worker treats a limit as a rate-limit error: 6 long-backoff
retries (cap 90s) then a persisted `failed` row. A multi-hour reset
outlasts that — the run would bleed failures instead of stopping cleanly.

## Non-goals

- No Celery/Redis path. This runner is deliberately in-process, matching
  `backend/scripts/serial_judge_run.py` (added for the same 7.8 GB WSL OOM
  constraint). The Celery `subscription_cli` queue is unchanged.
- No run-level `paused` state in the DB / API / dashboard. "Paused" is
  simply "the runner exited; cells without a `completed` RunResult remain".
- No auto-resume (cron/beat). The operator re-runs the script after the
  window resets; the script prints when.
- No change to stats, calibration, frontier, or the `run` API endpoints —
  the runner writes the same `RunResult` rows they already read.
- Codex CLI. Same pattern would apply; not in this pass.

## Components

### 1. `ClaudeCodeCLIAdapter` — usage-limit detection

`backend/app/adapters/claude_code_cli.py`.

New exception in the adapter module:

```python
class UsageLimitError(RuntimeError):
    """The Claude Code subscription seat hit its usage limit. Distinct from
    an ordinary failure: the caller should stop and retry after `retry_at`,
    not record a failed result."""
    def __init__(self, message: str, retry_at: datetime | None = None):
        super().__init__(message)
        self.retry_at = retry_at
```

It subclasses `RuntimeError` so the existing worker `is_retryable` /
`is_rate_limited` classification (which matches on `"usage limit"` in the
message) keeps working unchanged if an arm ever runs via Celery.

`generate()` raises `UsageLimitError` instead of a plain `RuntimeError`
when any of these hold (checked before the generic non-zero-exit raise):

1. `stdout` (or the parsed `result` field) contains `usage limit reached`
   (case-insensitive). The CLI emits `Claude AI usage limit reached|<unix>`
   — the trailing `|<digits>` is parsed into `retry_at`
   (`datetime.fromtimestamp(int(...), tz=timezone.utc)`); absent/malformed
   → `retry_at=None`.
2. The parsed JSON has `api_error_status` whose string form contains
   `rate_limit` or equals a 429, **or** `subtype` contains `limit`.
3. `stderr` (lower-cased) contains any of: `usage limit`, `rate limit`,
   `too many requests`, `5-hour limit`, `weekly limit`,
   `resource_exhausted`.

A non-zero exit with **empty** stderr and no JSON is *not* by itself
treated as a usage limit here (the worker's existing heuristic is for
choosing a backoff, not for a multi-hour pause) — it raises the generic
`RuntimeError` as today and the runner records it as a normal cell failure.
Rationale: a false "paused" stops the whole run on what might be a
transient blip; a false "failed" costs one cell that `--resume` re-attempts
anyway.

The exact live shape is verified at implementation time against a real
limited call if one occurs; the three-way check above is defensive against
all documented forms. Unit tests mock `subprocess.run` for each form, same
style as the existing adapter tests.

### 2. `serial_eval_run.py` — the runner

`backend/scripts/serial_eval_run.py`. Mirrors `serial_judge_run.py`: argparse
CLI, `asyncio.run(main(...))`, talks to Postgres via
`app.db.session.engine`, loads arms via `app.config.arms.load_arms` and the
task via `app.config.tasks.load_task`. Needs only Postgres reachable at
`DATABASE_URL` and each arm's backend reachable (Ollama for the local arm,
an authenticated `claude` CLI for the hosted arm).

**Two entry modes:**

```
uv run python -m scripts.serial_eval_run new \
    --arms qwen3-8b-local,claude-code-sonnet \
    --task financial_sentiment --sample-size 150 --seed 20260831 [--repeats 1]

uv run python -m scripts.serial_eval_run resume <run_id>

uv run python -m scripts.serial_eval_run new --arms ... --dry-run   # print
    # the resolved arms + cell plan and exit, no Run row, no calls

uv run python -m scripts.serial_eval_run resume <run_id> --max-cli-calls 30
    # do at most 30 subscription-CLI calls this invocation, then stop
    # cleanly (exit 0) — batches the CLI phase so it never drains the seat's
    # usage window in one go. Also accepted on `new`.
```

**`--max-cli-calls N`** (optional, both subcommands): caps *successful*
subscription-CLI calls per invocation. On reaching the cap the runner stops
exactly like a usage-limit pause (`Outcome.reason == "cli_batch_limit"`,
exit 0, resume message) — no row is skipped or lost. Local-arm cells are
never capped. Use it to leave headroom on a Claude seat that is **also
driving an interactive session** (the operator's own `claude`): run the CLI
phase as a few bounded batches instead of 150 back-to-back calls.

- `new`: samples examples exactly as `POST /runs` does — filter
  `EvalExample.source == task_cfg.source`, `ORDER BY id`,
  `random.Random(seed).sample(...)` (seed defaults to a random 2**31 int,
  echoed to stdout). Inserts a `Run` row (`task`, `arm_names`,
  `sample_size`, `repeats`, `seed`, `total_calls = n_examples × n_arms ×
  repeats`). Prints the new `run_id`. Then runs the work loop.
- `resume <run_id>`: loads the `Run`, re-derives the *same* example set
  from its stored `seed` + `sample_size` + `task`, and runs the work loop.
  422-style error to stderr + exit 2 if the run id is unknown.

**Work loop.** Build the full cell list = `chosen_examples × arm_names ×
range(repeats)`. Load the set of `(example_id, arm_name, repeat_index)`
that already have a `RunResult` with `status='completed'` for this run.
Process only the missing cells, **local arms first, then CLI arms** (so a
mid-run pause only ever leaves CLI cells outstanding, never a half-done
local arm). "CLI arm" = `getattr(arm.adapter, "celery_queue", None) ==
"subscription_cli"`; everything else sorts first. For each cell:

- Call `arm.adapter.generate(arm.render(example_text))` directly — no retry
  wrapper (the runner is interactive; the operator re-runs). One
  exception: ordinary transient errors get a single short retry (2s) to
  paper over a blip, matching how forgiving `serial_judge_run` is by just
  moving on.
- On success: insert a `RunResult` (`status='completed'`, all the
  `ModelResponse` fields), same mapping as `worker._persist_run_result`.
- On `UsageLimitError`: **do not insert a row.** Print
  `PAUSED: usage limit hit after {done}/{total} cells.` plus, if
  `retry_at` is set, `Resets at {retry_at:%Y-%m-%d %H:%M %Z} (~{delta}).`
  and always `Resume with:  uv run python -m scripts.serial_eval_run
  resume {run_id}`. Exit **0** (a clean, expected stop — not a failure).
- On any other exception: insert a `RunResult` with `status='failed'` and
  `error_message=str(exc)`, print a one-line `[i/total] ... FAILED`, and
  continue. `--resume` re-attempts failed cells too (the "already done"
  set is `status='completed'` only).

Progress line every 10 cells: `{i}/{total}  ok={ok} fail={fail}`.

On completion (no cells left): print
`done: run {run_id}  ok={ok} fail={fail} — now judge with:  uv run python
-m scripts.serial_judge_run {run_id}`.

**Idempotency / re-run safety:** only `status='completed'` cells are
skipped, and a completed cell is never re-called. Running `new` twice makes
two runs (two `Run` rows) — expected. Running `resume` on a finished run is
a no-op that prints `done:`.

### 3. `arms.yaml`

Add, uncommented, next to the commented subscription block:

```yaml
  - name: claude-code-sonnet
    adapter: claude_code_cli
    model: sonnet
```

`prompt_template` unset → the arm uses the active task's `eval_prompt`
(financial sentiment's default), identical to what `qwen3-8b-local` gets —
a fair paired prompt.

The commented `claude-code-sonnet-subscription` example block stays as-is
(it documents the Celery-queue path); a short comment points at this
runner for the no-Celery path.

## Data flow

```
serial_eval_run new
  └─ sample examples (seeded)  ─► INSERT Run
  └─ for cell in local-arm cells:  adapter.generate ─► INSERT RunResult(completed)
  └─ for cell in cli-arm cells:
        adapter.generate
          ├─ ok             ─► INSERT RunResult(completed)
          ├─ UsageLimitError ─► print resume cmd, exit 0   (no row)
          └─ other error    ─► INSERT RunResult(failed), continue
  └─ done ─► print "judge with serial_judge_run <run_id>"

serial_judge_run <run_id>   (unchanged)  ─► UPDATE RunResult.judge_*

GET /runs/<run_id>/compare|equivalence|summary   (unchanged)  ◄─ reads RunResult
```

## Error handling

| Condition | Behaviour |
|---|---|
| Claude usage limit | `UsageLimitError` → no row, print resume cmd + reset time, exit 0 |
| Claude not authenticated | adapter raises `RuntimeError("...is not authenticated")` → runner prints a fix hint (`claude login`) and exits 2 (whole run is blocked, not one cell) |
| Claude CLI missing | `RuntimeError("...not found on PATH")` → same as above, exit 2 |
| Ollama down / local arm error | one 2s retry, then `RunResult(failed)`, continue |
| Single CLI call error (non-limit) | `RunResult(failed)`, continue; `--resume` re-attempts |
| Unknown run id (resume) | stderr message, exit 2 |
| Postgres unreachable | exception propagates, non-zero exit (same as `serial_judge_run`) |

"Not authenticated" and "binary missing" are checked on the **first** CLI
cell and abort the CLI phase immediately rather than failing 150 cells one
by one; local-arm results already persisted stay valid for a later resume.

## Testing

Unit (pytest, no network, no DB — mock `subprocess.run` / inject a fake
adapter):

- `test_claude_code_cli.py`: `UsageLimitError` raised for (a) `result`
  text `Claude AI usage limit reached|1893456000` with `retry_at` parsed,
  (b) `api_error_status` containing `rate_limit`, (c) stderr `usage limit`;
  and *not* raised for a bare non-zero exit with empty stderr (stays
  `RuntimeError`).
- `serial_eval_run` loop logic extracted to a pure function
  `plan_cells(chosen, arm_names, repeats, completed_set)` → unit-tested for
  skip-completed, local-before-cli ordering, repeats expansion.
- A loop test with a fake adapter that raises `UsageLimitError` on the 3rd
  CLI cell: asserts no row written for that cell, exit code 0, resume
  message printed, and that a second pass with the same fake (now
  succeeding) completes the remaining cells and writes no duplicate for the
  already-completed ones. Uses the test DB like the existing
  `tests/api/test_runs.py` helpers.

Real end-to-end: none automated (needs a live subscription seat). The
operator runs it for the actual Deliverable; a `--dry-run` flag prints the
cell plan and the resolved arms without calling anything, for a smoke
check.

## Deliverable

After the run completes (across as many resume cycles as the limit
forces), and `serial_judge_run` has judged it:

- `GET /runs/<id>/compare?metric=judge_score` and `.../equivalence` give
  the paired local-vs-hosted result.
- A short writeup at `docs/reports/2026-09-01-local-vs-cli-hosted.md`
  and a line in `docs/RESULTS.md` closing "The open gap".
- Judge caveat noted: the judge is local `qwen3:8b`, the same base model
  as the `qwen3-8b-local` arm — self-preference could *understate* the
  hosted arm; flagged, not corrected (consistent with prior runs, see
  `backend/README.md` "Watch for judge/arm model overlap").
