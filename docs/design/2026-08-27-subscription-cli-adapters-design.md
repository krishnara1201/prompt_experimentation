# Subscription-CLI Adapters — Design Spec

Date: 2026-08-27
Status: Approved for implementation
Scope: Extends the model adapter layer of the platform described
in `docs/ARCHITECTURE.md` — adds a third category of arm alongside "local weights" and
"pay-per-token API": a subscription-seat CLI, driven non-interactively, with
no per-token price.

## Context

`docs/ARCHITECTURE.md`'s differentiator #4 is a cost/latency/quality frontier, not a
single leaderboard number. Today every arm is either free-to-run local
compute (Ollama) or a metered, per-token-priced API call
(`OpenAICompatibleAdapter`, `AnthropicAdapter`). There is a third real-world
option this project's audience will recognize: driving a model through a
CLI tool that's included in a flat-rate subscription seat — Claude Code
under a Claude Pro/Max plan, or OpenAI's Codex CLI under a ChatGPT
Plus/Pro plan. These have no per-call price at all, so they need a distinct
accounting treatment, not just a new adapter.

This spec covers exactly two new adapters. A third CLI (e.g. Gemini CLI)
should be addable later as a config entry plus a new adapter class,
following the same pattern — no changes to the shared protocol or the
orchestrator should be required.

## Decisions

These were confirmed with the user before writing this spec:

1. **Scope**: Claude Code CLI (`claude`) and OpenAI Codex CLI (`codex`)
   only, for this pass.
2. **Cost accounting**: `cost_estimate_usd` is always `None` for these
   arms. Reporting `$0` would misrepresent a seat that isn't actually free;
   reporting an amortized cost requires an assumed monthly call volume the
   project can't defend. `None` is the honest signal that this arm's
   cost model is fundamentally different from a metered one — the
   dashboard/frontier plot (phase 5, not built yet) will need to treat
   these as a separate category rather than plotting a misleading `$0`
   point next to metered arms. Not designed further here.
3. **Tool access**: both CLIs run with their default agentic behavior
   (tools enabled), rather than being forced into a "plain chat response"
   mode. The comparison is "how would someone actually get a sentiment
   judgment out of this tool," including whatever tool use the model
   chooses to do.
4. **Working directory**: because tool use stays enabled, each call still
   runs from a **fresh, empty scratch directory**, never this repo's
   working directory. Running hundreds of unattended agentic calls with
   file-editing tools enabled against the real codebase is an unacceptable
   blast radius, and letting the CLI read this repo's own files would
   bias every response. An empty directory keeps tools "on" while giving
   them nothing real to read or write.
5. **Concurrency**: these arms get a lower concurrency cap than API arms,
   enforced via Celery queue routing (see below), because a subprocess-based
   CLI call is heavier than an HTTP call and a subscription session may not
   tolerate the same parallelism the orchestrator already uses for API arms.

## Architecture

Two new adapter classes, not one generic one — consistent with the existing
adapter layer, where a new class exists per distinct request/response
schema rather than per provider:

- **`ClaudeCodeCLIAdapter`** (`backend/app/adapters/claude_code_cli.py`)
- **`CodexCLIAdapter`** (`backend/app/adapters/codex_cli.py`)

Both implement the existing `ModelAdapter` protocol unchanged:

```python
class ModelAdapter(Protocol):
    def generate(self, prompt: str) -> ModelResponse: ...
```

No changes to `ModelResponse` or the protocol itself — `cost_estimate_usd:
float | None` already accommodates "unknown/not applicable."

### Invocation

Per call, each adapter:

1. Creates a fresh directory via `tempfile.mkdtemp()`.
2. Runs its CLI as a subprocess with `cwd` set to that directory, JSON
   output requested, and a permission/approval mode that cannot block
   waiting on a human (the job runs unattended under a Celery worker).
3. Parses the JSON result into a `ModelResponse`.
4. Removes the scratch directory in a `finally` block, whether the call
   succeeded or failed.

**Claude Code** (`claude` CLI, confirmed installed and inspected via
`claude --help`):

```
claude -p "<prompt>" --output-format json --model <model> \
    --dangerously-skip-permissions
```

`--dangerously-skip-permissions` is required because tools stay enabled
(decision 3) and there is no human present to approve a tool call in a
batch job (decision-context, not a separate decision — this is the
mechanical consequence of 3+4 together). Its blast radius is the empty
scratch directory from decision 4, which has nothing in it to damage and
no path back to this repo. `--model` accepts either a full model name or
an alias (`sonnet`, `opus`, `haiku`); `arms.yaml` can use either.

Claude Code's JSON output includes (per current CLI documentation; verify
the exact field names against a live `claude -p ... --output-format json`
call during implementation, since this is external-tool surface that can
change): a `result` field with the final text, `is_error`, `duration_ms`,
and a `usage` object with `input_tokens`/`output_tokens`. These map
directly to `ModelResponse.text`, `.prompt_tokens`, `.completion_tokens`.
`latency_ms` is measured by the adapter itself (wall-clock around the
subprocess call) rather than trusted from the CLI's own timing field, for
consistency with how the existing HTTP adapters measure latency.

**Codex CLI**: not installed in this environment, so its exact flags
(non-interactive exec subcommand, JSON output flag, sandbox/approval-mode
flags, working-directory flag) must be confirmed against `codex --help`
and `codex exec --help` at implementation time. The shape is expected to
mirror Claude Code's: a non-interactive `exec`-style invocation, a flag
disabling interactive approval prompts (the Codex equivalent of
`--dangerously-skip-permissions`), a model flag, and a JSON output mode.
Implementation must re-verify these against the real CLI rather than
assuming the names above; this is the one open item this spec does not
fully close, tracked as a verification step in the implementation plan
rather than a design ambiguity.

### Response mapping and errors

- `cost_estimate_usd`: always `None` (decision 2).
- `prompt_tokens` / `completion_tokens`: whatever the CLI's JSON usage
  reports; informational only, since it doesn't drive cost here.
- `finish_reason`: mapped from the CLI's own status field when one exists
  (e.g. Claude Code's `is_error`/`subtype`), else `None`.
- Non-zero exit code, a JSON parse failure, or an error indicated in the
  parsed JSON all raise `RuntimeError` with a message identifying which
  CLI and what failed — matching how the existing adapters signal failure
  today.
- A distinguishable "not authenticated" condition (e.g. the CLI reports
  the user isn't logged in) raises `RuntimeError` with a message containing
  a fixed, greppable substring (mirroring the existing "No API key found in
  environment variable" pattern used for the same purpose). `is_retryable`
  in `backend/app/tasks/worker.py` gets one more string check alongside the
  existing API-key check, so an unauthenticated CLI fails fast instead of
  burning three retries with backoff sleeps.

### Configuration

No new fields on `ModelResponse` or the protocol. New `arms.yaml` entries
name one of the two new adapter types and a model; no `api_key_env`,
because these adapters manage no credentials at all:

```yaml
  - name: claude-code-sonnet-subscription
    adapter: claude_code_cli
    model: sonnet

  - name: codex-subscription
    adapter: codex_cli
    model: gpt-5-codex   # verify current model name against `codex --help`
```

**Precondition, not managed by this code**: the machine running the Celery
worker must already have an authenticated CLI session (`claude login` /
`codex login`) under the operator's subscription seat. The adapter never
reads or stores credentials — it shells out to whatever session already
exists, exactly as a human running that CLI interactively would.

`ADAPTER_TYPES` in `backend/app/config/arms.py` gains two entries:
`"claude_code_cli": ClaudeCodeCLIAdapter` and `"codex_cli": CodexCLIAdapter`.
No other change to the config loader — it already handles arbitrary
adapter-specific constructor kwargs generically.

### Concurrency: dedicated Celery queue

No new locking or coordination infrastructure. Both new adapter classes set
an instance attribute `self.celery_queue = "subscription_cli"` in
`__init__`; the existing adapters set nothing, so they keep the default
queue.

In `backend/app/api/routes/runs.py`, `create_run`'s enqueue loop changes
from:

```python
run_single_call.delay(run_id=..., example_id=..., ...)
```

to routing per arm:

```python
queue = getattr(available_arms[arm_name], "celery_queue", "celery")
run_single_call.apply_async(kwargs={...}, queue=queue)
```

This requires zero changes to `OpenAICompatibleAdapter` or
`AnthropicAdapter`. The operator runs a second Celery worker process
consuming only the `subscription_cli` queue at low concurrency, e.g.:

```
celery -A app.tasks.worker worker -Q subscription_cli --concurrency=1
```

alongside the existing worker consuming the default queue at its current
concurrency. This is a deployment/README instruction, not something
enforced in application code — documented in `backend/README.md` (or
equivalent) as part of implementation.

## Testing

- **Unit tests**, both adapters: mock `subprocess.run` (or the relevant
  `subprocess` call) to return canned JSON for the success path, a
  non-zero exit for the failure path, and an "unauthenticated" message for
  the non-retryable path — proving the JSON-to-`ModelResponse` mapping and
  error classification without needing a live subscription or spawning a
  real CLI process.
- **Real end-to-end test for Claude Code**, matching the existing pattern
  for the Ollama arm: skipped automatically (not failed) when `claude` is
  not on `PATH` or not authenticated, so CI and machines without a Claude
  subscription still pass the suite. Verifies the real JSON schema
  described above actually matches what's assumed.
- **Real end-to-end test for Codex**: same pattern, skipped when `codex`
  is unavailable or unauthenticated. Since `codex` isn't installed in this
  development environment, this test is written against the verified CLI
  behavior at implementation time and will simply skip here until it's
  available in an environment that has it.
- Scratch-directory cleanup (the `finally` removing the temp dir) is
  covered by a unit test asserting the directory is gone after both a
  successful and a failing call.

## Out of scope

- Dashboard/frontier-plot treatment of a flat-rate arm (phase 5, not built
  yet) — noted in decision 2 as a forward-looking implication only.
- A third subscription CLI (e.g. Gemini CLI) — the pattern established
  here (new adapter class, new `ADAPTER_TYPES` entry, `celery_queue`
  attribute if it needs the same concurrency treatment) should make this a
  small addition later, not a redesign.
- Any change to `ModelAdapter`, `ModelResponse`, or the existing
  `OpenAICompatibleAdapter`/`AnthropicAdapter` adapters.
- Preflight validation of CLI authentication before a run starts (e.g. a
  health-check endpoint). Errors surface per-call, same as a missing API
  key does today for the existing adapters.
