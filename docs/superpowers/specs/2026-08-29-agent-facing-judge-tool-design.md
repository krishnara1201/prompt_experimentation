# Agent-Facing Judge Tool — Design Spec

Date: 2026-08-29
Status: Approved for implementation
Scope: Build phase 6 of the platform described in `CLAUDE.md` — expose the
existing judge layer (phase 3) as a callable tool so local coding agents
(e.g. Claude Code sessions) can score a candidate financial-sentiment
response directly, independent of the eval-run pipeline.

## Context

Phase 3 (`docs/superpowers/specs/2026-08-27-judge-layer-calibration-design.md`)
built `score_output(adapter, input_text, gold_label, model_output) ->
JudgeResult` in `backend/app/judge/scorer.py` — already a pure function with
no dependency on Celery or Postgres. That spec deliberately deferred "expose
judging as a callable tool for external/local agents" to a later phase, to
be designed against a judge that already existed rather than guessed at in
parallel. This spec is that later phase.

The goal is narrow: let a coding agent iterating on a prompt or model arm
get a quick, calibrated quality score for one candidate response, without
running a full eval (seed examples, Celery orchestration, `RunResult` rows).

## Architecture

```
[Claude Code session] --MCP stdio--> [app/mcp_judge_server.py]
                                          -> load_judge_arm(arms.yaml)
                                          -> score_output(adapter, input_text, gold_label, model_output)
                                          -> {score, rationale}
```

A new module, `backend/app/mcp_judge_server.py`, wraps `score_output` as a
single MCP tool served over stdio using the `mcp` Python SDK's
`MCPServer` (`mcp.server.mcpserver.MCPServer` — the current SDK's
tool-server class; `FastMCP` was the pre-2.0 name, and `mcp.server.fastmcp`
deliberately raises `ModuleNotFoundError` in `mcp>=2` pointing at this
rename — verified against the installed `mcp==2.1.1` package source).
No new judging logic is introduced — this is glue over the existing pure
function, not a new scoring path. It is deliberately **not** a FastAPI
endpoint, a CLI script, or a DB-backed feature; see "Rejected alternatives"
below for why.

### Tool signature

```python
score_financial_sentiment(input_text: str, gold_label: str, model_output: str) -> dict
# -> {"score": int (1-5), "rationale": str, "judge_model": str}
```

**Amendment (2026-08-29, post-implementation):** the response also carries
`judge_model` — the `model` field of the loaded judge adapter (e.g.
`"opus"`, `"qwen3:8b"`). This is call *provenance*, not calibration status:
because the judge config is reloaded per call and can be tuned between
calls, an agent (or a saved transcript) otherwise has no record of which
model produced a given score. `gold_label` outside
`{"positive","negative","neutral"}` and blank `input_text`/`model_output`
now raise `ValueError` before the judge call rather than being passed
through.

The rubric stays fixed — the same `rubric.py` template used by the
automated pipeline, reference-guided against `gold_label`. The tool is
scoped to financial-sentiment grading, matching `CLAUDE.md`'s non-goal of
not being a general-purpose LLM chat/eval tool. The tool's docstring states
this scope explicitly so a calling agent understands what it's for.

The response is silent on calibration status (score + rationale only, no
calibration metadata) — consistent with the platform's existing
report-don't-gate philosophy: calibration is a deliberate, separate check
(`calibration_report.py`) run against real eval runs, not something an
ad-hoc single call should annotate or gate on.

### Config loading

Follows the same pattern already used in `worker.py`, `demo.py`, and
`api/routes/runs.py`:

```python
ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"
```

`load_judge_arm(str(ARMS_PATH))` is called **fresh on every tool call**, not
cached at server startup — mirroring how `run_judge_call` in `worker.py`
already reloads it per task. This means editing the `judge:` block in
`arms.yaml` (e.g. swapping judge model) takes effect on the next call with
no server restart, which matters for interactive iteration where the judge
config itself might be what's being tuned.

### Module structure and testability

```python
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from app.adapters.base import ModelAdapter
from app.config.arms import load_judge_arm
from app.judge.scorer import score_output

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"

mcp = MCPServer("financial-sentiment-judge")


def _score_financial_sentiment(
    input_text: str,
    gold_label: str,
    model_output: str,
    adapter: ModelAdapter | None = None,
) -> dict:
    if adapter is None:
        adapter = load_judge_arm(str(ARMS_PATH))
    result = score_output(adapter, input_text, gold_label, model_output)
    return {"score": result.score, "rationale": result.rationale}


@mcp.tool()
def score_financial_sentiment(input_text: str, gold_label: str, model_output: str) -> dict:
    """Score a candidate financial-sentiment response (1-5) against a gold
    label, using this platform's calibrated rubric judge."""
    return _score_financial_sentiment(input_text, gold_label, model_output)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

`mcp = MCPServer("financial-sentiment-judge")` is a module-level instance;
`@mcp.tool()` registers `score_financial_sentiment` as a side effect and
returns the function unchanged (confirmed in the SDK source — the
decorator calls `self.add_tool(fn, ...)` then `return fn`), so it remains
directly callable as a plain function, including from tests. `mcp.run()`
defaults to `transport="stdio"`.

The private `_score_financial_sentiment` takes an optional `adapter` so
tests can inject a fake adapter directly (same fake-adapter pattern already
used in `tests/judge/test_scorer.py`), without needing a real `arms.yaml`
judge config or a real model call. The public, MCP-decorated function has
no test-only parameters — it's the thin entry point `MCPServer` registers.

### Distribution / discovery

A project-level `.mcp.json` is added at the repo root:

```json
{
  "mcpServers": {
    "financial-sentiment-judge": {
      "command": "uv",
      "args": ["run", "--directory", "backend", "python", "-m", "app.mcp_judge_server"]
    }
  }
}
```

So any Claude Code session opened in this repo can discover and (after the
standard project-scoped-server approval prompt) use the tool with no manual
`claude mcp add` step.

### Dependency

Adds `mcp>=2.1.1` (the official MCP Python SDK, providing `MCPServer`) to
`backend/pyproject.toml`'s `dependencies`.

## Error handling

No retry/backoff logic — unlike `run_judge_call`'s Celery-task retry loop,
this is a single-shot interactive call with no run/DB state to keep
consistent across attempts. `JudgeParseError` (malformed rubric output) or
any adapter exception (network error, missing API key, etc.) propagates
out of `_score_financial_sentiment`/`score_financial_sentiment` uncaught.
`MCPServer`'s `_handle_call_tool` catches any exception raised by a tool
function and returns it as a `CallToolResult(is_error=True)` (confirmed in
the SDK source — it never lets a tool exception crash the server process),
surfacing the error directly to the calling agent, which can decide whether
to retry, rephrase, or give up — the same judgment call a human would make,
just left to the agent instead of hidden behind automatic retries.

## Testing

New file `backend/tests/test_mcp_judge_server.py`:

- `_score_financial_sentiment` with a fake adapter returning a well-formed
  `SCORE:`/`RATIONALE:` response returns the expected `{"score", "rationale"}`
  dict.
- `_score_financial_sentiment` with a fake adapter returning malformed
  output raises `JudgeParseError` (propagates, uncaught).
- `_score_financial_sentiment` with `adapter=None` calls `load_judge_arm`
  with `ARMS_PATH` (verifies the config-loading wiring, e.g. via monkeypatch
  on `load_judge_arm`).

No test drives the actual MCP/stdio protocol layer — `MCPServer`'s tool
registration and transport are the SDK's responsibility, not this project's;
testing stops at the plain-Python function boundary.

## Out of scope

- Arbitrary/custom rubrics per call (rejected in brainstorming — keeps this
  tool's output backed by the same rubric the calibration work already
  validated, and keeps the project scoped to financial-sentiment grading
  per `CLAUDE.md`'s non-goals).
- Persisting ad-hoc calls to Postgres (rejected — this tool is for
  disposable checks during iteration, not part of the auditable
  `RunResult`/judge-score history; adding a log table would couple a
  pure-function tool to the DB for no current consumer).
- Surfacing calibration status/metadata in the tool response (rejected —
  consistent with the platform's report-don't-gate philosophy; calibration
  is checked deliberately via `calibration_report.py`, not auto-attached to
  every ad-hoc score).
- A FastAPI HTTP endpoint or standalone CLI script for the same
  functionality — MCP is the only interface this phase builds (see
  "Rejected alternatives").
- Retry/backoff on judge call failures — left to the calling agent.

## Rejected alternatives

- **FastAPI endpoint** (`POST /judge/score`) — consistent with the rest of
  the platform's HTTP API and reusable by the frontend later, but a coding
  agent would need to know to `curl` it and the API server would need to be
  running just for a one-shot check. MCP gives a native tool call inside
  the agent's own session instead.
- **CLI script** (`uv run python -m app.judge_cli ...`) — simple and
  dependency-light, but every calling agent needs to know the argv shape
  and parse stdout JSON itself, versus a typed tool call with a schema the
  agent already understands natively via MCP.

Both remain easy to add later on top of the same `score_output` function if
a concrete consumer needs them; nothing in this design forecloses that.
