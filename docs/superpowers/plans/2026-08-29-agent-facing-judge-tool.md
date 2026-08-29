# Agent-Facing Judge Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing judge layer (`app/judge/scorer.py:score_output`) as a single MCP tool, `score_financial_sentiment`, so a coding agent (e.g. a Claude Code session) can score one candidate financial-sentiment response against a gold label directly, without running a full eval.

**Architecture:** A new module `backend/app/mcp_judge_server.py` wraps `score_output` behind an `mcp.server.mcpserver.MCPServer` instance served over stdio. A private, testable `_score_financial_sentiment` function holds the actual logic (config load + judge call); the public `@mcp.tool()`-decorated `score_financial_sentiment` is a one-line delegator. A repo-root `.mcp.json` registers the server for any Claude Code session opened in this repo.

**Tech Stack:** Python 3.12, the official `mcp` SDK (`mcp>=2.1.1`, `MCPServer` class — the 2.x successor to `FastMCP`), `uv`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-29-agent-facing-judge-tool-design.md`

**Status:** ✅ Complete (all tasks committed). Post-implementation follow-ups
also landed: `judge_model` added to the tool response for provenance, input
validation (`gold_label` domain, non-blank `input_text`/`model_output`), and
expanded `backend/README.md` Phase 6 notes (judge prerequisite, calibration
pointer). See the spec's 2026-08-29 amendment.

## Global Constraints

- Rubric stays fixed (financial-sentiment only) — no arbitrary/custom rubric per call.
- No DB persistence of ad-hoc calls — stateless.
- No calibration status/metadata in the tool response — score + rationale only.
- No retry/backoff in the tool — single-shot call, errors propagate to become MCP tool-call errors.
- `mcp` dependency must be `>=2.1.1` and imported as `from mcp.server.mcpserver import MCPServer` — **not** `mcp.server.fastmcp.FastMCP`, which is mcp 1.x and deliberately raises `ModuleNotFoundError` under `mcp>=2` (verified against the installed `mcp==2.1.1` package source).
- Config loading follows the existing repo pattern: `ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"`, reloaded via `load_judge_arm(str(ARMS_PATH))` on every call (not cached at startup).

---

### Task 1: Core scoring function (`_score_financial_sentiment`)

**Files:**
- Create: `backend/app/mcp_judge_server.py`
- Test: `backend/tests/test_mcp_judge_server.py`

**Interfaces:**
- Consumes: `app.adapters.base.ModelAdapter` (protocol), `app.config.arms.load_judge_arm(config_path: str) -> ModelAdapter`, `app.judge.scorer.score_output(adapter, input_text, gold_label, model_output) -> JudgeResult` (`JudgeResult.score: int`, `JudgeResult.rationale: str`), `app.judge.scorer.JudgeParseError`.
- Produces: `app.mcp_judge_server.ARMS_PATH: Path`, `app.mcp_judge_server._score_financial_sentiment(input_text: str, gold_label: str, model_output: str, adapter: ModelAdapter | None = None) -> dict` returning `{"score": int, "rationale": str}`. Task 2 builds on this exact name and signature.

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_mcp_judge_server.py`:

```python
import pytest

from app.adapters.base import ModelResponse
from app.judge.scorer import JudgeParseError
from app.mcp_judge_server import ARMS_PATH, _score_financial_sentiment


class _FakeAdapter:
    def __init__(self, text):
        self._text = text

    def generate(self, prompt):
        return ModelResponse(text=self._text, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)


def test_score_financial_sentiment_returns_score_and_rationale():
    adapter = _FakeAdapter("SCORE: 4\nRATIONALE: Correct sentiment, slightly terse.")

    result = _score_financial_sentiment(
        "Profits rose sharply.", "positive", "The tone is positive.", adapter=adapter
    )

    assert result == {"score": 4, "rationale": "Correct sentiment, slightly terse."}


def test_score_financial_sentiment_raises_on_malformed_judge_output():
    adapter = _FakeAdapter("not in the expected format")

    with pytest.raises(JudgeParseError):
        _score_financial_sentiment("Profits rose.", "positive", "Bullish.", adapter=adapter)


def test_score_financial_sentiment_loads_judge_arm_when_no_adapter_given(monkeypatch):
    fake_adapter = _FakeAdapter("SCORE: 5\nRATIONALE: Nailed it.")
    calls = []

    def fake_load_judge_arm(config_path):
        calls.append(config_path)
        return fake_adapter

    monkeypatch.setattr("app.mcp_judge_server.load_judge_arm", fake_load_judge_arm)

    result = _score_financial_sentiment("Profits rose.", "positive", "Bullish outlook.")

    assert result == {"score": 5, "rationale": "Nailed it."}
    assert calls == [str(ARMS_PATH)]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'app.mcp_judge_server'` (the module doesn't exist yet).

- [x] **Step 3: Implement `_score_financial_sentiment`**

Create `backend/app/mcp_judge_server.py`:

```python
from pathlib import Path

from app.adapters.base import ModelAdapter
from app.config.arms import load_judge_arm
from app.judge.scorer import score_output

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"


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
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py -v`
Expected: PASS (3 passed)

- [x] **Step 5: Commit**

```bash
git add backend/app/mcp_judge_server.py backend/tests/test_mcp_judge_server.py
git commit -m "feat: add core scoring function for agent-facing judge tool"
```

---

### Task 2: Wire the MCP tool and server entrypoint

**Files:**
- Modify: `backend/pyproject.toml` (add `mcp>=2.1.1` to `dependencies`)
- Modify: `backend/app/mcp_judge_server.py` (add `MCPServer` instance, decorated tool, entrypoint)
- Modify: `backend/tests/test_mcp_judge_server.py` (add tests for the decorated tool)

**Interfaces:**
- Consumes: Task 1's `_score_financial_sentiment(input_text, gold_label, model_output, adapter=None) -> dict` and `ARMS_PATH`.
- Produces: `app.mcp_judge_server.mcp: MCPServer` (module-level instance, name `"financial-sentiment-judge"`), `app.mcp_judge_server.score_financial_sentiment(input_text: str, gold_label: str, model_output: str) -> dict` (the MCP-registered tool, directly callable as a plain function), `app.mcp_judge_server.main() -> None`.

- [x] **Step 1: Add the `mcp` dependency**

Edit `backend/pyproject.toml` — in the `dependencies` list, add a new line after `"pymc>=6.3.1",`:

```toml
    "pymc>=6.3.1",
    "mcp>=2.1.1",
]
```

Run: `cd backend && uv sync`
Expected: resolves and installs `mcp` and its transitive dependencies with no errors.

- [x] **Step 2: Write the failing test for the decorated tool**

Append to `backend/tests/test_mcp_judge_server.py`:

```python
def test_score_financial_sentiment_tool_is_directly_callable(monkeypatch):
    fake_adapter = _FakeAdapter("SCORE: 3\nRATIONALE: Hedged but on-topic.")
    monkeypatch.setattr("app.mcp_judge_server.load_judge_arm", lambda config_path: fake_adapter)

    from app.mcp_judge_server import score_financial_sentiment

    result = score_financial_sentiment("Revenue was flat.", "neutral", "Results were mixed.")

    assert result == {"score": 3, "rationale": "Hedged but on-topic."}


def test_mcp_server_instance_has_expected_name():
    from app.mcp_judge_server import mcp

    assert mcp.name == "financial-sentiment-judge"
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_financial_sentiment' from 'app.mcp_judge_server'` (and same for `mcp`).

- [x] **Step 4: Implement the MCP wiring**

Modify `backend/app/mcp_judge_server.py` — add these imports at the top (after the existing `pathlib` import) and this code at the end of the file:

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

(This replaces the whole file from Task 1 — the only change is inserting the `MCPServer` import/instance before `_score_financial_sentiment` and adding the decorated tool + entrypoint after it.)

- [x] **Step 5: Run the tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py -v`
Expected: PASS (5 passed)

- [x] **Step 6: Manual sanity check that the server object constructs and the module runs as an entrypoint**

Run: `cd backend && uv run python -c "from app.mcp_judge_server import mcp; print(mcp.name)"`
Expected output: `financial-sentiment-judge`

Run: `cd backend && timeout 3 uv run python -m app.mcp_judge_server; echo "exit code: $?"`
Expected: the process starts and blocks reading stdio (no crash/traceback); `timeout` kills it after 3s, printing `exit code: 124`. A traceback or immediate non-124 exit means the wiring is broken — stop and fix before continuing.

- [x] **Step 7: Run the full backend test suite to confirm no regressions**

Run: `cd backend && uv run pytest -v`
Expected: all tests pass (or skip, for the Ollama/Postgres tests that skip when those services aren't running) — no new failures.

- [x] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/mcp_judge_server.py backend/tests/test_mcp_judge_server.py
git commit -m "feat: register score_financial_sentiment as an MCP tool"
```

---

### Task 3: Discovery config and documentation

**Files:**
- Create: `.mcp.json` (repo root)
- Modify: `backend/README.md` (insert a new `## Phase 6: Agent-facing judge tool` section immediately before the existing `## Tests` section)
- Modify: `CLAUDE.md` (mark build phase 6 as done)

**Interfaces:**
- Consumes: nothing new — references the `uv run --directory backend python -m app.mcp_judge_server` command and `score_financial_sentiment` tool name from Task 2.
- Produces: nothing consumed by later tasks — this is the terminal task.

- [x] **Step 1: Create the repo-root `.mcp.json`**

Create `.mcp.json`:

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

- [x] **Step 2: Verify the JSON is well-formed**

Run: `python3 -c "import json; json.load(open('.mcp.json')); print('valid')"`
Expected output: `valid`

- [x] **Step 3: Add the Phase 6 section to `backend/README.md`**

In `backend/README.md`, insert the following section immediately before the line `## Tests` (i.e. right after the Phase 3 section's "Watch for judge/arm model overlap" bullet, which is currently the last content before `## Tests`):

````markdown
## Phase 6: Agent-facing judge tool

Exposes the judge layer (Phase 3) as an MCP tool so a coding agent (e.g. a
Claude Code session) can score one candidate financial-sentiment response
against a gold label directly — no eval run, no Celery, no Postgres.

The repo-root `.mcp.json` registers this automatically for any Claude Code
session opened in this repo (the standard project-scoped-server approval
prompt applies). To use it from another MCP client, or to run it directly:

```bash
uv run --directory backend python -m app.mcp_judge_server
```

Tool: `score_financial_sentiment(input_text, gold_label, model_output) ->
{"score": 1-5, "rationale": str}`, using the same fixed rubric and `judge:`
config in `arms.yaml` as the automated pipeline. The judge config is
reloaded on every call, so editing `arms.yaml`'s `judge:` block takes
effect on the next call with no restart.

Ad-hoc calls are not persisted — this is for disposable checks during
iteration, not part of the auditable run history.
````

- [x] **Step 4: Mark Phase 6 done in `CLAUDE.md`**

In `CLAUDE.md`'s "Build phases" section, replace:

```markdown
6. **Agent-facing judge tool** — expose the judge layer (Phase 3) as a
   callable tool so local coding agents (e.g. Claude Code sessions) can use
   this platform's calibrated rubric judge directly, independent of the
   eval-run pipeline. Deliberately designed after Phase 3 lands, against its
   actual interface rather than a guess at one.
```

with:

```markdown
6. **Agent-facing judge tool** ✅ **Done.** MCP server
   (`backend/app/mcp_judge_server.py`) exposes a single
   `score_financial_sentiment` tool wrapping the existing
   `judge/scorer.py:score_output`, so a Claude Code session (or any other
   MCP client) can score a candidate response against a gold label
   directly, without running a full eval. Discovered automatically via the
   repo-root `.mcp.json`. Spec:
   `docs/superpowers/specs/2026-08-29-agent-facing-judge-tool-design.md`.
```

- [x] **Step 5: Commit**

```bash
git add .mcp.json backend/README.md CLAUDE.md
git commit -m "docs: document and register the agent-facing judge tool"
```
