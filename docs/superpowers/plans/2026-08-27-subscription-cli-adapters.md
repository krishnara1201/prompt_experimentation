# Subscription-CLI Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new `ModelAdapter` implementations — `ClaudeCodeCLIAdapter` and `CodexCLIAdapter` — so `arms.yaml` can declare arms backed by a subscription-seat CLI (Claude Code under a Claude Pro/Max plan, Codex CLI under a ChatGPT Plus/Pro plan) instead of a metered, per-token API key.

**Architecture:** Each adapter shells out to its CLI non-interactively from a fresh, empty scratch directory per call, with tool use left enabled but permission prompts disabled (since there's no human present in a batch job). Both report `cost_estimate_usd=None` — there is no per-call price for a flat-rate seat. Both set `self.celery_queue = "subscription_cli"` so the orchestrator can route them to a lower-concurrency Celery queue without touching the existing HTTP-based adapters.

**Tech Stack:** Python stdlib `subprocess` and `tempfile` (no new dependencies); existing `pytest`/`monkeypatch` test patterns already used for the HTTP adapters.

**Spec:** `docs/superpowers/specs/2026-08-27-subscription-cli-adapters-design.md`

## Global Constraints

- `cost_estimate_usd` is always `None` for both new adapters — never `0` and never an amortized estimate (spec decision 2).
- Tool use stays enabled on both CLIs; each call must still run with `cwd` set to a **fresh, empty** directory created via `tempfile.TemporaryDirectory()`, cleaned up whether the call succeeds or fails (spec decisions 3–4).
- Neither adapter manages credentials. Both assume the CLI is already authenticated on the host machine via the operator's subscription (`claude login` / `codex login`) — the adapter only shells out.
- A non-zero exit whose stderr indicates the CLI isn't logged in must raise `RuntimeError` whose message contains the literal substring `is not authenticated`, so `worker.py`'s `is_retryable` can fail fast instead of retrying 3 times (spec, "Response mapping and errors").
- Claude Code's CLI flags and JSON schema (`--output-format json`) are based on `claude --help`, inspected directly on this machine during brainstorming — treat them as verified. Codex CLI is **not installed** in this environment; its flags in this plan are best-effort and each Codex step says explicitly where to double-check against `codex --help` / `codex exec --help` before trusting the literal flag list.
- New adapter classes only. No changes to `ModelAdapter`, `ModelResponse`, `OpenAICompatibleAdapter`, or `AnthropicAdapter`.

---

## Task 1: `ClaudeCodeCLIAdapter`

**Files:**
- Create: `backend/app/adapters/claude_code_cli.py`
- Test: `backend/tests/adapters/test_claude_code_cli.py`
- Test: `backend/tests/adapters/test_claude_code_cli_e2e.py`

**Interfaces:**
- Consumes: `ModelResponse` from `app.adapters.base` (unchanged: `text`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `cost_estimate_usd`, `finish_reason`).
- Produces: `ClaudeCodeCLIAdapter(model: str, binary: str = "claude", timeout: float = 120.0)` with `.generate(prompt: str) -> ModelResponse`, matching the `ModelAdapter` protocol. Instance attribute `self.celery_queue = "subscription_cli"` (consumed by Task 5). Instance attribute `self.model` (consumed by Task 3's config passthrough test).

- [ ] **Step 1: Write the failing unit tests**

Create `backend/tests/adapters/test_claude_code_cli.py`:

```python
import json
import os
import subprocess

import pytest

from app.adapters import claude_code_cli
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_generate_returns_text_and_token_counts(monkeypatch):
    payload = {
        "result": "positive",
        "is_error": False,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(stdout=json.dumps(payload)),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    response = adapter.generate("Classify sentiment: Profits rose sharply.")

    assert response.text == "positive"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 3
    assert response.cost_estimate_usd is None
    assert response.latency_ms > 0


def test_generate_passes_expected_cli_flags(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _completed(stdout=json.dumps({"result": "ok", "usage": {}}))

    monkeypatch.setattr(claude_code_cli.subprocess, "run", fake_run)
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    adapter.generate("hello")

    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "hello" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert "--dangerously-skip-permissions" in cmd


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Something went wrong"),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(RuntimeError, match="Something went wrong"):
        adapter.generate("hello")


def test_generate_raises_distinguishable_error_when_not_authenticated(monkeypatch):
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: not logged in. Run `claude login`."),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(RuntimeError, match="is not authenticated"):
        adapter.generate("hello")


def test_generate_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(stdout="not json"),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(RuntimeError, match="non-JSON"):
        adapter.generate("hello")


def test_generate_raises_when_binary_missing(monkeypatch):
    def fake_run(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(claude_code_cli.subprocess, "run", fake_run)
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(RuntimeError, match="not found on PATH"):
        adapter.generate("hello")


def test_generate_uses_fresh_scratch_directory_and_cleans_up(monkeypatch):
    seen_dirs = []

    def fake_run(cmd, cwd, **kwargs):
        seen_dirs.append(cwd)
        assert os.path.isdir(cwd)
        assert os.listdir(cwd) == []
        return _completed(stdout=json.dumps({"result": "ok", "usage": {}}))

    monkeypatch.setattr(claude_code_cli.subprocess, "run", fake_run)
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    adapter.generate("hello")
    adapter.generate("hello again")

    assert len(seen_dirs) == 2
    assert seen_dirs[0] != seen_dirs[1]
    for d in seen_dirs:
        assert not os.path.exists(d)


def test_celery_queue_is_subscription_cli():
    adapter = ClaudeCodeCLIAdapter(model="sonnet")
    assert adapter.celery_queue == "subscription_cli"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/adapters/test_claude_code_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.claude_code_cli'`

- [ ] **Step 3: Implement `ClaudeCodeCLIAdapter`**

Create `backend/app/adapters/claude_code_cli.py`:

```python
import json
import subprocess
import tempfile
import time

from app.adapters.base import ModelResponse


class ClaudeCodeCLIAdapter:
    def __init__(
        self,
        model: str,
        binary: str = "claude",
        timeout: float = 120.0,
    ):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.celery_queue = "subscription_cli"

    def generate(self, prompt: str) -> ModelResponse:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--dangerously-skip-permissions",
        ]

        with tempfile.TemporaryDirectory(prefix="claude-code-cli-arm-") as scratch_dir:
            start = time.perf_counter()
            try:
                result = subprocess.run(
                    cmd,
                    cwd=scratch_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Claude Code CLI binary '{self.binary}' not found on PATH"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _looks_unauthenticated(stderr):
                raise RuntimeError(f"Claude Code CLI is not authenticated: {stderr}")
            raise RuntimeError(f"Claude Code CLI exited with {result.returncode}: {stderr}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude Code CLI returned non-JSON output: {result.stdout[:500]!r}"
            ) from exc

        if data.get("is_error"):
            raise RuntimeError(f"Claude Code CLI reported an error: {data.get('result')}")

        usage = data.get("usage", {})
        return ModelResponse(
            text=data.get("result", ""),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            cost_estimate_usd=None,
            finish_reason=data.get("subtype"),
        )


def _looks_unauthenticated(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        phrase in lowered
        for phrase in ("not logged in", "not authenticated", "please log in", "claude login")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/adapters/test_claude_code_cli.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Write the real end-to-end test (skips if `claude` isn't on PATH)**

Create `backend/tests/adapters/test_claude_code_cli_e2e.py`:

```python
import shutil

import pytest

from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")
def test_claude_code_cli_generates_nonempty_text():
    adapter = ClaudeCodeCLIAdapter(model="haiku")

    response = adapter.generate("Reply with the single word: hello")

    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.latency_ms > 0
    assert response.cost_estimate_usd is None
```

- [ ] **Step 6: Run the e2e test**

Run: `cd backend && uv run pytest tests/adapters/test_claude_code_cli_e2e.py -v`
Expected: PASS if `claude` is installed and authenticated on this machine (it is, per brainstorming); otherwise SKIPPED, never FAILED.

- [ ] **Step 7: Manually verify the "not authenticated" heuristic against real output (non-destructive)**

Do **not** log the real session out. Instead, point Claude Code at an empty, throwaway config directory to simulate a logged-out state without touching real credentials:

```bash
mkdir -p /tmp/claude-empty-config
CLAUDE_CONFIG_DIR=/tmp/claude-empty-config claude -p "hello" --output-format json
```

Inspect the printed stderr/exit behavior. If the real wording doesn't contain any of `"not logged in"`, `"not authenticated"`, `"please log in"`, `"claude login"` (case-insensitive), update the phrase list in `_looks_unauthenticated` in `backend/app/adapters/claude_code_cli.py` to match, then re-run Step 4's test command to confirm nothing broke.

- [ ] **Step 8: Commit**

```bash
cd backend && git add app/adapters/claude_code_cli.py tests/adapters/test_claude_code_cli.py tests/adapters/test_claude_code_cli_e2e.py
git commit -m "feat: add ClaudeCodeCLIAdapter for subscription-seat Claude Code arms"
```

---

## Task 2: `CodexCLIAdapter`

**Files:**
- Create: `backend/app/adapters/codex_cli.py`
- Test: `backend/tests/adapters/test_codex_cli.py`
- Test: `backend/tests/adapters/test_codex_cli_e2e.py`

**Interfaces:**
- Consumes: `ModelResponse` from `app.adapters.base` (same as Task 1).
- Produces: `CodexCLIAdapter(model: str, binary: str = "codex", timeout: float = 120.0)` with `.generate(prompt: str) -> ModelResponse`. Instance attributes `self.celery_queue = "subscription_cli"` and `self.model` (same contract as Task 1's adapter, consumed by Tasks 3 and 5).

**Note before starting:** `codex` is not installed in this development environment, so its exact CLI flags could not be verified against `codex --help` during brainstorming (unlike Claude Code, which was inspected directly). This task therefore keeps the CLI surface deliberately small: plain stdout capture instead of a guessed JSON event-stream schema, so `prompt_tokens`/`completion_tokens` are `0` rather than parsed from an unverified format. If `codex` is available wherever this task is implemented, run `codex --help` and `codex exec --help` first and adjust the flag list in Step 3 to match reality before writing the implementation — the flags below (`exec`, `--model`, `--sandbox workspace-write`, `--ask-for-approval never`, `--cd`) are the plan's best-effort placeholder for "non-interactive, sandboxed to the scratch dir, no approval prompts," not a verified contract.

- [ ] **Step 1: Write the failing unit tests**

Create `backend/tests/adapters/test_codex_cli.py`:

```python
import os
import subprocess

import pytest

from app.adapters import codex_cli
from app.adapters.codex_cli import CodexCLIAdapter


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["codex"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_generate_returns_stdout_as_text(monkeypatch):
    monkeypatch.setattr(codex_cli.subprocess, "run", lambda *a, **k: _completed(stdout="positive\n"))
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    response = adapter.generate("Classify sentiment: Profits rose sharply.")

    assert response.text == "positive"
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.cost_estimate_usd is None
    assert response.latency_ms > 0


def test_generate_passes_expected_cli_flags(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _completed(stdout="ok")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    adapter.generate("hello")

    cmd = captured["cmd"]
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "hello" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-5-codex"
    assert cmd[cmd.index("--ask-for-approval") + 1] == "never"
    assert "--sandbox" in cmd


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(codex_cli.subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="boom"))
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    with pytest.raises(RuntimeError, match="boom"):
        adapter.generate("hello")


def test_generate_raises_distinguishable_error_when_not_authenticated(monkeypatch):
    monkeypatch.setattr(
        codex_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: not logged in. Run `codex login`."),
    )
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    with pytest.raises(RuntimeError, match="is not authenticated"):
        adapter.generate("hello")


def test_generate_raises_when_binary_missing(monkeypatch):
    def fake_run(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    with pytest.raises(RuntimeError, match="not found on PATH"):
        adapter.generate("hello")


def test_generate_uses_fresh_scratch_directory_and_cleans_up(monkeypatch):
    seen_dirs = []

    def fake_run(cmd, cwd, **kwargs):
        seen_dirs.append(cwd)
        assert os.path.isdir(cwd)
        return _completed(stdout="ok")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    adapter.generate("hello")

    assert len(seen_dirs) == 1
    assert not os.path.exists(seen_dirs[0])


def test_celery_queue_is_subscription_cli():
    adapter = CodexCLIAdapter(model="gpt-5-codex")
    assert adapter.celery_queue == "subscription_cli"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/adapters/test_codex_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.codex_cli'`

- [ ] **Step 3: Implement `CodexCLIAdapter`**

Create `backend/app/adapters/codex_cli.py`:

```python
import subprocess
import tempfile
import time

from app.adapters.base import ModelResponse


class CodexCLIAdapter:
    def __init__(
        self,
        model: str,
        binary: str = "codex",
        timeout: float = 120.0,
    ):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.celery_queue = "subscription_cli"

    def generate(self, prompt: str) -> ModelResponse:
        with tempfile.TemporaryDirectory(prefix="codex-cli-arm-") as scratch_dir:
            cmd = [
                self.binary,
                "exec",
                prompt,
                "--model",
                self.model,
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "--cd",
                scratch_dir,
            ]
            start = time.perf_counter()
            try:
                result = subprocess.run(
                    cmd,
                    cwd=scratch_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Codex CLI binary '{self.binary}' not found on PATH"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _looks_unauthenticated(stderr):
                raise RuntimeError(f"Codex CLI is not authenticated: {stderr}")
            raise RuntimeError(f"Codex CLI exited with {result.returncode}: {stderr}")

        return ModelResponse(
            text=result.stdout.strip(),
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cost_estimate_usd=None,
            finish_reason=None,
        )


def _looks_unauthenticated(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        phrase in lowered
        for phrase in ("not logged in", "not authenticated", "please log in", "codex login")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/adapters/test_codex_cli.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write the real end-to-end test (skips if `codex` isn't on PATH)**

Create `backend/tests/adapters/test_codex_cli_e2e.py`:

```python
import shutil

import pytest

from app.adapters.codex_cli import CodexCLIAdapter


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex CLI not on PATH")
def test_codex_cli_generates_nonempty_text():
    adapter = CodexCLIAdapter(model="gpt-5-codex")

    response = adapter.generate("Reply with the single word: hello")

    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.latency_ms > 0
    assert response.cost_estimate_usd is None
```

- [ ] **Step 6: Run the e2e test**

Run: `cd backend && uv run pytest tests/adapters/test_codex_cli_e2e.py -v`
Expected: SKIPPED in this environment (`codex` not installed). If implementing on a machine with `codex` installed and authenticated: PASS, and if it fails, revisit the flags in Step 3 against real `codex --help` output first.

- [ ] **Step 7: Commit**

```bash
cd backend && git add app/adapters/codex_cli.py tests/adapters/test_codex_cli.py tests/adapters/test_codex_cli_e2e.py
git commit -m "feat: add CodexCLIAdapter for subscription-seat Codex CLI arms"
```

---

## Task 3: Register both adapters in the config loader

**Files:**
- Modify: `backend/app/config/arms.py`
- Test: `backend/tests/config/test_arms.py`

**Interfaces:**
- Consumes: `ClaudeCodeCLIAdapter` from `app.adapters.claude_code_cli` (Task 1), `CodexCLIAdapter` from `app.adapters.codex_cli` (Task 2).
- Produces: `ADAPTER_TYPES` dict gains `"claude_code_cli"` and `"codex_cli"` keys, so `load_arms`/`load_judge_arm` (unchanged function bodies) can resolve them from `arms.yaml`.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/config/test_arms.py`, extend `VALID_CONFIG` with two new arms and update the assertions that iterate over it:

```python
VALID_CONFIG = """
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b

  - name: gpt-4o-mini
    adapter: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    price_per_1m_input: 0.15
    price_per_1m_output: 0.60

  - name: claude-haiku
    adapter: anthropic
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
    price_per_1m_input: 1.00
    price_per_1m_output: 5.00

  - name: claude-code-sonnet-subscription
    adapter: claude_code_cli
    model: sonnet

  - name: codex-subscription
    adapter: codex_cli
    model: gpt-5-codex
"""
```

Update the two tests that assert over all arm names/types:

```python
def test_load_arms_builds_correct_adapter_types(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert set(arms.keys()) == {
        "qwen3-8b-local",
        "gpt-4o-mini",
        "claude-haiku",
        "claude-code-sonnet-subscription",
        "codex-subscription",
    }
    assert isinstance(arms["qwen3-8b-local"], OpenAICompatibleAdapter)
    assert isinstance(arms["gpt-4o-mini"], OpenAICompatibleAdapter)
    assert isinstance(arms["claude-haiku"], AnthropicAdapter)
    assert isinstance(arms["claude-code-sonnet-subscription"], ClaudeCodeCLIAdapter)
    assert isinstance(arms["codex-subscription"], CodexCLIAdapter)
```

Add a new passthrough test and a queue-attribute test:

```python
def test_load_arms_passes_config_fields_through_for_subscription_cli_arms(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].model == "sonnet"
    assert arms["codex-subscription"].model == "gpt-5-codex"


def test_subscription_cli_arms_route_to_dedicated_celery_queue(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].celery_queue == "subscription_cli"
    assert arms["codex-subscription"].celery_queue == "subscription_cli"
```

Add the two new imports at the top of the file:

```python
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from app.adapters.codex_cli import CodexCLIAdapter
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: FAIL — `set(arms.keys())` assertion fails with `UnknownAdapterError` not raised... actually `load_arms` raises `UnknownAdapterError: Unknown adapter type 'claude_code_cli'`.

- [ ] **Step 3: Register the new adapter types**

In `backend/app/config/arms.py`, add the imports and extend `ADAPTER_TYPES`:

```python
from app.adapters.anthropic import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from app.adapters.codex_cli import CodexCLIAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
```

```python
ADAPTER_TYPES = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "claude_code_cli": ClaudeCodeCLIAdapter,
    "codex_cli": CodexCLIAdapter,
}
```

No other change — `load_arms`/`load_judge_arm` already handle arbitrary adapter kwargs generically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/config/arms.py tests/config/test_arms.py
git commit -m "feat: register claude_code_cli and codex_cli adapter types"
```

---

## Task 4: Fail fast on an unauthenticated subscription CLI

**Files:**
- Modify: `backend/app/tasks/worker.py`
- Test: `backend/tests/tasks/test_execute_call.py`

**Interfaces:**
- Consumes: nothing new — this task only recognizes a message shape both Task 1 and Task 2's adapters already raise (`RuntimeError` containing `"is not authenticated"`).
- Produces: `is_retryable(exc: Exception) -> bool` returns `False` for that shape, in addition to its existing checks.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/tasks/test_execute_call.py`, add a case to the existing `test_is_retryable_classification` parametrize list:

```python
@pytest.mark.parametrize(
    "exc, retryable",
    [
        (RuntimeError("No API key found in environment variable 'OPENAI_API_KEY'"), False),
        (RuntimeError("Claude Code CLI is not authenticated: not logged in"), False),
        (RuntimeError("Codex CLI is not authenticated: please run codex login"), False),
        (_http_status_error(400), False),
        (_http_status_error(401), False),
        (_http_status_error(404), False),
        (_http_status_error(429), True),
        (_http_status_error(500), True),
        (_http_status_error(503), True),
        (httpx.ConnectError("network down"), True),
        (httpx.ReadTimeout("slow"), True),
        (RuntimeError("transient blip"), True),
        (JudgeParseError("bad format"), False),
    ],
)
def test_is_retryable_classification(exc, retryable):
    assert worker.is_retryable(exc) is retryable
```

Add a dedicated end-to-end-through-`execute_call` test, mirroring `test_does_not_retry_missing_api_key`:

```python
def test_does_not_retry_unauthenticated_subscription_cli(monkeypatch):
    exc = RuntimeError("Claude Code CLI is not authenticated: not logged in")
    adapter = FakeAdapter([exc] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1  # no retries, no backoff sleep
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "is not authenticated" in kwargs["error_message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v -k "authenticated"`
Expected: FAIL — the two new authentication cases are currently classified `retryable=True`.

- [ ] **Step 3: Update `is_retryable`**

In `backend/app/tasks/worker.py`, update the function and its docstring:

```python
def is_retryable(exc: Exception) -> bool:
    """False for errors that cannot possibly succeed on a retry.

    A missing API key raises RuntimeError from the adapters
    (`app/adapters/openai_compatible.py`, `app/adapters/anthropic.py`), and
    a 4xx response raises httpx.HTTPStatusError via raise_for_status(). Both
    are permanent — retrying only burns backoff sleep. 429 is the exception:
    rate limiting is transient, so it stays retryable, as do network errors,
    timeouts and 5xx. An unauthenticated subscription CLI
    (`app/adapters/claude_code_cli.py`, `app/adapters/codex_cli.py`) is the
    same kind of permanent failure as a missing API key.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return True
        return not (400 <= status < 500)
    if isinstance(exc, RuntimeError) and "No API key found in environment variable" in str(exc):
        return False
    if isinstance(exc, RuntimeError) and "is not authenticated" in str(exc):
        return False
    if isinstance(exc, JudgeParseError):
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/tasks/worker.py tests/tasks/test_execute_call.py
git commit -m "fix: fail fast on an unauthenticated subscription-CLI arm instead of retrying"
```

---

## Task 5: Route subscription-CLI arms to a dedicated Celery queue

**Files:**
- Modify: `backend/app/api/routes/runs.py`
- Modify: `backend/tests/api/test_runs.py`

**Interfaces:**
- Consumes: `getattr(adapter, "celery_queue", "celery")` — every adapter from Tasks 1–2 has `.celery_queue == "subscription_cli"`; `OpenAICompatibleAdapter`/`AnthropicAdapter` have no such attribute, so they fall back to Celery's own default queue name, `"celery"`.
- Produces: `run_single_call.apply_async(kwargs={...}, queue=...)` replacing the previous `run_single_call.delay(...)` call in `create_run`'s enqueue loop.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/api/test_runs.py`, update the two tests that reference `.delay` to reference `.apply_async` instead:

```python
@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_enqueues_expected_number_of_calls(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        response = TestClient(app).post("/runs", json={"repeats": 2, "sample_size": 1, "seed": 1})
        assert response.status_code == 200
        body = response.json()
        run_id = body["run_id"]
        assert body["status"] == "pending"
        assert body["total_calls"] == 2  # 1 example x 1 arm x 2 repeats
        assert mock_task.apply_async.call_count == 2
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)
```

```python
@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_deletes_run_row_when_enqueue_fails(mock_load_arms, mock_task):
    mock_task.apply_async.side_effect = RuntimeError("redis is down")
    example_id = _insert_example()
    run_id_before = _latest_run_id()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/runs", json={"repeats": 1, "sample_size": 1, "seed": 1})
        assert response.status_code >= 500

        # No orphaned Run row survives the failed enqueue.
        assert _latest_run_id() == run_id_before
    finally:
        _delete_example(example_id)
```

Add a new test proving per-arm queue routing, near the other `FAKE_ARMS`-based tests:

```python
class _FakeSubscriptionAdapter:
    celery_queue = "subscription_cli"


FAKE_ARMS_MIXED = {"fake-arm": object(), "fake-subscription-arm": _FakeSubscriptionAdapter()}


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS_MIXED)
def test_create_run_routes_arms_to_correct_queue(mock_load_arms, mock_task):
    example_id = _insert_example()
    run_id = None
    try:
        response = TestClient(app).post(
            "/runs",
            json={"arms": ["fake-arm", "fake-subscription-arm"], "repeats": 1, "sample_size": 1, "seed": 1},
        )
        assert response.status_code == 200
        run_id = response.json()["run_id"]

        queues_used = {call.kwargs["queue"] for call in mock_task.apply_async.call_args_list}
        assert queues_used == {"celery", "subscription_cli"}
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v -k "queue or enqueues_expected or enqueue_fails"`
Expected: FAIL — `mock_task.apply_async` has 0 calls because `runs.py` still calls `.delay(...)`.

- [ ] **Step 3: Route by adapter's `celery_queue`**

In `backend/app/api/routes/runs.py`, replace the `_enqueue_all` function body:

```python
    def _enqueue_all() -> None:
        for example_id, example_text in chosen:
            for arm_name in arm_names:
                queue = getattr(available_arms[arm_name], "celery_queue", "celery")
                for repeat_index in range(payload.repeats):
                    run_single_call.apply_async(
                        kwargs={
                            "run_id": run_id,
                            "example_id": example_id,
                            "example_text": example_text,
                            "arm_name": arm_name,
                            "repeat_index": repeat_index,
                        },
                        queue=queue,
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: PASS (all tests in the file; requires Postgres reachable per the module's existing `pytestmark` skip condition)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/api/routes/runs.py tests/api/test_runs.py
git commit -m "feat: route subscription-CLI arms to a dedicated low-concurrency Celery queue"
```

---

## Task 6: Example config and operator docs

**Files:**
- Modify: `backend/arms.yaml`
- Modify: `backend/README.md`

**Interfaces:**
- Consumes: `adapter: claude_code_cli` / `adapter: codex_cli`, both registered in Task 3.
- Produces: no code interface — this task is config + documentation only.

- [ ] **Step 1: Add example arms to `arms.yaml`**

Append to `backend/arms.yaml` (after the existing `claude-haiku` arm, before the `judge:` section):

```yaml
  - name: claude-code-sonnet-subscription
    adapter: claude_code_cli
    model: sonnet

  - name: codex-subscription
    adapter: codex_cli
    model: gpt-5-codex
```

These have no `price_per_1m_input`/`price_per_1m_output` fields — `ClaudeCodeCLIAdapter`/`CodexCLIAdapter` don't accept them (they always report `cost_estimate_usd=None`), so adding pricing fields here would raise `InvalidArmConfigError` on load.

- [ ] **Step 2: Document the new arm type and the second worker process in `backend/README.md`**

Add a new subsection after "## Add or swap an arm":

```markdown
## Subscription-seat CLI arms (Claude Code, Codex)

`adapter: claude_code_cli` and `adapter: codex_cli` drive the `claude` and
`codex` CLIs directly, non-interactively, instead of calling a metered API.
Unlike every other arm type, these have no per-call price — `arms.yaml`
should not set `price_per_1m_input`/`price_per_1m_output` on them, and
`cost_estimate_usd` on their results is always `null`.

**Precondition**: the machine running the Celery worker must already have
an authenticated CLI session under your subscription — run `claude login`
or `codex login` yourself first. Neither adapter reads or stores
credentials; they only shell out to whatever session already exists.

**Tool use stays on.** Each call still runs from a fresh, empty scratch
directory (created and torn down per call), so neither CLI can read this
repo's `CLAUDE.md`/`AGENTS.md` or touch real files, but within that empty
directory the CLI is free to use tools exactly as it would interactively.

**Run a second, low-concurrency worker for these arms** — a CLI subprocess
call is heavier than an HTTP call, and a subscription session may not
tolerate the same parallelism as the API arms:

```bash
uv run celery -A app.tasks.worker.celery_app worker -Q subscription_cli --concurrency=1 --loglevel=info
```

Keep your existing worker command (`-Q celery`, or no `-Q` flag, which
defaults to the same queue) running alongside it — one worker per queue,
both pointed at the same Redis broker.
```

- [ ] **Step 3: Verify the config still loads**

Run: `cd backend && uv run python -c "from app.config.arms import load_arms; print(sorted(load_arms('arms.yaml').keys()))"`
Expected: prints a list including `'claude-code-sonnet-subscription'` and `'codex-subscription'` alongside the existing three arm names, with no exception.

- [ ] **Step 4: Commit**

```bash
cd backend && git add arms.yaml README.md
git commit -m "docs: document subscription-CLI arms and the dedicated Celery queue"
```

---

## Final verification

- [ ] Run the full backend test suite: `cd backend && uv run pytest -v`. Expected: all tests pass; the two new e2e tests are PASSED (Claude Code) / SKIPPED (Codex, not installed here) rather than FAILED.
- [ ] Confirm `CLAUDE.md`'s Phase 1 bullet still accurately describes the adapter layer, or update it to mention the two subscription-CLI adapters alongside the existing two (small documentation touch-up, not a plan task on its own since it's a one-line edit to reflect what Tasks 1–3 already built).
