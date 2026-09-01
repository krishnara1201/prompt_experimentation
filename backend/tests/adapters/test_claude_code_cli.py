import json
import os
import subprocess
from datetime import timezone

import pytest

from app.adapters import claude_code_cli
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter, UsageLimitError


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


def test_usage_limit_from_result_text_parses_retry_at(monkeypatch):
    # The CLI emits "Claude AI usage limit reached|<unix ts>" as the result.
    payload = {"result": "Claude AI usage limit reached|1893456000", "is_error": True}
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stdout=json.dumps(payload)),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(UsageLimitError) as excinfo:
        adapter.generate("hello")

    retry_at = excinfo.value.retry_at
    assert retry_at is not None
    assert retry_at.tzinfo is not None
    assert int(retry_at.astimezone(timezone.utc).timestamp()) == 1893456000


def test_usage_limit_from_api_error_status(monkeypatch):
    payload = {"result": "", "is_error": True, "api_error_status": "rate_limit_error"}
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=0, stdout=json.dumps(payload)),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(UsageLimitError):
        adapter.generate("hello")


def test_usage_limit_from_stderr_phrase(monkeypatch):
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: usage limit exceeded for this account"),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(UsageLimitError):
        adapter.generate("hello")


def test_bare_nonzero_exit_is_not_a_usage_limit(monkeypatch):
    # Empty stderr, no JSON — the worker treats this as a rate-limit for
    # backoff, but the adapter must NOT escalate it to a multi-hour pause.
    monkeypatch.setattr(
        claude_code_cli.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr=""),
    )
    adapter = ClaudeCodeCLIAdapter(model="sonnet")

    with pytest.raises(RuntimeError) as excinfo:
        adapter.generate("hello")
    assert not isinstance(excinfo.value, UsageLimitError)


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
