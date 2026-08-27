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
