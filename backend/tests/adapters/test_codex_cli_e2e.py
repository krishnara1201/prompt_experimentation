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
