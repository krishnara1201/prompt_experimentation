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
