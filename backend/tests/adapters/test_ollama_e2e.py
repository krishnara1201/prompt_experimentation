import socket

import pytest

from app.adapters.openai_compatible import OpenAICompatibleAdapter


def _ollama_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running on localhost:11434")
def test_qwen3_generates_nonempty_text():
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("Reply with the single word: hello")

    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.latency_ms > 0
    assert response.cost_estimate_usd is None
