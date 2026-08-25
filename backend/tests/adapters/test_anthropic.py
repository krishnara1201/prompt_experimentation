import httpx
import pytest
import respx

from app.adapters.anthropic import AnthropicAdapter


@respx.mock
def test_generate_returns_text_and_token_counts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Paris"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )
    )
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    response = adapter.generate("What is the capital of France?")

    assert response.text == "Paris"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 1
    assert response.latency_ms > 0


def test_generate_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    with pytest.raises(RuntimeError):
        adapter.generate("hello")


@respx.mock
def test_generate_sends_required_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "hi"}], "usage": {}},
        )
    )
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    adapter.generate("hello")

    sent = route.calls.last.request.headers
    assert sent["x-api-key"] == "sk-ant-test"
    assert sent["anthropic-version"] == "2023-06-01"


@respx.mock
def test_generate_computes_cost_when_pricing_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            },
        )
    )
    adapter = AnthropicAdapter(
        model="claude-haiku-4-5-20251001",
        price_per_1m_input=1.00,
        price_per_1m_output=5.00,
    )

    response = adapter.generate("hello")

    assert response.cost_estimate_usd == pytest.approx(6.00)
