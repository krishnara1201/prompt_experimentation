import httpx
import pytest
import respx

from app.adapters.openai_compatible import OpenAICompatibleAdapter


@respx.mock
def test_generate_returns_text_and_token_counts():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("What is the capital of France?")

    assert response.text == "Paris"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 1
    assert response.latency_ms > 0


@respx.mock
def test_generate_cost_is_none_without_pricing_config():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("What is the capital of France?")

    assert response.cost_estimate_usd is None


@respx.mock
def test_generate_computes_cost_when_pricing_configured():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    )

    response = adapter.generate("hello")

    assert response.cost_estimate_usd == pytest.approx(0.75)


@respx.mock
def test_generate_sends_bearer_token_when_key_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}], "usage": {}},
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )

    adapter.generate("hello")

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test123"


@respx.mock
def test_generate_omits_auth_header_without_api_key_env():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}], "usage": {}},
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    adapter.generate("hello")

    assert "Authorization" not in route.calls.last.request.headers
