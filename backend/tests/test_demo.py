from app.adapters.base import ModelResponse
from app.demo import format_row


def test_format_row_shows_cost_when_known():
    response = ModelResponse(
        text="hi",
        latency_ms=123.4,
        prompt_tokens=10,
        completion_tokens=2,
        cost_estimate_usd=0.000015,
    )
    row = format_row("gpt-4o-mini", "hello", response)
    assert "gpt-4o-mini" in row
    assert "hi" in row
    assert "0.000015" in row


def test_format_row_shows_na_when_cost_unknown():
    response = ModelResponse(
        text="hi",
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        cost_estimate_usd=None,
    )
    row = format_row("qwen3-8b-local", "hello", response)
    assert "n/a" in row
