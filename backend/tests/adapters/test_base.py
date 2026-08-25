from app.adapters.base import ModelResponse


def test_model_response_defaults_cost_to_none():
    response = ModelResponse(
        text="hello",
        latency_ms=12.3,
        prompt_tokens=5,
        completion_tokens=2,
    )
    assert response.cost_estimate_usd is None


def test_model_response_defaults_finish_reason_to_none():
    response = ModelResponse(
        text="hello",
        latency_ms=12.3,
        prompt_tokens=5,
        completion_tokens=2,
    )
    assert response.finish_reason is None


def test_model_response_holds_all_fields():
    response = ModelResponse(
        text="hello",
        latency_ms=12.3,
        prompt_tokens=5,
        completion_tokens=2,
        cost_estimate_usd=0.001,
        finish_reason="stop",
    )
    assert response.text == "hello"
    assert response.latency_ms == 12.3
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 2
    assert response.cost_estimate_usd == 0.001
    assert response.finish_reason == "stop"
