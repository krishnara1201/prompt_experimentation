import pytest

from app.adapters.base import ModelResponse
from app.judge.scorer import JudgeParseError
from app.mcp_judge_server import ARMS_PATH, _score_financial_sentiment


class _FakeAdapter:
    def __init__(self, text):
        self._text = text

    def generate(self, prompt):
        return ModelResponse(text=self._text, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)


def test_score_financial_sentiment_returns_score_and_rationale():
    adapter = _FakeAdapter("SCORE: 4\nRATIONALE: Correct sentiment, slightly terse.")

    result = _score_financial_sentiment(
        "Profits rose sharply.", "positive", "The tone is positive.", adapter=adapter
    )

    assert result == {"score": 4, "rationale": "Correct sentiment, slightly terse."}


def test_score_financial_sentiment_raises_on_malformed_judge_output():
    adapter = _FakeAdapter("not in the expected format")

    with pytest.raises(JudgeParseError):
        _score_financial_sentiment("Profits rose.", "positive", "Bullish.", adapter=adapter)


def test_score_financial_sentiment_loads_judge_arm_when_no_adapter_given(monkeypatch):
    fake_adapter = _FakeAdapter("SCORE: 5\nRATIONALE: Nailed it.")
    calls = []

    def fake_load_judge_arm(config_path):
        calls.append(config_path)
        return fake_adapter

    monkeypatch.setattr("app.mcp_judge_server.load_judge_arm", fake_load_judge_arm)

    result = _score_financial_sentiment("Profits rose.", "positive", "Bullish outlook.")

    assert result == {"score": 5, "rationale": "Nailed it."}
    assert calls == [str(ARMS_PATH)]
