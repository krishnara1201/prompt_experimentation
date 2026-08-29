import os

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


def test_score_financial_sentiment_tool_is_directly_callable(monkeypatch):
    fake_adapter = _FakeAdapter("SCORE: 3\nRATIONALE: Hedged but on-topic.")
    monkeypatch.setattr("app.mcp_judge_server.load_judge_arm", lambda config_path: fake_adapter)

    from app.mcp_judge_server import score_financial_sentiment

    result = score_financial_sentiment("Revenue was flat.", "neutral", "Results were mixed.")

    assert result == {"score": 3, "rationale": "Hedged but on-topic."}


def test_mcp_server_instance_has_expected_name():
    from app.mcp_judge_server import mcp

    assert mcp.name == "financial-sentiment-judge"


def test_score_financial_sentiment_tool_publishes_structured_output_schema():
    from app.mcp_judge_server import mcp

    tool = mcp._tool_manager.get_tool("score_financial_sentiment")

    assert tool is not None
    assert tool.output_schema is not None
    assert set(tool.output_schema["properties"]) == {"score", "rationale"}


def test_dotenv_is_loaded_from_backend_env_file(monkeypatch, tmp_path):
    from app import mcp_judge_server as mjs

    env_file = tmp_path / ".env"
    env_file.write_text("MCP_JUDGE_SERVER_TEST_VAR=loaded\n")
    monkeypatch.setattr(mjs, "ENV_PATH", env_file)
    monkeypatch.delenv("MCP_JUDGE_SERVER_TEST_VAR", raising=False)

    mjs._load_dotenv_if_present()

    assert os.environ.get("MCP_JUDGE_SERVER_TEST_VAR") == "loaded"
