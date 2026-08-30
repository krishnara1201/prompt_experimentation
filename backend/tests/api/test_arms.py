from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config.arms import Arm
from app.eval_prompt import EVAL_PROMPT_TEMPLATE
from app.main import app


class _FakeAdapter:
    def __init__(self, model):
        self.model = model


FAKE_ARMS = {
    "qwen3-8b-local": Arm("qwen3-8b-local", _FakeAdapter("qwen3:8b")),
    "claude-haiku": Arm(
        "claude-haiku",
        _FakeAdapter("claude-haiku-4-5-20251001"),
        prompt_template="Sentiment of {text}?",
    ),
}

FAKE_RAW = {
    "arms": [
        {"name": "qwen3-8b-local", "adapter": "openai_compatible"},
        {"name": "claude-haiku", "adapter": "anthropic"},
    ]
}


@patch("app.api.routes.arms.yaml.safe_load", return_value=FAKE_RAW)
@patch("app.api.routes.arms.load_arms", return_value=FAKE_ARMS)
def test_list_arms_reports_name_adapter_model_and_prompt_template(mock_load, mock_yaml):
    response = TestClient(app).get("/arms")
    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "qwen3-8b-local",
            "adapter": "openai_compatible",
            "model": "qwen3:8b",
            "prompt_template": EVAL_PROMPT_TEMPLATE,
        },
        {
            "name": "claude-haiku",
            "adapter": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "prompt_template": "Sentiment of {text}?",
        },
    ]


@patch("app.api.routes.arms.load_arms", side_effect=ValueError("bad config"))
def test_list_arms_500_on_broken_config(mock_load):
    response = TestClient(app).get("/arms")
    assert response.status_code == 500
    assert "bad config" in response.json()["detail"]
