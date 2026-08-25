import pytest

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.config.arms import UnknownAdapterError, load_arms

VALID_CONFIG = """
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b

  - name: gpt-4o-mini
    adapter: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    price_per_1m_input: 0.15
    price_per_1m_output: 0.60

  - name: claude-haiku
    adapter: anthropic
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
    price_per_1m_input: 1.00
    price_per_1m_output: 5.00
"""

INVALID_CONFIG = """
arms:
  - name: mystery-arm
    adapter: telepathy
    model: mind-reader-v1
"""


def test_load_arms_builds_correct_adapter_types(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert set(arms.keys()) == {"qwen3-8b-local", "gpt-4o-mini", "claude-haiku"}
    assert isinstance(arms["qwen3-8b-local"], OpenAICompatibleAdapter)
    assert isinstance(arms["gpt-4o-mini"], OpenAICompatibleAdapter)
    assert isinstance(arms["claude-haiku"], AnthropicAdapter)


def test_load_arms_passes_config_fields_through(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    local = arms["qwen3-8b-local"]
    assert local.base_url == "http://localhost:11434/v1"
    assert local.model == "qwen3:8b"

    hosted = arms["gpt-4o-mini"]
    assert hosted.price_per_1m_input == 0.15
    assert hosted.price_per_1m_output == 0.60


def test_load_arms_raises_on_unknown_adapter_type(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(INVALID_CONFIG)

    with pytest.raises(UnknownAdapterError):
        load_arms(str(config_path))
