import pytest

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from app.adapters.codex_cli import CodexCLIAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.config.arms import (
    InvalidArmConfigError,
    InvalidJudgeConfigError,
    UnknownAdapterError,
    load_arms,
    load_judge_arm,
)

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

  - name: claude-code-sonnet-subscription
    adapter: claude_code_cli
    model: sonnet

  - name: codex-subscription
    adapter: codex_cli
    model: gpt-5-codex
"""

INVALID_CONFIG = """
arms:
  - name: mystery-arm
    adapter: telepathy
    model: mind-reader-v1
"""

MISSING_ARMS_KEY_CONFIG = """
not_arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
"""

MISSING_NAME_CONFIG = """
arms:
  - adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b
"""

MISSING_ADAPTER_CONFIG = """
arms:
  - name: qwen3-8b-local
    base_url: http://localhost:11434/v1
    model: qwen3:8b
"""

UNEXPECTED_FIELD_CONFIG = """
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_urls: http://localhost:11434/v1
    model: qwen3:8b
"""


def test_load_arms_builds_correct_adapter_types(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert set(arms.keys()) == {
        "qwen3-8b-local",
        "gpt-4o-mini",
        "claude-haiku",
        "claude-code-sonnet-subscription",
        "codex-subscription",
    }
    assert isinstance(arms["qwen3-8b-local"], OpenAICompatibleAdapter)
    assert isinstance(arms["gpt-4o-mini"], OpenAICompatibleAdapter)
    assert isinstance(arms["claude-haiku"], AnthropicAdapter)
    assert isinstance(arms["claude-code-sonnet-subscription"], ClaudeCodeCLIAdapter)
    assert isinstance(arms["codex-subscription"], CodexCLIAdapter)


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


def test_load_arms_passes_config_fields_through_for_subscription_cli_arms(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].model == "sonnet"
    assert arms["codex-subscription"].model == "gpt-5-codex"


def test_subscription_cli_arms_route_to_dedicated_celery_queue(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].celery_queue == "subscription_cli"
    assert arms["codex-subscription"].celery_queue == "subscription_cli"


def test_load_arms_raises_on_unknown_adapter_type(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(INVALID_CONFIG)

    with pytest.raises(UnknownAdapterError):
        load_arms(str(config_path))


def test_load_arms_raises_when_arms_key_missing(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(MISSING_ARMS_KEY_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


def test_load_arms_raises_when_entry_missing_name(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(MISSING_NAME_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


def test_load_arms_raises_when_entry_missing_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(MISSING_ADAPTER_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


def test_load_arms_raises_on_unexpected_field_for_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(UNEXPECTED_FIELD_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


VALID_JUDGE_CONFIG = (
    VALID_CONFIG
    + """
judge:
  adapter: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY
"""
)

MISSING_JUDGE_ADAPTER_CONFIG = (
    VALID_CONFIG
    + """
judge:
  model: claude-haiku-4-5-20251001
"""
)

UNKNOWN_JUDGE_ADAPTER_CONFIG = (
    VALID_CONFIG
    + """
judge:
  adapter: telepathy
  model: mind-reader-v1
"""
)


def test_load_judge_arm_builds_correct_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_JUDGE_CONFIG)

    judge = load_judge_arm(str(config_path))

    assert isinstance(judge, AnthropicAdapter)
    assert judge.model == "claude-haiku-4-5-20251001"


def test_load_judge_arm_raises_when_judge_key_missing(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)  # no judge: key at all

    with pytest.raises(InvalidJudgeConfigError):
        load_judge_arm(str(config_path))


def test_load_judge_arm_raises_when_adapter_missing(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(MISSING_JUDGE_ADAPTER_CONFIG)

    with pytest.raises(InvalidJudgeConfigError):
        load_judge_arm(str(config_path))


def test_load_judge_arm_raises_on_unknown_adapter_type(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(UNKNOWN_JUDGE_ADAPTER_CONFIG)

    with pytest.raises(UnknownAdapterError):
        load_judge_arm(str(config_path))
