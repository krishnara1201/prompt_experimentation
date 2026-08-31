import pytest

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from app.adapters.codex_cli import CodexCLIAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.config.arms import (
    Arm,
    InvalidArmConfigError,
    InvalidJudgeConfigError,
    UnknownAdapterError,
    load_arms,
    load_judge_arm,
)
from app.eval_prompt import EVAL_PROMPT_TEMPLATE

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


PROMPT_TEMPLATE_CONFIG = """
arms:
  - name: default-prompt-arm
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b

  - name: custom-prompt-arm
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b
    prompt_template: "Classify the sentiment of: {text}"
"""

PROMPT_TEMPLATE_NO_PLACEHOLDER_CONFIG = """
arms:
  - name: broken-prompt-arm
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b
    prompt_template: "Classify this sentence, please."
"""

PROMPT_TEMPLATE_UNKNOWN_PLACEHOLDER_CONFIG = """
arms:
  - name: broken-prompt-arm
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b
    prompt_template: "Classify {text} in the {register} register."
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
    assert isinstance(arms["qwen3-8b-local"].adapter, OpenAICompatibleAdapter)
    assert isinstance(arms["gpt-4o-mini"].adapter, OpenAICompatibleAdapter)
    assert isinstance(arms["claude-haiku"].adapter, AnthropicAdapter)
    assert isinstance(arms["claude-code-sonnet-subscription"].adapter, ClaudeCodeCLIAdapter)
    assert isinstance(arms["codex-subscription"].adapter, CodexCLIAdapter)


def test_load_arms_passes_config_fields_through(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    local = arms["qwen3-8b-local"].adapter
    assert local.base_url == "http://localhost:11434/v1"
    assert local.model == "qwen3:8b"

    hosted = arms["gpt-4o-mini"].adapter
    assert hosted.price_per_1m_input == 0.15
    assert hosted.price_per_1m_output == 0.60


def test_load_arms_passes_extra_body_through_to_openai_compatible_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        """
arms:
  - name: qwen3-no-think
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b
    extra_body:
      reasoning_effort: none
"""
    )

    arms = load_arms(str(config_path))

    assert arms["qwen3-no-think"].adapter.extra_body == {"reasoning_effort": "none"}


def test_load_arms_passes_config_fields_through_for_subscription_cli_arms(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].adapter.model == "sonnet"
    assert arms["codex-subscription"].adapter.model == "gpt-5-codex"


def test_subscription_cli_arms_route_to_dedicated_celery_queue(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["claude-code-sonnet-subscription"].adapter.celery_queue == "subscription_cli"
    assert arms["codex-subscription"].adapter.celery_queue == "subscription_cli"


def test_load_arms_returns_arm_objects_with_name_and_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arm = load_arms(str(config_path))["qwen3-8b-local"]

    assert isinstance(arm, Arm)
    assert arm.name == "qwen3-8b-local"
    assert isinstance(arm.adapter, OpenAICompatibleAdapter)


def test_arm_prompt_template_defaults_to_the_shared_eval_prompt(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(PROMPT_TEMPLATE_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["default-prompt-arm"].prompt_template == EVAL_PROMPT_TEMPLATE


def test_arm_uses_explicit_prompt_template_when_configured(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(PROMPT_TEMPLATE_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["custom-prompt-arm"].prompt_template == "Classify the sentiment of: {text}"


def test_arm_render_substitutes_the_example_text(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(PROMPT_TEMPLATE_CONFIG)

    arms = load_arms(str(config_path))

    assert arms["custom-prompt-arm"].render("Profits soared.") == (
        "Classify the sentiment of: Profits soared."
    )


def test_load_arms_rejects_prompt_template_without_text_placeholder(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(PROMPT_TEMPLATE_NO_PLACEHOLDER_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


def test_load_arms_rejects_prompt_template_with_unknown_placeholder(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(PROMPT_TEMPLATE_UNKNOWN_PLACEHOLDER_CONFIG)

    with pytest.raises(InvalidArmConfigError):
        load_arms(str(config_path))


def test_arm_bare_construction_still_yields_concrete_default_template():
    # The None default on the dataclass field is invisible to bare
    # construction -- __post_init__ fills it with the concrete template.
    assert Arm("x", object()).prompt_template == EVAL_PROMPT_TEMPLATE


def test_load_arms_without_task_uses_legacy_default_template(tmp_path):
    from app.eval_prompt import EVAL_PROMPT_TEMPLATE
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
    )
    arm = load_arms(str(config_path))["a"]
    assert arm.prompt_template == EVAL_PROMPT_TEMPLATE


def test_load_arms_with_task_uses_task_eval_prompt_for_unset_arms(tmp_path):
    from app.config.tasks import load_task
    task = load_task("financial_sentiment")
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
    )
    arm = load_arms(str(config_path), task=task)["a"]
    assert arm.prompt_template == task.eval_prompt


def test_load_arms_arm_level_template_overrides_task(tmp_path):
    from app.config.tasks import load_task
    task = load_task("financial_sentiment")
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
        '    prompt_template: "custom {text}"\n'
    )
    arm = load_arms(str(config_path), task=task)["a"]
    assert arm.prompt_template == "custom {text}"


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
