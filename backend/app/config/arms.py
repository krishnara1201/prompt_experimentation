from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from app.adapters.codex_cli import CodexCLIAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.eval_prompt import EVAL_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from app.config.tasks import TaskConfig


class UnknownAdapterError(ValueError):
    pass


class InvalidArmConfigError(ValueError):
    pass


class InvalidJudgeConfigError(ValueError):
    pass


ADAPTER_TYPES = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "claude_code_cli": ClaudeCodeCLIAdapter,
    "codex_cli": CodexCLIAdapter,
}


@dataclass
class Arm:
    """A single experiment arm: an adapter plus the task prompt it is asked.

    Two arms with the same adapter/model but different `prompt_template` is
    how prompts are A/B-tested — every arm still sees the same eval examples,
    so the paired stats compare the prompts.
    """

    name: str
    adapter: ModelAdapter
    prompt_template: str | None = None

    def __post_init__(self) -> None:
        # The field default is None so `load_arms` can tell "unset" from an
        # explicit template, but a bare `Arm(name, adapter)` (worker, tests)
        # must keep behaving exactly as before: a concrete template.
        if self.prompt_template is None:
            self.prompt_template = EVAL_PROMPT_TEMPLATE

    def render(self, text: str) -> str:
        return self.prompt_template.format(text=text)


def _validate_prompt_template(name: str, template: str) -> None:
    if "{text}" not in template:
        raise InvalidArmConfigError(
            f"arm '{name}' prompt_template must contain the '{{text}}' placeholder"
        )
    try:
        template.format(text="probe")
    except (KeyError, IndexError, ValueError) as exc:
        raise InvalidArmConfigError(
            f"arm '{name}' prompt_template has an unsupported placeholder ({exc}); "
            "only '{text}' is substituted"
        ) from exc


def load_arms(
    config_path: str, *, task: "TaskConfig | None" = None
) -> dict[str, "Arm"]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not isinstance(raw.get("arms"), list):
        raise InvalidArmConfigError(
            f"'{config_path}' must be a mapping with a top-level 'arms' list"
        )

    arms: dict[str, Arm] = {}
    for i, entry in enumerate(raw["arms"]):
        if "name" not in entry:
            raise InvalidArmConfigError(f"arms[{i}] is missing required key 'name'")
        if "adapter" not in entry:
            raise InvalidArmConfigError(
                f"arm '{entry['name']}' is missing required key 'adapter'"
            )

        entry = dict(entry)
        name = entry.pop("name")
        adapter_type = entry.pop("adapter")
        default_template = task.eval_prompt if task is not None else EVAL_PROMPT_TEMPLATE
        prompt_template = entry.pop("prompt_template", default_template)
        _validate_prompt_template(name, prompt_template)

        adapter_cls = ADAPTER_TYPES.get(adapter_type)
        if adapter_cls is None:
            raise UnknownAdapterError(
                f"Unknown adapter type '{adapter_type}' for arm '{name}'"
            )

        try:
            adapter = adapter_cls(**entry)
        except TypeError as exc:
            raise InvalidArmConfigError(
                f"arm '{name}' has invalid fields for adapter '{adapter_type}': {exc}"
            ) from exc

        arms[name] = Arm(name=name, adapter=adapter, prompt_template=prompt_template)

    return arms


def load_judge_arm(config_path: str) -> ModelAdapter:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not isinstance(raw.get("judge"), dict):
        raise InvalidJudgeConfigError(f"'{config_path}' must have a top-level 'judge' mapping")

    entry = dict(raw["judge"])
    if "adapter" not in entry:
        raise InvalidJudgeConfigError("'judge' entry is missing required key 'adapter'")

    adapter_type = entry.pop("adapter")
    adapter_cls = ADAPTER_TYPES.get(adapter_type)
    if adapter_cls is None:
        raise UnknownAdapterError(f"Unknown adapter type '{adapter_type}' for judge")

    try:
        return adapter_cls(**entry)
    except TypeError as exc:
        raise InvalidJudgeConfigError(
            f"judge config has invalid fields for adapter '{adapter_type}': {exc}"
        ) from exc
