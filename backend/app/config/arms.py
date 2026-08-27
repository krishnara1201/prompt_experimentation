import yaml

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter


class UnknownAdapterError(ValueError):
    pass


class InvalidArmConfigError(ValueError):
    pass


class InvalidJudgeConfigError(ValueError):
    pass


ADAPTER_TYPES = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


def load_arms(config_path: str) -> dict[str, ModelAdapter]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not isinstance(raw.get("arms"), list):
        raise InvalidArmConfigError(
            f"'{config_path}' must be a mapping with a top-level 'arms' list"
        )

    arms: dict[str, ModelAdapter] = {}
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

        adapter_cls = ADAPTER_TYPES.get(adapter_type)
        if adapter_cls is None:
            raise UnknownAdapterError(
                f"Unknown adapter type '{adapter_type}' for arm '{name}'"
            )

        try:
            arms[name] = adapter_cls(**entry)
        except TypeError as exc:
            raise InvalidArmConfigError(
                f"arm '{name}' has invalid fields for adapter '{adapter_type}': {exc}"
            ) from exc

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
