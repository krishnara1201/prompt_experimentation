import yaml

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter


class UnknownAdapterError(ValueError):
    pass


ADAPTER_TYPES = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


def load_arms(config_path: str) -> dict[str, ModelAdapter]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    arms: dict[str, ModelAdapter] = {}
    for entry in raw["arms"]:
        entry = dict(entry)
        name = entry.pop("name")
        adapter_type = entry.pop("adapter")

        adapter_cls = ADAPTER_TYPES.get(adapter_type)
        if adapter_cls is None:
            raise UnknownAdapterError(
                f"Unknown adapter type '{adapter_type}' for arm '{name}'"
            )

        arms[name] = adapter_cls(**entry)

    return arms
