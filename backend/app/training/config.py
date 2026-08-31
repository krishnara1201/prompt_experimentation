"""Fine-tuning config -- the training.yaml counterpart to app/config/arms.py."""
from dataclasses import dataclass, fields
from typing import get_args, get_origin

import yaml


class InvalidTrainingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    task: str
    base_model: str
    source_dataset: str
    source_config: str
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    epochs: float
    learning_rate: float
    batch_size: int
    grad_accum: int
    seed: int
    val_fraction: float
    balance_neutral: bool
    min_pool_size: int
    gguf_quant: str
    ollama_tag: str
    output_dir: str


def _check_type(name: str, value: object, annotation: object) -> None:
    origin = get_origin(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise InvalidTrainingConfigError(f"'{name}' must be a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation) or (str,)
        for item in value:
            if not isinstance(item, item_type):
                raise InvalidTrainingConfigError(f"'{name}' items must be {item_type.__name__}")
        return
    # bool is a subclass of int -- check it first so `epochs: true` is rejected
    if annotation in (int, float) and isinstance(value, bool):
        raise InvalidTrainingConfigError(f"'{name}' must be {annotation.__name__}, got bool")
    if annotation is float and isinstance(value, int):
        return  # YAML ints are acceptable where a float is wanted
    if not isinstance(value, annotation):
        raise InvalidTrainingConfigError(
            f"'{name}' must be {annotation.__name__}, got {type(value).__name__}"
        )


def load_training_config(path: str = "training.yaml") -> TrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise InvalidTrainingConfigError(f"'{path}' must be a mapping")

    known = {f.name: f for f in fields(TrainingConfig)}
    unknown = set(raw) - set(known)
    if unknown:
        raise InvalidTrainingConfigError(f"unknown key(s) in '{path}': {', '.join(sorted(unknown))}")

    missing = set(known) - set(raw)
    if missing:
        raise InvalidTrainingConfigError(
            f"missing required key(s) in '{path}': {', '.join(sorted(missing))}"
        )

    for name, field in known.items():
        _check_type(name, raw[name], field.type)

    coerced = dict(raw)
    coerced["epochs"] = float(coerced["epochs"])
    coerced["learning_rate"] = float(coerced["learning_rate"])
    coerced["lora_dropout"] = float(coerced["lora_dropout"])
    coerced["val_fraction"] = float(coerced["val_fraction"])
    return TrainingConfig(**coerced)
