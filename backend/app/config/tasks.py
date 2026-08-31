"""Task packs -- the swappable other half of "prompts as arms".

A task pack lives at ``backend/tasks/<name>/`` and carries everything the
eval loop needs that is not a model: the dataset pointer, the valid label
set, the default eval prompt, and the judge rubric. ``arms.yaml`` names the
active task with a top-level ``task:`` key (default ``financial_sentiment``).
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks"
DEFAULT_TASK = "financial_sentiment"

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_REQUIRED_KEYS = {
    "name",
    "description",
    "labels",
    "source",
    "data",
    "format",
    "eval_prompt",
    "rubric",
}
_VALID_FORMATS = {"jsonl", "phrasebank"}
_RUBRIC_FIELDS = {"input_text", "gold_label", "model_output", "description"}


class InvalidTaskConfigError(ValueError):
    pass


class UnknownTaskError(ValueError):
    pass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    labels: tuple[str, ...]
    source: str
    data_path: Path
    data_format: str
    eval_prompt: str
    rubric: str
    label_names: tuple[str, ...] | None


def _check_placeholders(
    field: str, template: str, allowed: set[str], required: set[str]
) -> None:
    if not isinstance(template, str):
        raise InvalidTaskConfigError(f"'{field}' must be a string")
    found = set(_PLACEHOLDER_RE.findall(template))
    missing = required - found
    if missing:
        raise InvalidTaskConfigError(
            f"'{field}' must contain the placeholder(s) "
            f"{sorted('{' + m + '}' for m in missing)}"
        )
    unknown = found - allowed
    if unknown:
        raise InvalidTaskConfigError(
            f"'{field}' has unsupported placeholder(s) "
            f"{sorted('{' + u + '}' for u in unknown)}; "
            f"only {sorted('{' + a + '}' for a in allowed)} are substituted"
        )


def load_task(name: str, tasks_dir: Path | None = None) -> TaskConfig:
    root = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    task_dir = root / name
    config_path = task_dir / "task.yaml"
    if not config_path.is_file():
        raise UnknownTaskError(f"no task pack '{name}' (looked for {config_path})")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise InvalidTaskConfigError(f"{config_path} must be a mapping")

    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise InvalidTaskConfigError(
            f"{config_path} missing key(s): {', '.join(sorted(missing))}"
        )

    labels = raw["labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(x, str) and x for x in labels)
    ):
        raise InvalidTaskConfigError(
            "'labels' must be a non-empty list of non-empty strings"
        )

    data_format = raw["format"]
    if data_format not in _VALID_FORMATS:
        raise InvalidTaskConfigError(
            f"'format' must be one of {sorted(_VALID_FORMATS)}, got {data_format!r}"
        )

    _check_placeholders("eval_prompt", raw["eval_prompt"], {"text"}, {"text"})
    _check_placeholders(
        "rubric",
        raw["rubric"],
        _RUBRIC_FIELDS,
        {"input_text", "gold_label", "model_output"},
    )

    data_path = (task_dir / raw["data"]).resolve()
    if not data_path.is_file():
        raise InvalidTaskConfigError(f"'data' file not found: {data_path}")

    label_names = raw.get("label_names")
    if label_names is not None:
        if not isinstance(label_names, list) or set(label_names) != set(labels):
            raise InvalidTaskConfigError(
                "'label_names' must be a list covering the same set as 'labels'"
            )
        label_names = tuple(label_names)

    return TaskConfig(
        name=raw["name"],
        description=raw["description"],
        labels=tuple(labels),
        source=raw["source"],
        data_path=data_path,
        data_format=data_format,
        eval_prompt=raw["eval_prompt"],
        rubric=raw["rubric"],
        label_names=label_names,
    )


def list_tasks(tasks_dir: Path | None = None) -> list[str]:
    root = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "task.yaml").is_file())


def active_task_name(arms_path: str | os.PathLike) -> str:
    raw = yaml.safe_load(Path(arms_path).read_text())
    if isinstance(raw, dict) and isinstance(raw.get("task"), str):
        return raw["task"]
    return DEFAULT_TASK
