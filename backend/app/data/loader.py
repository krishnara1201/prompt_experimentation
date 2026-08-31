"""Loads a task pack's dataset into (text, gold_label) rows.

JSONL is the default; the ``phrasebank`` format delegates to the existing
Financial PhraseBank reader so the vendored ``sentence@label`` file needs
no conversion.
"""
import json
from dataclasses import dataclass

from app.config.tasks import TaskConfig
from app.data import financial_phrasebank


class MalformedDataError(ValueError):
    pass


@dataclass(frozen=True)
class TaskExample:
    text: str
    gold_label: str


def _load_jsonl(task: TaskConfig) -> list[TaskExample]:
    allowed = set(task.labels)
    out: list[TaskExample] = []
    with open(task.data_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedDataError(f"{task.data_path} line {lineno}: invalid JSON ({exc})") from exc
            text = obj.get("text")
            label = obj.get("gold_label")
            if not isinstance(text, str) or not text.strip():
                raise MalformedDataError(f"{task.data_path} line {lineno}: missing/empty 'text'")
            if not isinstance(label, str) or not label.strip():
                raise MalformedDataError(f"{task.data_path} line {lineno}: missing/empty 'gold_label'")
            if label not in allowed:
                raise MalformedDataError(
                    f"{task.data_path} line {lineno}: gold_label {label!r} not in task labels {sorted(allowed)}"
                )
            out.append(TaskExample(text=text, gold_label=label))
    return out


def _load_phrasebank(task: TaskConfig) -> list[TaskExample]:
    rows = financial_phrasebank.load_examples(task.data_path)
    return [TaskExample(text=r.text, gold_label=r.label) for r in rows]


def load_task_examples(task: TaskConfig) -> list[TaskExample]:
    if task.data_format == "jsonl":
        return _load_jsonl(task)
    if task.data_format == "phrasebank":
        return _load_phrasebank(task)
    raise MalformedDataError(f"unknown data format {task.data_format!r}")  # unreachable; validated at load
