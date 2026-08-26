from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = {"positive", "negative", "neutral"}


@dataclass
class PhrasebankExample:
    text: str
    label: str


def load_examples(path: str | Path) -> list[PhrasebankExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            text, _, label = line.rpartition("@")
            if not text or label not in VALID_LABELS:
                raise ValueError(f"Malformed line in {path}: {line!r}")
            examples.append(PhrasebankExample(text=text, label=label))
    return examples
