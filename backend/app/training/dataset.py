"""Builds the supervised fine-tuning dataset from Financial PhraseBank's
lower-agreement subset, with a hard guard that nothing in it overlaps the
all-agree eval set (or, transitively, the judge-calibration rows -- all of
which are all-agree sentences).

Dataset licence: CC BY-NC-SA 3.0, Malo et al. 2014. Downloaded at runtime,
never vendored.
"""
import asyncio
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.tasks import load_task
from app.db.models import EvalExample
from app.db.session import engine
from app.eval_prompt import render_eval_prompt
from app.training.config import TrainingConfig

# Financial-specific: only written when the financial source is used. A
# non-financial task pack would need its own license note (out of scope here).
LICENSE_TEXT = (
    "Financial PhraseBank (Malo, P., Sinha, A., Korhonen, P., Wallenius, J.,\n"
    "and Takala, P. 2014. 'Good debt or bad debt: Detecting semantic\n"
    "orientations in economic texts.'). Lower-agreement subset used for\n"
    "fine-tuning only. Licensed CC BY-NC-SA 3.0 (non-commercial);\n"
    "see http://creativecommons.org/licenses/by-nc-sa/3.0/. Not redistributed.\n"
)

_WS = re.compile(r"\s+")


class LeakageError(Exception):
    pass


@dataclass(frozen=True)
class DatasetBuildResult:
    train_path: Path
    val_path: Path
    license_path: Path
    train_class_counts: dict[str, int]
    val_class_counts: dict[str, int]
    dropped_count: int
    pool_size: int
    source: str


def normalize_sentence(s: str) -> str:
    # NFKC first: the eval set is seeded from a vendored UTF-8 file while the
    # training rows come from a different HF mirror, so quote-style / ligature
    # / full-width divergence would silently defeat the leakage intersection.
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip().casefold()


def load_source_examples(cfg: TrainingConfig) -> list[tuple[str, str]]:
    """(sentence, label_word) pairs from the configured HF subset.

    Monkeypatched in tests -- keep it a thin, side-effect-only loader.
    """
    from datasets import concatenate_datasets, load_dataset

    task = load_task(cfg.task)
    valid_labels = set(task.labels)
    int_to_word = dict(enumerate(task.label_names)) if task.label_names else {}

    dataset = load_dataset(cfg.source_dataset, cfg.source_config)
    combined = concatenate_datasets(list(dataset.values()))
    pairs: list[tuple[str, str]] = []
    for row in combined:
        sentence = str(row["sentence"]).replace("\n", " ").strip()
        raw_label = row["label"]
        if isinstance(raw_label, int):
            if not int_to_word:
                raise ValueError(
                    f"int label {raw_label!r} in {cfg.source_dataset} but task "
                    f"'{cfg.task}' task.yaml needs `label_names` to train from this source"
                )
            label = int_to_word[int(raw_label)]
        else:
            label = str(raw_label)
        if label not in valid_labels:
            raise ValueError(f"Unexpected label {raw_label!r} in {cfg.source_dataset}")
        pairs.append((sentence, label))
    return pairs


def fetch_eval_texts() -> set[str]:
    """Every eval_example.text, for the leakage guard. Monkeypatched in tests."""

    async def _run() -> set[str]:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(EvalExample.text))
            return set(result.scalars().all())

    return asyncio.run(_run())


def _balance_neutral(rows: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    minority = max(
        (len(v) for k, v in by_label.items() if k != "neutral"), default=0
    )
    neutral = by_label.get("neutral", [])
    if len(neutral) > minority:
        rng = random.Random(seed)
        by_label["neutral"] = rng.sample(neutral, minority)
    # No shuffle here -- build_sft_dataset reshuffles the pool with the same
    # seed right after this returns.
    return [row for rows_ in by_label.values() for row in rows_]


def _write_jsonl(path: Path, rows: list[tuple[str, str]], eval_prompt: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sentence, label in rows:
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": render_eval_prompt(sentence, template=eval_prompt),
                    },
                    {"role": "assistant", "content": label},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_dataset(cfg: TrainingConfig) -> DatasetBuildResult:
    task = load_task(cfg.task)
    valid_labels = set(task.labels)
    int_to_word = dict(enumerate(task.label_names)) if task.label_names else {}

    source_rows = load_source_examples(cfg)
    # The real loader already maps int labels to words; this also covers a
    # monkeypatched / alternative loader that yields raw int labels. String
    # labels pass straight through unchanged.
    resolved_rows: list[tuple[str, str]] = []
    for sentence, label in source_rows:
        if isinstance(label, int):
            if not int_to_word:
                raise ValueError(
                    f"int label {label!r} from load_source_examples but task "
                    f"'{cfg.task}' task.yaml has no `label_names` to map it"
                )
            label = int_to_word[int(label)]
        resolved_rows.append((sentence, label))
    source_rows = resolved_rows

    eval_norm = {normalize_sentence(t) for t in fetch_eval_texts()}

    kept: list[tuple[str, str]] = []
    dropped = 0
    for sentence, label in source_rows:
        if normalize_sentence(sentence) in eval_norm:
            dropped += 1
            continue
        kept.append((sentence, label))

    # Guard: recheck the survivors are clean, and the pool is plausible.
    still_overlapping = [s for s, _ in kept if normalize_sentence(s) in eval_norm]
    if still_overlapping:
        raise LeakageError(
            f"{len(still_overlapping)} training sentences still overlap the eval set after dropping"
        )
    # Fail CLOSED when the intersection can't have run: an empty eval table or
    # zero drops both produce a training set that may CONTAIN the eval set.
    if not eval_norm:
        raise LeakageError(
            "eval_example is empty -- seed the eval set (uv run python -m scripts.seed_eval_examples) "
            "before building training data"
        )
    if dropped == 0:
        raise LeakageError(
            "0 eval-set rows overlapped the training pool -- since the all-agree eval set is a "
            "subset of the configured lower-agreement subset, 0 drops means a source-config or "
            "text-normalization mismatch between the two corpora, not a clean split"
        )

    do_balance = cfg.balance_neutral and "neutral" in valid_labels
    if do_balance:
        kept = _balance_neutral(kept, cfg.seed)

    # min_pool_size is checked AFTER balancing -- balancing shrinks the pool,
    # so a pre-balance check could pass while the returned pool_size is below
    # the floor.
    if len(kept) < cfg.min_pool_size:
        stage = "after neutral balancing" if do_balance else "after leakage drop"
        raise LeakageError(
            f"training pool is {len(kept)} rows {stage}, below min_pool_size "
            f"({cfg.min_pool_size}) -- wrong subset or normalization?"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(kept)
    n_val = int(round(len(kept) * cfg.val_fraction))
    val_rows = kept[:n_val]
    train_rows = kept[n_val:]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    license_path = out_dir / "LICENSE.txt"
    _write_jsonl(train_path, train_rows, task.eval_prompt)
    _write_jsonl(val_path, val_rows, task.eval_prompt)
    license_path.write_text(LICENSE_TEXT, encoding="utf-8")

    return DatasetBuildResult(
        train_path=train_path,
        val_path=val_path,
        license_path=license_path,
        train_class_counts=dict(Counter(l for _, l in train_rows)),
        val_class_counts=dict(Counter(l for _, l in val_rows)),
        dropped_count=dropped,
        pool_size=len(kept),
        source=f"{cfg.source_dataset}:{cfg.source_config}",
    )
