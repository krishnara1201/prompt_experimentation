"""QLoRA fine-tune. GPU-only unless --dry-run. Run from inside backend/:

    uv run python -m scripts.finetune_train
    uv run python -m scripts.finetune_train --dry-run   # dataset + config only
    uv run python -m scripts.finetune_train --reuse-dataset   # skip the rebuild
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from app.training.config import load_training_config
from app.training.dataset import DatasetBuildResult, build_sft_dataset
from app.training.train import run_training


def _reuse_dataset(cfg) -> DatasetBuildResult:
    """Reconstruct a DatasetBuildResult from a previous run's train/val jsonl
    instead of re-downloading the corpus and re-hitting the DB."""
    out_dir = Path(cfg.output_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"--reuse-dataset: {train_path} and/or {val_path} missing -- "
            "run once without --reuse-dataset first"
        )

    def _counts(path: Path) -> dict[str, int]:
        c: Counter[str] = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            c[json.loads(line)["messages"][-1]["content"]] += 1
        return dict(c)

    train_counts = _counts(train_path)
    val_counts = _counts(val_path)
    return DatasetBuildResult(
        train_path=train_path,
        val_path=val_path,
        license_path=out_dir / "LICENSE.txt",
        train_class_counts=train_counts,
        val_class_counts=val_counts,
        dropped_count=-1,  # unknown when reusing
        pool_size=sum(train_counts.values()) + sum(val_counts.values()),
        source="(reused)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Reuse train.jsonl/val.jsonl from a previous build instead of rebuilding.",
    )
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    if args.reuse_dataset:
        dataset = _reuse_dataset(cfg)
        print(f"dataset reused: pool={dataset.pool_size} (leakage drop count unknown)")
    else:
        dataset = build_sft_dataset(cfg)
        print(f"dataset ready: pool={dataset.pool_size} dropped={dataset.dropped_count}")

    if args.dry_run:
        sample = json.loads(dataset.train_path.read_text().splitlines()[0])
        print("dry run -- not loading the model. First formatted record:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        return

    result = run_training(cfg, dataset)
    print(f"adapter saved : {result.adapter_path}")
    print(f"wall seconds  : {result.wall_seconds:.0f}")
    for entry in result.loss_history:
        print(entry)


if __name__ == "__main__":
    main()
