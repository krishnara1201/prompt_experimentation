"""Build + leakage-check the fine-tuning dataset. Run from inside backend/:

    uv run python -m scripts.finetune_prep
"""
import argparse

from app.training.config import load_training_config
from app.training.dataset import build_sft_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    result = build_sft_dataset(cfg)

    print(f"source            : {result.source}")
    print(f"pool size         : {result.pool_size}")
    print(f"dropped (leakage) : {result.dropped_count}")
    print(f"train class counts: {result.train_class_counts}")
    print(f"val class counts  : {result.val_class_counts}")
    print(f"train jsonl       : {result.train_path}")
    print(f"val jsonl         : {result.val_path}")


if __name__ == "__main__":
    main()
