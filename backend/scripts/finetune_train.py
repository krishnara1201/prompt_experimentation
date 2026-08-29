"""QLoRA fine-tune. GPU-only unless --dry-run. Run from inside backend/:

    uv run python -m scripts.finetune_train
    uv run python -m scripts.finetune_train --dry-run   # dataset + config only
"""
import argparse
import json

from app.training.config import load_training_config
from app.training.dataset import build_sft_dataset
from app.training.train import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-dataset", action="store_true",
                        help="(reserved) reuse existing artifacts; currently always rebuilds")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
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
