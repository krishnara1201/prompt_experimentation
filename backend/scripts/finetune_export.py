"""Merge the adapter, convert to GGUF, `ollama create`, print the arms.yaml
entry to paste. GPU-only. Run from inside backend/:

    uv run python -m scripts.finetune_export
"""
import argparse

from app.training.config import load_training_config
from app.training.export import export_arm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    result = export_arm(cfg)

    print(f"gguf      : {result.gguf_path}")
    print(f"modelfile : {result.modelfile_path}")
    print(f"ollama tag: {result.ollama_tag}")
    print()
    print("Paste this into arms.yaml under `arms:` --")
    print(result.snippet)


if __name__ == "__main__":
    main()
