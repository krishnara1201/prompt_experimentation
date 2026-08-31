"""Regenerates backend/tasks/ag_news/data.jsonl: a stratified sample of AG
News (topic classification) for the prompt-A/B demo.

    uv run --with datasets python backend/tasks/ag_news/fetch_ag_news.py --per-class 30

AG News is the AG corpus (ComeToMyHead news aggregator), redistributed for
research via Hugging Face. Only this small sample is vendored; see LICENSE.txt.
"""
import argparse
import json
import random
from pathlib import Path

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
OUT = Path(__file__).resolve().parent / "data.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="test")
    by_label: dict[int, list[str]] = {k: [] for k in LABELS}
    for row in ds:
        by_label[row["label"]].append(row["text"].replace("\n", " ").strip())

    rng = random.Random(args.seed)
    rows = []
    for label_int, word in LABELS.items():
        picks = rng.sample(by_label[label_int], args.per_class)
        rows.extend({"text": t, "gold_label": word} for t in picks)
    rng.shuffle(rows)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
