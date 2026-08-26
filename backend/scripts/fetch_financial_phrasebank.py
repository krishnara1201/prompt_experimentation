"""One-time vendoring script -- downloads the Financial PhraseBank
100%-agreement subset and writes it to
backend/data/financial_phrasebank/sentences_allagree.txt in
`sentence@label` format.

Run once, from inside backend/:

    uv run --with datasets --with pandas python scripts/fetch_financial_phrasebank.py

Source: gtfintechlab/financial_phrasebank_sentences_allagree on Hugging
Face, a mirror of Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and
Takala, P. (2014), "Good debt or bad debt: Detecting semantic
orientations in economic texts." Licensed CC-BY-NC-SA-3.0 (non-commercial
use); see http://creativecommons.org/licenses/by-nc-sa/3.0/.
"""

from pathlib import Path

from datasets import concatenate_datasets, load_dataset

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# The HF dataset requires picking one of three configs ("5768", "78516",
# "944601") -- these are just different random train/test splits (seeds)
# over the *same* underlying 2264-sentence corpus, verified by comparing
# the union of sentences across configs. Any config works; we pick one
# arbitrarily and concatenate its train+test splits to recover the full
# 2264-row corpus (a single split alone is only a subset).
DATASET_CONFIG = "5768"

LICENSE_NOTE = (
    "# Financial PhraseBank, 100%-agreement subset (2264 sentences).\n"
    "# Source: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P.\n"
    '# (2014). "Good debt or bad debt: Detecting semantic orientations in\n'
    '# economic texts." Mirrored via gtfintechlab/financial_phrasebank_sentences_allagree\n'
    "# on Hugging Face. Licensed CC-BY-NC-SA-3.0 (non-commercial use);\n"
    "# see http://creativecommons.org/licenses/by-nc-sa/3.0/.\n"
    "# Format: one sentence per line, '<sentence>@<label>', UTF-8.\n"
)


def main() -> None:
    dataset = load_dataset(
        "gtfintechlab/financial_phrasebank_sentences_allagree", DATASET_CONFIG
    )
    split = concatenate_datasets([dataset["train"], dataset["test"]])

    out_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "financial_phrasebank"
        / "sentences_allagree.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(LICENSE_NOTE)
        for row in split:
            sentence = row["sentence"].replace("\n", " ").strip()
            label = LABEL_NAMES[row["label"]]
            f.write(f"{sentence}@{label}\n")

    print(f"Wrote {len(split)} sentences to {out_path}")


if __name__ == "__main__":
    main()
