from pathlib import Path

import pytest

from app.data.financial_phrasebank import load_examples

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_phrasebank.txt"
VENDORED_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "financial_phrasebank"
    / "sentences_allagree.txt"
)


def test_load_examples_parses_fixture_and_skips_comments():
    examples = load_examples(FIXTURE_PATH)

    assert len(examples) == 3
    assert examples[0].text == "Profits rose to $5.2@million in Q3."
    assert examples[0].label == "positive"
    assert examples[1].label == "negative"
    assert examples[2].label == "neutral"


def test_load_examples_raises_on_malformed_line(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("no delimiter here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_examples(bad_file)


def test_vendored_file_has_expected_shape():
    examples = load_examples(VENDORED_PATH)

    assert len(examples) == 2264
    assert all(e.label in {"positive", "negative", "neutral"} for e in examples)
    assert all(e.text for e in examples)
