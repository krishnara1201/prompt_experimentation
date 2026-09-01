import pytest

from app.data.financial_phrasebank import PhrasebankExample
from scripts.judge_tool_dryrun import (
    LABELS,
    NEGATIVE_CASES,
    TOOL_NAME,
    format_row,
    sample_by_label,
    synthetic_output,
)


def test_tool_name_is_the_generalised_judge_tool():
    assert TOOL_NAME == "score_output_against_gold"


def test_synthetic_output_wrong_label_is_some_other_task_label():
    # No hardcoded 3-class map: "wrong" is just another label in the active
    # task, so this still holds if DEFAULT_TASK is ever repointed.
    for label in LABELS:
        ex = PhrasebankExample(text="x", label=label)
        assert label in synthetic_output(ex, wrong=False)
        wrong = synthetic_output(ex, wrong=True)
        assert label not in wrong
        assert any(other in wrong for other in LABELS if other != label)


def _examples() -> list[PhrasebankExample]:
    out: list[PhrasebankExample] = []
    for i in range(10):
        out.append(PhrasebankExample(text=f"pos {i}", label="positive"))
        out.append(PhrasebankExample(text=f"neg {i}", label="negative"))
        out.append(PhrasebankExample(text=f"neu {i}", label="neutral"))
    return out


def test_sample_by_label_covers_every_label():
    sample = sample_by_label(_examples(), n=6, seed=0)

    assert len(sample) == 6
    assert {e.label for e in sample} == {"positive", "negative", "neutral"}


def test_sample_by_label_is_deterministic_for_a_seed():
    first = sample_by_label(_examples(), n=9, seed=42)
    second = sample_by_label(_examples(), n=9, seed=42)

    assert [e.text for e in first] == [e.text for e in second]


def test_sample_by_label_returns_all_when_n_exceeds_available():
    examples = _examples()
    sample = sample_by_label(examples, n=999, seed=0)

    assert len(sample) == len(examples)


def test_format_row_includes_score_rationale_and_judge_model():
    row = format_row(
        "Sales climbed 9%.",
        "positive",
        "Positive tone overall.",
        {"score": 5, "rationale": "Correct and direct.", "judge_model": "opus"},
    )

    assert "positive" in row
    assert "5" in row
    assert "Correct and direct." in row
    assert "opus" in row


def test_format_row_marks_a_tool_error():
    row = format_row("x", "positive", "y", {"error": "Error executing tool"})

    assert "ERROR" in row
    assert "Error executing tool" in row


def test_negative_cases_are_all_expected_to_error():
    assert len(NEGATIVE_CASES) >= 3
    for case in NEGATIVE_CASES:
        assert set(case) == {"label", "args"}
        assert set(case["args"]) == {"input_text", "gold_label", "model_output"}
