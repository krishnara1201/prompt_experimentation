import pytest

from app.adapters.base import ModelResponse
from app.judge.scorer import JudgeParseError, JudgeResult, parse_judge_response, score_output


def test_parses_well_formed_response():
    text = "SCORE: 4\nRATIONALE: Correct sentiment but a bit terse."
    result = parse_judge_response(text)
    assert result == JudgeResult(score=4, rationale="Correct sentiment but a bit terse.")


def test_parses_response_with_extra_whitespace():
    text = "  SCORE:   5  \n  RATIONALE:   Clear and direct.  \n"
    result = parse_judge_response(text)
    assert result.score == 5
    assert result.rationale == "Clear and direct."


def test_raises_when_score_missing():
    with pytest.raises(JudgeParseError):
        parse_judge_response("RATIONALE: no score given")


def test_raises_when_rationale_missing():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 3")


def test_raises_on_out_of_range_score():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 9\nRATIONALE: out of range")


def test_raises_on_multi_digit_score():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 10\nRATIONALE: looks like ten")


class _FakeAdapter:
    def __init__(self, text):
        self._text = text

    def generate(self, prompt):
        return ModelResponse(text=self._text, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)


def test_score_output_renders_prompt_and_parses_response():
    adapter = _FakeAdapter("SCORE: 5\nRATIONALE: Nailed it.")
    result = score_output(adapter, "Profits rose.", "positive", "The tone is positive.")
    assert result.score == 5
    assert result.rationale == "Nailed it."
