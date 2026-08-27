import re
from dataclasses import dataclass

from app.adapters.base import ModelAdapter
from app.judge.rubric import render_prompt

SCORE_PATTERN = re.compile(r"SCORE:\s*([1-5])\b", re.IGNORECASE)
RATIONALE_PATTERN = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL)


class JudgeParseError(ValueError):
    pass


@dataclass
class JudgeResult:
    score: int
    rationale: str


def parse_judge_response(text: str) -> JudgeResult:
    score_match = SCORE_PATTERN.search(text)
    if not score_match:
        raise JudgeParseError(f"No SCORE found in judge response: {text!r}")

    rationale_match = RATIONALE_PATTERN.search(text)
    if not rationale_match:
        raise JudgeParseError(f"No RATIONALE found in judge response: {text!r}")

    return JudgeResult(
        score=int(score_match.group(1)),
        rationale=rationale_match.group(1).strip(),
    )


def score_output(
    adapter: ModelAdapter, input_text: str, gold_label: str, model_output: str
) -> JudgeResult:
    prompt = render_prompt(input_text, gold_label, model_output)
    response = adapter.generate(prompt)
    return parse_judge_response(response.text)
