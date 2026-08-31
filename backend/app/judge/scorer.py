import re
from dataclasses import dataclass

from app.adapters.base import ModelAdapter
from app.judge.rubric import RUBRIC_PROMPT_TEMPLATE, render_prompt

SCORE_PATTERN = re.compile(r"^\s*SCORE:\s*([1-5])\s*$", re.IGNORECASE | re.MULTILINE)
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
    adapter: ModelAdapter,
    input_text: str,
    gold_label: str,
    model_output: str,
    *,
    rubric_template: str = RUBRIC_PROMPT_TEMPLATE,
    description: str = "",
) -> JudgeResult:
    prompt = render_prompt(input_text, gold_label, model_output, template=rubric_template, description=description)
    response = adapter.generate(prompt)
    return parse_judge_response(response.text)
