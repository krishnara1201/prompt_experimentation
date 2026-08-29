from pathlib import Path

from app.adapters.base import ModelAdapter
from app.config.arms import load_judge_arm
from app.judge.scorer import score_output

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"


def _score_financial_sentiment(
    input_text: str,
    gold_label: str,
    model_output: str,
    adapter: ModelAdapter | None = None,
) -> dict:
    if adapter is None:
        adapter = load_judge_arm(str(ARMS_PATH))
    result = score_output(adapter, input_text, gold_label, model_output)
    return {"score": result.score, "rationale": result.rationale}
