from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from app.adapters.base import ModelAdapter
from app.config.arms import load_judge_arm
from app.judge.scorer import score_output

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv_if_present() -> None:
    load_dotenv(ENV_PATH)


_load_dotenv_if_present()

mcp = MCPServer("financial-sentiment-judge")


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


@mcp.tool()
def score_financial_sentiment(
    input_text: str,
    gold_label: Literal["positive", "negative", "neutral"],
    model_output: str,
) -> dict:
    """Score a candidate financial-sentiment response (1-5) against a gold label ("positive", "negative", or "neutral"), using this platform's fixed rubric judge."""
    return _score_financial_sentiment(input_text, gold_label, model_output)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
