from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from app.adapters.base import ModelAdapter
from app.config.arms import load_judge_arm
from app.config.tasks import active_task_name, load_task
from app.judge.scorer import score_output

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class ScoreResult(TypedDict):
    score: int
    rationale: str
    judge_model: str
    task: str


def _load_dotenv_if_present() -> None:
    load_dotenv(ENV_PATH)


_load_dotenv_if_present()

mcp = MCPServer("rubric-judge")


def _score_output_against_gold(
    input_text: str,
    gold_label: str,
    model_output: str,
    adapter: ModelAdapter | None = None,
) -> ScoreResult:
    if not input_text.strip():
        raise ValueError("input_text must not be empty")
    if not model_output.strip():
        raise ValueError("model_output must not be empty")
    task = load_task(active_task_name(str(ARMS_PATH)))
    if gold_label not in task.labels:
        raise ValueError(
            f"gold_label must be one of {list(task.labels)} for task {task.name!r}, "
            f"got {gold_label!r}"
        )
    if adapter is None:
        adapter = load_judge_arm(str(ARMS_PATH))
    result = score_output(
        adapter,
        input_text,
        gold_label,
        model_output,
        rubric_template=task.rubric,
        description=task.description,
    )
    return {
        "score": result.score,
        "rationale": result.rationale,
        "judge_model": getattr(adapter, "model", "unknown"),
        "task": task.name,
    }


@mcp.tool()
def score_output_against_gold(
    input_text: str,
    gold_label: str,
    model_output: str,
) -> ScoreResult:
    """Score a candidate response (1-5) against a gold label using this platform's active evaluation task rubric. The valid gold labels depend on the configured task (see GET /tasks or `pe tasks`). Returns the score, a one-sentence rationale, the judge model, and the task name."""
    return _score_output_against_gold(input_text, gold_label, model_output)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
