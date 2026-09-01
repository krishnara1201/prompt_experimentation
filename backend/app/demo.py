from pathlib import Path

from dotenv import load_dotenv

from app.adapters.base import ModelResponse
from app.config.arms import load_arms
from app.config.tasks import active_task_name, load_task

ARMS_PATH = Path(__file__).resolve().parent.parent / "arms.yaml"

PROMPTS = [
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Is the following sentence positive, negative, or neutral: "
    "'The company's stock dropped sharply after the earnings call.'",
]


def format_row(arm_name: str, prompt: str, response: ModelResponse) -> str:
    cost = (
        f"${response.cost_estimate_usd:.6f}"
        if response.cost_estimate_usd is not None
        else "n/a"
    )
    return (
        f"[{arm_name}] prompt={prompt!r}\n"
        f"  text={response.text!r}\n"
        f"  latency_ms={response.latency_ms:.1f} "
        f"prompt_tokens={response.prompt_tokens} "
        f"completion_tokens={response.completion_tokens} "
        f"cost={cost}\n"
    )


def main() -> None:
    load_dotenv()
    # This demo sends raw PROMPTS straight to each adapter and never reads an
    # arm's prompt_template, so the task only matters for consistency with
    # every other load_arms call site.
    task = load_task(active_task_name(str(ARMS_PATH)))
    arms = load_arms(str(ARMS_PATH), task=task)
    for prompt in PROMPTS:
        for arm_name, arm in arms.items():
            try:
                response = arm.adapter.generate(prompt)
            except Exception as exc:
                print(f"[{arm_name}] prompt={prompt!r} FAILED: {exc}")
                continue
            print(format_row(arm_name, prompt, response))


if __name__ == "__main__":
    main()
