"""The eval prompt is shared by the orchestrator (app/tasks/worker.py) and
the fine-tuning dataset builder (app/training/dataset.py) so an arm is
trained on the identical string it is later asked at eval time.

EvalExample.text is a bare dataset sentence with no task instruction. Left
unframed, different models guess the implied task with different
reliability -- the judge rubric (app/judge/rubric.py) assumes the model
attempted a classification, so the arm has to actually be asked for one.

The EVAL_PROMPT_TEMPLATE below is the default for the financial_sentiment
task and is per-arm overridable via the Arm's prompt_template field.
"""

EVAL_PROMPT_TEMPLATE = (
    "Is the following sentence positive, negative, or neutral from a "
    "financial-news perspective? Respond with just the sentiment label.\n\n"
    "Sentence: {text}"
)


def render_eval_prompt(text: str, template: str = EVAL_PROMPT_TEMPLATE) -> str:
    return template.format(text=text)
