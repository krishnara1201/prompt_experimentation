RUBRIC_PROMPT_TEMPLATE = """You are grading a financial-sentiment model's response.

Input text: {input_text}
Correct sentiment: {gold_label}
Model's response: {model_output}

Score the response 1-5:
5 = correctly identifies the sentiment as {gold_label}, clearly and directly
4 = correctly identifies the sentiment, but with minor clarity/formatting issues
3 = ambiguous, hedged, or only partially matches the correct sentiment
2 = identifies the wrong sentiment but the response is otherwise coherent/on-topic
1 = wrong sentiment, off-topic, malformed, or non-responsive

Respond in exactly this format:
SCORE: <1-5>
RATIONALE: <one sentence>
"""


def render_prompt(input_text: str, gold_label: str, model_output: str) -> str:
    return RUBRIC_PROMPT_TEMPLATE.format(
        input_text=input_text, gold_label=gold_label, model_output=model_output
    )
