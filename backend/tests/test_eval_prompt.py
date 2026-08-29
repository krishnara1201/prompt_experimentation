from app.eval_prompt import EVAL_PROMPT_TEMPLATE, render_eval_prompt

EXPECTED = (
    "Is the following sentence positive, negative, or neutral from a "
    "financial-news perspective? Respond with just the sentiment label.\n\n"
    "Sentence: Operating profit rose to EUR 1.5 mn."
)


def test_render_eval_prompt_exact_string():
    assert render_eval_prompt("Operating profit rose to EUR 1.5 mn.") == EXPECTED


def test_template_has_one_placeholder():
    assert EVAL_PROMPT_TEMPLATE.count("{text}") == 1
