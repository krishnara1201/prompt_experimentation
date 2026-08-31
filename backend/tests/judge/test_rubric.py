from app.judge.rubric import render_prompt


def test_render_prompt_includes_all_fields():
    prompt = render_prompt(
        input_text="Profits rose sharply.",
        gold_label="positive",
        model_output="The tone here is clearly positive.",
    )
    assert "Profits rose sharply." in prompt
    assert "positive" in prompt
    assert "The tone here is clearly positive." in prompt
    assert "SCORE:" in prompt
    assert "RATIONALE:" in prompt


def test_render_prompt_accepts_custom_template():
    tmpl = "Grading {description}. IN {input_text} GOLD {gold_label} OUT {model_output}"
    out = render_prompt("x", "pos", "y", template=tmpl, description="a topic task")
    assert out == "Grading a topic task. IN x GOLD pos OUT y"


def test_render_prompt_default_template_unchanged():
    from app.judge.rubric import RUBRIC_PROMPT_TEMPLATE
    out = render_prompt("x", "pos", "y")
    assert out == RUBRIC_PROMPT_TEMPLATE.format(input_text="x", gold_label="pos", model_output="y")
