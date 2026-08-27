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
