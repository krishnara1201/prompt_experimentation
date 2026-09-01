import pytest

from app.config.tasks import load_task
from app.data.loader import MalformedDataError, TaskExample, load_task_examples


def _task(tmp_path, jsonl_lines, labels=("a", "b")):
    import yaml
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data.jsonl").write_text("\n".join(jsonl_lines) + "\n")
    (tmp_path / "task.yaml").write_text(yaml.safe_dump({
        "name": "t", "description": "a test", "labels": list(labels),
        "source": "t", "data": "data.jsonl", "format": "jsonl",
        "eval_prompt": "x {text}",
        "rubric": "{input_text} {gold_label} {model_output}\nSCORE: <1-5>\nRATIONALE: <text>",
    }))
    return load_task("t", tasks_dir=tmp_path.parent)


def test_jsonl_happy_path(tmp_path):
    task = _task(tmp_path / "t", [
        '{"text": "hello", "gold_label": "a"}',
        '{"text": "world", "gold_label": "b"}',
    ])
    examples = load_task_examples(task)
    assert examples == [TaskExample("hello", "a"), TaskExample("world", "b")]


def test_jsonl_skips_blank_and_comment_lines(tmp_path):
    task = _task(tmp_path / "t", [
        "# a comment",
        "",
        '{"text": "hello", "gold_label": "a"}',
    ])
    assert load_task_examples(task) == [TaskExample("hello", "a")]


def test_jsonl_ignores_extra_keys(tmp_path):
    task = _task(tmp_path / "t", ['{"text": "hi", "gold_label": "a", "id": 7, "split": "train"}'])
    assert load_task_examples(task) == [TaskExample("hi", "a")]


@pytest.mark.parametrize("bad_line, fragment", [
    ('{"text": "hi"}', "gold_label"),
    ('{"gold_label": "a"}', "text"),
    ('{"text": "", "gold_label": "a"}', "text"),
    ('{"text": "hi", "gold_label": ""}', "gold_label"),
    ('{"text": "hi", "gold_label": "z"}', "z"),
    ('not json', "line 1"),
])
def test_jsonl_malformed_lines_raise_with_line_number(tmp_path, bad_line, fragment):
    task = _task(tmp_path / "t", [bad_line])
    with pytest.raises(MalformedDataError) as exc:
        load_task_examples(task)
    assert fragment in str(exc.value)


def test_phrasebank_branch_delegates(tmp_path):
    task = load_task("financial_sentiment")
    examples = load_task_examples(task)
    assert len(examples) > 2000
    assert all(isinstance(e, TaskExample) for e in examples)
    assert {e.gold_label for e in examples} == {"positive", "negative", "neutral"}
