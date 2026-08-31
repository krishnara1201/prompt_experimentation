import pytest

from app.config.tasks import (
    DEFAULT_TASK,
    InvalidTaskConfigError,
    TaskConfig,
    UnknownTaskError,
    active_task_name,
    list_tasks,
    load_task,
)
from app.eval_prompt import EVAL_PROMPT_TEMPLATE
from app.judge.rubric import RUBRIC_PROMPT_TEMPLATE

FIXTURE_DIR = "tests/data/fixtures/task_sample"


def test_load_financial_sentiment_is_byte_identical_to_legacy_constants():
    task = load_task("financial_sentiment")
    assert task.eval_prompt == EVAL_PROMPT_TEMPLATE
    assert task.rubric == RUBRIC_PROMPT_TEMPLATE
    assert task.labels == ("positive", "negative", "neutral")
    assert task.source == "financial_phrasebank_allagree"
    assert task.data_format == "phrasebank"
    assert task.data_path.is_file()


def test_load_task_returns_frozen_taskconfig():
    task = load_task("financial_sentiment")
    assert isinstance(task, TaskConfig)
    with pytest.raises(Exception):
        task.name = "x"


def test_unknown_task_raises_unknowntaskerror():
    with pytest.raises(UnknownTaskError):
        load_task("does_not_exist")


def test_list_tasks_includes_financial_sentiment():
    assert "financial_sentiment" in list_tasks()
    assert list_tasks() == sorted(list_tasks())


def test_active_task_name_defaults_when_key_absent(tmp_path):
    p = tmp_path / "arms.yaml"
    p.write_text("arms: []\n")
    assert active_task_name(str(p)) == DEFAULT_TASK


def test_active_task_name_reads_top_level_task_key(tmp_path):
    p = tmp_path / "arms.yaml"
    p.write_text("task: ag_news\narms: []\n")
    assert active_task_name(str(p)) == "ag_news"


@pytest.mark.parametrize(
    "mutation, error_fragment",
    [
        ({"labels": []}, "labels"),
        ({"eval_prompt": "no placeholder"}, "{text}"),
        ({"eval_prompt": "two {text} {bogus}"}, "bogus"),
        ({"rubric": "missing fields {gold_label}"}, "input_text"),
        ({"rubric": "all {input_text} {gold_label} {model_output} {oops}"}, "oops"),
        ({"format": "csv"}, "format"),
        ({"data": "nonexistent.jsonl"}, "data"),
        ({"label_names": ["only", "two"]}, "label_names"),
    ],
)
def test_task_config_validation_rejects_bad_config(tmp_path, mutation, error_fragment):
    import shutil, pathlib, yaml
    src = pathlib.Path(FIXTURE_DIR)
    dst = tmp_path / "mytask"
    shutil.copytree(src, dst)
    raw = yaml.safe_load((dst / "task.yaml").read_text())
    raw.update(mutation)
    (dst / "task.yaml").write_text(yaml.safe_dump(raw))

    with pytest.raises(InvalidTaskConfigError) as exc:
        load_task("mytask", tasks_dir=tmp_path)
    assert error_fragment in str(exc.value)
