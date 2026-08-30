# Task-agnostic Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eval loop task-agnostic — a new evaluation task becomes a `task.yaml` + a JSONL data file, no code changes — while keeping the Financial PhraseBank task the byte-identical default.

**Architecture:** A "task pack" is a directory `backend/tasks/<name>/` holding `task.yaml` (labels, data pointer, eval prompt, judge rubric) and a data file. A new `app/config/tasks.py` loads it into a frozen `TaskConfig`. `arms.yaml` gains a top-level `task:` key (default `financial_sentiment`). The active task's name is recorded on each `Run` and threaded through the Celery kwargs so a run's judge always renders that run's rubric. The MCP judge server and the training dataset builder read the active/configured task instead of hardcoded financial labels. Score scale stays a fixed integer 1–5, so the stats and calibration layers are untouched.

**Tech Stack:** Python 3.12, FastAPI, SQLModel, Celery, Alembic, pytest, PyYAML; React + Vite + TypeScript (frontend); `uv` package manager.

**Spec:** `docs/superpowers/specs/2026-08-30-task-agnostic-eval-design.md`

## Global Constraints

- **Score scale is fixed integer 1–5.** Do not parameterize it. The judge response parser (`SCORE: <1-5>`), `calibration.py:CORRECT_THRESHOLD = 4`, the stats layer, and the dashboard all depend on it. Only the rubric *text* and the label *set* become config.
- **`financial_sentiment` stays byte-identical.** Its `task.yaml` carries the current `app/eval_prompt.py:EVAL_PROMPT_TEMPLATE` and `app/judge/rubric.py:RUBRIC_PROMPT_TEMPLATE` strings verbatim. A regression test asserts `load_task("financial_sentiment").eval_prompt == EVAL_PROMPT_TEMPLATE` and `.rubric == RUBRIC_PROMPT_TEMPLATE`.
- **`EvalExample.source` = the task's `source` value.** The existing financial source string is `financial_phrasebank_allagree` — do not change it.
- **Backward-compatible signatures.** `load_arms`, `render_prompt`, `score_output`, `render_eval_prompt`, and the Celery task functions all gain *optional* parameters with defaults that reproduce today's behaviour. Existing call sites and test monkeypatches must keep working with, at most, a one-line signature widening.
- **Default task name literal:** `"financial_sentiment"` — used as the default in `TaskConfig` resolution, `Run.task`'s server default, and the Celery task-function default argument.
- **`uv` for everything.** Run tests from `backend/` with `uv run pytest ...`. Never `pip`.
- **Commit after every task** with a `feat:` / `test:` / `docs:` prefix and the repo's trailer:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc
  ```
- Branch: work directly on `master` (per repo convention).

---

## File Structure

**New files:**
- `backend/tasks/financial_sentiment/task.yaml` — the default task pack config.
- `backend/tasks/ag_news/task.yaml`, `backend/tasks/ag_news/data.jsonl`, `backend/tasks/ag_news/fetch_ag_news.py`, `backend/tasks/ag_news/LICENSE.txt` — the demo task pack (Task 13).
- `backend/app/config/tasks.py` — `TaskConfig`, `load_task`, `list_tasks`, `active_task_name`, validation.
- `backend/app/data/loader.py` — `TaskExample`, `load_task_examples`.
- `backend/app/api/routes/tasks.py` — `GET /tasks`.
- `backend/migrations/versions/0003_add_run_task.py` — the `run.task` column migration.
- Test files: `backend/tests/config/test_tasks.py`, `backend/tests/data/test_loader.py`, `backend/tests/api/test_tasks.py`, `backend/tests/data/fixtures/task_sample/` (fixture task pack).

**Modified files:**
- `backend/app/eval_prompt.py` — optional `template` arg on `render_eval_prompt`; docstring.
- `backend/app/judge/rubric.py` — optional `template` arg on `render_prompt`.
- `backend/app/judge/scorer.py` — optional `rubric_template` kwarg on `score_output`.
- `backend/app/config/arms.py` — `Arm.prompt_template: str | None`; `load_arms(config_path, *, task=None)`; lift `_validate_prompt_template` reuse.
- `backend/app/db/models.py` — `Run.task` field.
- `backend/app/api/routes/runs.py` — `task` request field, source filter, `Run(task=)`, enqueue `task_name`, `task` in status/summary responses.
- `backend/app/api/routes/arms.py` — resolve active task when reporting templates.
- `backend/app/api/__init__.py` / wherever routers are registered — register the tasks router.
- `backend/app/tasks/worker.py` — `task_name` threaded through `run_single_call` / `execute_call` / `run_judge_call` / `execute_judge_call`.
- `backend/scripts/seed_eval_examples.py` — `--task` argument.
- `backend/app/cli/__init__.py` — `pe seed --task`, `pe tasks`.
- `backend/app/mcp_judge_server.py` — rename server + tool, active-task validation.
- `.mcp.json` — rename server key.
- `backend/scripts/judge_tool_dryrun.py` — updated tool name + task-driven framing.
- `backend/app/training/config.py` + `backend/training.yaml` — `task` key.
- `backend/app/training/dataset.py` — task-driven labels + eval prompt.
- `backend/arms.yaml` — add `task: financial_sentiment`.
- `docker-compose.yml` — bind-mount `./backend/tasks`.
- `frontend/src/api/types.ts`, `frontend/src/api/client.ts`, `frontend/src/components/NewRunForm.tsx`, `frontend/src/components/RunDetail.tsx` (or equivalent run-header component).
- `CLAUDE.md`, `backend/README.md`.
- Existing tests: `backend/tests/api/test_runs.py`, `backend/tests/api/test_arms.py`, `backend/tests/tasks/test_execute_call.py`, `backend/tests/tasks/test_execute_judge_call.py`, `backend/tests/judge/test_rubric.py`, `backend/tests/judge/test_scorer.py`, `backend/tests/test_mcp_judge_server.py`, `backend/tests/scripts/test_seed_eval_examples.py`, `backend/tests/scripts/test_judge_tool_dryrun.py`, `backend/tests/training/test_dataset.py`.

---

## Task 1: `TaskConfig` loader + `financial_sentiment` task pack

**Files:**
- Create: `backend/tasks/financial_sentiment/task.yaml`
- Create: `backend/app/config/tasks.py`
- Create: `backend/tests/config/test_tasks.py`
- Create: `backend/tests/data/fixtures/task_sample/task.yaml` (a valid non-financial fixture pack, `format: jsonl`)
- Create: `backend/tests/data/fixtures/task_sample/data.jsonl` (3 lines)
- Reference: `backend/app/eval_prompt.py`, `backend/app/judge/rubric.py` (verbatim strings), `backend/app/config/arms.py:_validate_prompt_template` (validation pattern)

**Interfaces:**
- Produces:
  - `TaskConfig` — frozen dataclass: `name: str`, `description: str`, `labels: tuple[str, ...]`, `source: str`, `data_path: pathlib.Path` (resolved absolute), `data_format: str` (`"jsonl"` | `"phrasebank"`), `eval_prompt: str`, `rubric: str`, `label_names: tuple[str, ...] | None`.
  - `load_task(name: str) -> TaskConfig` — reads `backend/tasks/<name>/task.yaml`.
  - `list_tasks() -> list[str]` — sorted directory names under `backend/tasks/` that contain a `task.yaml`.
  - `active_task_name(arms_path: str | os.PathLike) -> str` — the top-level `task:` key in `arms.yaml`, or `"financial_sentiment"` if absent.
  - `DEFAULT_TASK = "financial_sentiment"`.
  - `InvalidTaskConfigError(ValueError)`, `UnknownTaskError(ValueError)`.
  - `TASKS_DIR: pathlib.Path` — `backend/tasks/`.

- [ ] **Step 1: Write `backend/tasks/financial_sentiment/task.yaml`**

Copy the eval prompt and rubric **verbatim** from the current code. The `eval_prompt` value must equal `app/eval_prompt.py:EVAL_PROMPT_TEMPLATE`; the `rubric` value must equal `app/judge/rubric.py:RUBRIC_PROMPT_TEMPLATE`.

```yaml
# The default evaluation task: 3-class financial-news sentiment on the
# vendored Financial PhraseBank 100%-agreement subset. See the repo README
# "Data & license" section. This pack reproduces the pre-task-pack
# hardcoded behaviour byte-for-byte.
name: financial_sentiment
description: a financial-sentiment
labels: [positive, negative, neutral]
source: financial_phrasebank_allagree
data: ../../data/financial_phrasebank/sentences_allagree.txt
format: phrasebank
label_names: [negative, neutral, positive]  # int->word, training only
eval_prompt: |-
  Is the following sentence positive, negative, or neutral from a financial-news perspective? Respond with just the sentiment label.

  Sentence: {text}
rubric: |
  You are grading a financial-sentiment model's response.

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
```

**Critical:** `EVAL_PROMPT_TEMPLATE` is `"Is the following sentence positive, negative, or neutral from a financial-news perspective? Respond with just the sentiment label.\n\nSentence: {text}"` — no trailing newline. YAML `|-` (strip) on the two-line block above produces exactly that. `RUBRIC_PROMPT_TEMPLATE` ends with `RATIONALE: <one sentence>\n` — one trailing newline — so use `|` (clip) for `rubric`. Verify both in Step 4's test.

- [ ] **Step 2: Write the failing test** — `backend/tests/config/test_tasks.py`

```python
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
```

Also create the fixture pack:

`backend/tests/data/fixtures/task_sample/task.yaml`:
```yaml
name: task_sample
description: a news-topic classification
labels: [World, Sports, Business, SciTech]
source: task_sample
data: data.jsonl
format: jsonl
eval_prompt: |-
  Classify the topic. Answer with only the label.

  Snippet: {text}
rubric: |
  You are grading a {description} model's response.

  Input text: {input_text}
  Correct label: {gold_label}
  Model's response: {model_output}

  Score 1-5 (5 = correct label {gold_label}, 1 = wrong/off-topic).

  SCORE: <1-5>
  RATIONALE: <one sentence>
```

`backend/tests/data/fixtures/task_sample/data.jsonl`:
```
{"text": "The national team won the final on penalties.", "gold_label": "Sports"}
{"text": "The central bank raised interest rates by half a point.", "gold_label": "Business"}
{"text": "Researchers unveiled a new low-power chip architecture.", "gold_label": "SciTech"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/config/test_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config.tasks'`.

- [ ] **Step 4: Implement `backend/app/config/tasks.py`**

```python
"""Task packs -- the swappable other half of "prompts as arms".

A task pack lives at ``backend/tasks/<name>/`` and carries everything the
eval loop needs that is not a model: the dataset pointer, the valid label
set, the default eval prompt, and the judge rubric. ``arms.yaml`` names the
active task with a top-level ``task:`` key (default ``financial_sentiment``).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks"
DEFAULT_TASK = "financial_sentiment"

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_REQUIRED_KEYS = {"name", "description", "labels", "source", "data", "format", "eval_prompt", "rubric"}
_VALID_FORMATS = {"jsonl", "phrasebank"}
_RUBRIC_FIELDS = {"input_text", "gold_label", "model_output", "description"}


class InvalidTaskConfigError(ValueError):
    pass


class UnknownTaskError(ValueError):
    pass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    labels: tuple[str, ...]
    source: str
    data_path: Path
    data_format: str
    eval_prompt: str
    rubric: str
    label_names: tuple[str, ...] | None


def _check_placeholders(field: str, template: str, allowed: set[str], required: set[str]) -> None:
    found = set(_PLACEHOLDER_RE.findall(template))
    missing = required - found
    if missing:
        raise InvalidTaskConfigError(
            f"'{field}' must contain the placeholder(s) {sorted('{' + m + '}' for m in missing)}"
        )
    unknown = found - allowed
    if unknown:
        raise InvalidTaskConfigError(
            f"'{field}' has unsupported placeholder(s) {sorted('{' + u + '}' for u in unknown)}; "
            f"only {sorted('{' + a + '}' for a in allowed)} are substituted"
        )


def load_task(name: str, tasks_dir: Path | None = None) -> TaskConfig:
    root = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    task_dir = root / name
    config_path = task_dir / "task.yaml"
    if not config_path.is_file():
        raise UnknownTaskError(f"no task pack '{name}' (looked for {config_path})")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise InvalidTaskConfigError(f"{config_path} must be a mapping")

    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise InvalidTaskConfigError(f"{config_path} missing key(s): {', '.join(sorted(missing))}")

    labels = raw["labels"]
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
        raise InvalidTaskConfigError("'labels' must be a non-empty list of non-empty strings")

    data_format = raw["format"]
    if data_format not in _VALID_FORMATS:
        raise InvalidTaskConfigError(f"'format' must be one of {sorted(_VALID_FORMATS)}, got {data_format!r}")

    _check_placeholders("eval_prompt", raw["eval_prompt"], {"text"}, {"text"})
    _check_placeholders("rubric", raw["rubric"], _RUBRIC_FIELDS, {"input_text", "gold_label", "model_output"})

    data_path = (task_dir / raw["data"]).resolve()
    if not data_path.is_file():
        raise InvalidTaskConfigError(f"'data' file not found: {data_path}")

    label_names = raw.get("label_names")
    if label_names is not None:
        if not isinstance(label_names, list) or set(label_names) != set(labels):
            raise InvalidTaskConfigError(
                "'label_names' must be a list covering the same set as 'labels'"
            )
        label_names = tuple(label_names)

    return TaskConfig(
        name=raw["name"],
        description=raw["description"],
        labels=tuple(labels),
        source=raw["source"],
        data_path=data_path,
        data_format=data_format,
        eval_prompt=raw["eval_prompt"],
        rubric=raw["rubric"],
        label_names=label_names,
    )


def list_tasks(tasks_dir: Path | None = None) -> list[str]:
    root = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "task.yaml").is_file())


def active_task_name(arms_path: str) -> str:
    raw = yaml.safe_load(Path(arms_path).read_text())
    if isinstance(raw, dict) and isinstance(raw.get("task"), str):
        return raw["task"]
    return DEFAULT_TASK
```

Note the `{description}` placeholder is allowed but not required in the rubric — `render_prompt` (Task 3) will pass `description` too. If a rubric omits it, `str.format` ignores the extra kwarg.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/config/test_tasks.py -v`
Expected: PASS (all).

Then the byte-identical guard specifically:
Run: `cd backend && uv run pytest tests/config/test_tasks.py::test_load_financial_sentiment_is_byte_identical_to_legacy_constants -v`
Expected: PASS. If it fails on `eval_prompt` or `rubric`, fix the YAML block-scalar style (`|-` vs `|`) — do not change the Python constants.

- [ ] **Step 6: Full regression**

Run: `cd backend && uv run pytest -q`
Expected: no new failures (this task adds files only).

- [ ] **Step 7: Commit**

```bash
git add backend/tasks/financial_sentiment backend/app/config/tasks.py backend/tests/config/test_tasks.py backend/tests/data/fixtures/task_sample
git commit -m "feat(tasks): TaskConfig loader + financial_sentiment task pack

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 2: Task-agnostic dataset loader

**Files:**
- Create: `backend/app/data/loader.py`
- Create: `backend/tests/data/test_loader.py`
- Reference: `backend/app/data/financial_phrasebank.py` (the `phrasebank` branch delegates to it)

**Interfaces:**
- Consumes: `app.config.tasks.TaskConfig` (from Task 1) — `.data_path`, `.data_format`, `.labels`.
- Produces:
  - `TaskExample` — frozen dataclass: `text: str`, `gold_label: str`.
  - `load_task_examples(task: TaskConfig) -> list[TaskExample]`.
  - `MalformedDataError(ValueError)`.

- [ ] **Step 1: Write the failing test** — `backend/tests/data/test_loader.py`

```python
import pytest

from app.config.tasks import load_task
from app.data.loader import MalformedDataError, TaskExample, load_task_examples


def _task(tmp_path, jsonl_lines, labels=("a", "b")):
    import yaml
    (tmp_path / "data.jsonl").write_text("\n".join(jsonl_lines) + "\n")
    (tmp_path / "task.yaml").write_text(yaml.safe_dump({
        "name": "t", "description": "a test", "labels": list(labels),
        "source": "t", "data": "data.jsonl", "format": "jsonl",
        "eval_prompt": "x {text}",
        "rubric": "{input_text} {gold_label} {model_output}",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/data/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.data.loader'`.

- [ ] **Step 3: Implement `backend/app/data/loader.py`**

```python
"""Loads a task pack's dataset into (text, gold_label) rows.

JSONL is the default; the ``phrasebank`` format delegates to the existing
Financial PhraseBank reader so the vendored ``sentence@label`` file needs
no conversion.
"""
import json
from dataclasses import dataclass

from app.config.tasks import TaskConfig
from app.data import financial_phrasebank


class MalformedDataError(ValueError):
    pass


@dataclass(frozen=True)
class TaskExample:
    text: str
    gold_label: str


def _load_jsonl(task: TaskConfig) -> list[TaskExample]:
    allowed = set(task.labels)
    out: list[TaskExample] = []
    with open(task.data_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedDataError(f"{task.data_path} line {lineno}: invalid JSON ({exc})") from exc
            text = obj.get("text")
            label = obj.get("gold_label")
            if not isinstance(text, str) or not text.strip():
                raise MalformedDataError(f"{task.data_path} line {lineno}: missing/empty 'text'")
            if not isinstance(label, str) or not label.strip():
                raise MalformedDataError(f"{task.data_path} line {lineno}: missing/empty 'gold_label'")
            if label not in allowed:
                raise MalformedDataError(
                    f"{task.data_path} line {lineno}: gold_label {label!r} not in task labels {sorted(allowed)}"
                )
            out.append(TaskExample(text=text, gold_label=label))
    return out


def _load_phrasebank(task: TaskConfig) -> list[TaskExample]:
    rows = financial_phrasebank.load_examples(task.data_path)
    return [TaskExample(text=r.text, gold_label=r.label) for r in rows]


def load_task_examples(task: TaskConfig) -> list[TaskExample]:
    if task.data_format == "jsonl":
        return _load_jsonl(task)
    if task.data_format == "phrasebank":
        return _load_phrasebank(task)
    raise MalformedDataError(f"unknown data format {task.data_format!r}")  # unreachable; validated at load
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/data/test_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/data/loader.py backend/tests/data/test_loader.py
git commit -m "feat(data): task-agnostic dataset loader (jsonl + phrasebank)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 3: Optional templates on the prompt/rubric/scorer helpers

**Files:**
- Modify: `backend/app/eval_prompt.py`
- Modify: `backend/app/judge/rubric.py`
- Modify: `backend/app/judge/scorer.py:score_output`
- Modify: `backend/tests/judge/test_rubric.py`, `backend/tests/judge/test_scorer.py`
- Reference: `backend/tests/test_eval_prompt.py` (must stay green unchanged)

**Interfaces:**
- Produces:
  - `render_eval_prompt(text: str, template: str = EVAL_PROMPT_TEMPLATE) -> str`
  - `render_prompt(input_text: str, gold_label: str, model_output: str, template: str = RUBRIC_PROMPT_TEMPLATE, description: str = "") -> str`
  - `score_output(adapter, input_text, gold_label, model_output, *, rubric_template: str = RUBRIC_PROMPT_TEMPLATE, description: str = "") -> JudgeResult`
- Constant names `EVAL_PROMPT_TEMPLATE` and `RUBRIC_PROMPT_TEMPLATE` are unchanged.

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/judge/test_rubric.py`:

```python
def test_render_prompt_accepts_custom_template():
    tmpl = "Grading {description}. IN {input_text} GOLD {gold_label} OUT {model_output}"
    out = render_prompt("x", "pos", "y", template=tmpl, description="a topic task")
    assert out == "Grading a topic task. IN x GOLD pos OUT y"


def test_render_prompt_default_template_unchanged():
    from app.judge.rubric import RUBRIC_PROMPT_TEMPLATE
    out = render_prompt("x", "pos", "y")
    assert out == RUBRIC_PROMPT_TEMPLATE.format(input_text="x", gold_label="pos", model_output="y")
```

Append to `backend/tests/judge/test_scorer.py`:

```python
def test_score_output_uses_custom_rubric_template():
    class _Echo:
        def generate(self, prompt):
            from app.adapters.base import ModelResponse
            assert "TOPIC-RUBRIC" in prompt
            return ModelResponse(text="SCORE: 5\nRATIONALE: ok", latency_ms=1.0,
                                 prompt_tokens=1, completion_tokens=1, cost_estimate_usd=None)
    result = score_output(_Echo(), "in", "World", "out",
                          rubric_template="TOPIC-RUBRIC {input_text} {gold_label} {model_output}")
    assert result.score == 5
```

(Check `ModelResponse`'s real field list in `app/adapters/base.py` and adjust the constructor call if needed.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && uv run pytest tests/judge/test_rubric.py tests/judge/test_scorer.py -v`
Expected: FAIL — `render_prompt() got an unexpected keyword argument 'template'`.

- [ ] **Step 3: Implement**

`backend/app/eval_prompt.py` — update the docstring's first paragraph to note the prompt is now the *default* for the `financial_sentiment` task and per-arm overridable, then:

```python
def render_eval_prompt(text: str, template: str = EVAL_PROMPT_TEMPLATE) -> str:
    return template.format(text=text)
```

`backend/app/judge/rubric.py`:

```python
def render_prompt(
    input_text: str,
    gold_label: str,
    model_output: str,
    template: str = RUBRIC_PROMPT_TEMPLATE,
    description: str = "",
) -> str:
    return template.format(
        input_text=input_text,
        gold_label=gold_label,
        model_output=model_output,
        description=description,
    )
```

`str.format` tolerates unused kwargs, so the default template (no `{description}`) still works.

`backend/app/judge/scorer.py`:

```python
def score_output(
    adapter: ModelAdapter,
    input_text: str,
    gold_label: str,
    model_output: str,
    *,
    rubric_template: str = RUBRIC_PROMPT_TEMPLATE,
    description: str = "",
) -> JudgeResult:
    prompt = render_prompt(input_text, gold_label, model_output, template=rubric_template, description=description)
    response = adapter.generate(prompt)
    return parse_judge_response(response.text)
```

Add `from app.judge.rubric import RUBRIC_PROMPT_TEMPLATE, render_prompt` to the imports.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/judge/ tests/test_eval_prompt.py -v`
Expected: PASS — including the untouched `test_eval_prompt.py` and `test_render_prompt_includes_all_fields`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/eval_prompt.py backend/app/judge/rubric.py backend/app/judge/scorer.py backend/tests/judge/test_rubric.py backend/tests/judge/test_scorer.py
git commit -m "feat(judge): optional custom template on render_prompt/score_output/render_eval_prompt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 4: `Arm.prompt_template` optional + task-aware `load_arms`

**Files:**
- Modify: `backend/app/config/arms.py`
- Modify: `backend/app/api/routes/arms.py`
- Modify: `backend/tests/config/test_arms.py`, `backend/tests/api/test_arms.py`
- Reference: Task 1 `TaskConfig`

**Interfaces:**
- Consumes: `app.config.tasks.TaskConfig`, `app.config.tasks.load_task`, `app.config.tasks.active_task_name`.
- Produces:
  - `Arm.prompt_template: str | None = None` — after `load_arms`, always a concrete string.
  - `load_arms(config_path: str, *, task: "TaskConfig | None" = None) -> dict[str, Arm]` — when `task` is given, an arm with no `prompt_template` in YAML resolves to `task.eval_prompt`; when `task` is `None`, it resolves to `EVAL_PROMPT_TEMPLATE` (today's behaviour).
  - `Arm.render(text)` unchanged (`self.prompt_template.format(text=text)`).

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/config/test_arms.py`:

```python
def test_load_arms_without_task_uses_legacy_default_template(tmp_path):
    from app.eval_prompt import EVAL_PROMPT_TEMPLATE
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
    )
    arm = load_arms(str(config_path))["a"]
    assert arm.prompt_template == EVAL_PROMPT_TEMPLATE


def test_load_arms_with_task_uses_task_eval_prompt_for_unset_arms(tmp_path):
    from app.config.tasks import load_task
    task = load_task("financial_sentiment")
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
    )
    arm = load_arms(str(config_path), task=task)["a"]
    assert arm.prompt_template == task.eval_prompt


def test_load_arms_arm_level_template_overrides_task(tmp_path):
    from app.config.tasks import load_task
    task = load_task("financial_sentiment")
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(
        "arms:\n  - name: a\n    adapter: openai_compatible\n"
        "    base_url: http://x/v1\n    model: m\n"
        '    prompt_template: "custom {text}"\n'
    )
    arm = load_arms(str(config_path), task=task)["a"]
    assert arm.prompt_template == "custom {text}"
```

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/config/test_arms.py -k "task or legacy_default" -v`
Expected: FAIL — `load_arms() got an unexpected keyword argument 'task'`.

- [ ] **Step 3: Implement `backend/app/config/arms.py`**

- Change the dataclass field: `prompt_template: str | None = None`.
- In `load_arms`, change the signature to `def load_arms(config_path: str, *, task=None) -> dict[str, "Arm"]:` (avoid importing `TaskConfig` at module top if it risks a cycle — `app.config.tasks` imports nothing from `arms`, so a top-level `from app.config.tasks import TaskConfig` under `TYPE_CHECKING` is fine; the runtime code only reads `task.eval_prompt`).
- Replace:
  ```python
  prompt_template = entry.pop("prompt_template", EVAL_PROMPT_TEMPLATE)
  _validate_prompt_template(name, prompt_template)
  ```
  with:
  ```python
  default_template = task.eval_prompt if task is not None else EVAL_PROMPT_TEMPLATE
  prompt_template = entry.pop("prompt_template", default_template)
  _validate_prompt_template(name, prompt_template)
  ```
- Keep `Arm(name=name, adapter=adapter, prompt_template=prompt_template)` — `prompt_template` is always concrete here.

`backend/app/api/routes/arms.py` — resolve the active task so `/arms` reports the template an actual run would use:

```python
from app.config.tasks import active_task_name, load_task
...
try:
    task = load_task(active_task_name(str(ARMS_PATH)))
    arms = load_arms(str(ARMS_PATH), task=task)
    raw = yaml.safe_load(ARMS_PATH.read_text())
except (OSError, ValueError) as exc:
    raise HTTPException(status_code=500, detail=f"Cannot load arms.yaml: {exc}") from exc
```

`backend/tests/api/test_arms.py` — the `FAKE_ARMS` patch target and `Arm(...)` construction: `Arm("name", object())` now yields `prompt_template=None`. If any assertion reads `prompt_template` off a `FAKE_ARMS` arm, set it explicitly in the fixture (`Arm("fake", object(), prompt_template="x {text}")`). The `@patch("app.api.routes.arms.load_arms", ...)` decorator still works; add `@patch("app.api.routes.arms.load_task")` and `@patch("app.api.routes.arms.active_task_name", return_value="financial_sentiment")` where a test drives the endpoint with a mocked `load_arms` so the real task loader isn't hit unexpectedly (the real one works too — it reads a committed file — so this is only needed if a test asserts on template text).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/config/test_arms.py tests/api/test_arms.py -v`
Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `cd backend && uv run pytest -q`
Expected: `tests/config/test_arms.py`, `tests/api/test_arms.py` green; note any `test_execute_call.py` / `test_runs.py` failures — those are fixed in Tasks 6–7, so if they fail *only* on the `load_arms(..., task=...)` kwarg, that is expected and noted here; otherwise investigate.

Actually: at this step `worker.py` and `runs.py` still call `load_arms(path)` with no `task`, which still works (task defaults to `None`). So there should be **no** new failures. If there are, stop and investigate.

- [ ] **Step 6: Commit**

```bash
git add backend/app/config/arms.py backend/app/api/routes/arms.py backend/tests/config/test_arms.py backend/tests/api/test_arms.py
git commit -m "feat(arms): prompt_template falls back to the active task's eval_prompt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 5: `Run.task` column + Alembic migration 0003

**Files:**
- Modify: `backend/app/db/models.py` (`Run`)
- Create: `backend/migrations/versions/0003_add_run_task.py`
- Create: `backend/tests/db/test_run_task_column.py`
- Reference: `backend/migrations/versions/0002_add_judge_layer.py` (style)

**Interfaces:**
- Produces: `Run.task: str` (SQLModel field, `Field(default="financial_sentiment")`), DB column `run.task` NOT NULL with `server_default='financial_sentiment'`.

- [ ] **Step 1: Write the failing test** — `backend/tests/db/test_run_task_column.py`

```python
import asyncio

import pytest
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import Run
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running")


def test_run_row_defaults_task_to_financial_sentiment():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=["a"], repeats=1, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)
            rid = run.id
        async with AsyncSession(db_test_engine) as session:
            fetched = (await session.execute(select(Run).where(Run.id == rid))).scalar_one()
            assert fetched.task == "financial_sentiment"
            await session.delete(fetched)
            await session.commit()

    asyncio.run(_run())


def test_run_row_persists_explicit_task():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=["a"], repeats=1, total_calls=1, task="ag_news")
            session.add(run)
            await session.commit()
            await session.refresh(run)
            assert run.task == "ag_news"
            await session.delete(run)
            await session.commit()

    asyncio.run(_run())
```

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/db/test_run_task_column.py -v`
Expected: FAIL — `TypeError: 'task' is an invalid keyword argument for Run` (and/or a DB error: column does not exist).

- [ ] **Step 3: Add the model field**

`backend/app/db/models.py`, in `class Run`, after `created_at`:

```python
    task: str = Field(default="financial_sentiment")
```

- [ ] **Step 4: Write the migration** — `backend/migrations/versions/0003_add_run_task.py`

```python
"""add task column to run

Revision ID: 0003_add_run_task
Revises: 0002_add_judge_layer
Create Date: 2026-08-30

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0003_add_run_task"
down_revision = "0002_add_judge_layer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "run",
        sa.Column(
            "task",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="financial_sentiment",
        ),
    )


def downgrade() -> None:
    op.drop_column("run", "task")
```

- [ ] **Step 5: Apply the migration**

Run: `cd /home/shreyash/projects/prompt_experimentation && docker compose run --rm migrate uv run alembic upgrade head`
Expected: `Running upgrade 0002_add_judge_layer -> 0003_add_run_task`.

(If working outside Docker with a local `.env` and `alembic.ini` DB URL, `cd backend && uv run alembic upgrade head` also works.)

- [ ] **Step 6: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/db/test_run_task_column.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/db/models.py backend/migrations/versions/0003_add_run_task.py backend/tests/db/test_run_task_column.py
git commit -m "feat(db): Run.task column (migration 0003)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 6: `POST /runs` — task selection, source filter, enqueue `task_name`

**Files:**
- Modify: `backend/app/api/routes/runs.py`
- Modify: `backend/tests/api/test_runs.py`
- Reference: Tasks 1, 5

**Interfaces:**
- Consumes: `active_task_name`, `load_task`, `UnknownTaskError` (Task 1); `Run.task` (Task 5); `load_arms(..., task=)` (Task 4).
- Produces:
  - `RunCreateRequest.task: str | None = None`.
  - `POST /runs` behaviour: resolve `task_name = payload.task or active_task_name(ARMS_PATH)`; `load_task(task_name)` → 422 `{"detail": "unknown task '<n>'; available: [...]"}` on `UnknownTaskError`; sample only `EvalExample WHERE source == task_cfg.source`; 400 if none; `Run(..., task=task_name)`; each `run_single_call.apply_async(kwargs=...)` includes `"task_name": task_name`.
  - `RunStatusResponse` and `RunSummary` gain `task: str`.

- [ ] **Step 1: Write failing tests**

In `backend/tests/api/test_runs.py`, add a fake-task helper and patch it. Add near `FAKE_ARMS`:

```python
from types import SimpleNamespace

FAKE_TASK = SimpleNamespace(
    name="financial_sentiment", source="test", eval_prompt="x {text}",
    rubric="{input_text} {gold_label} {model_output}", labels=("positive",),
    description="a test", label_names=None,
)
```

The existing `_insert_example()` inserts `source="test"`, so `FAKE_TASK.source = "test"` keeps every current test valid. Add `@patch("app.api.routes.runs.load_task", return_value=FAKE_TASK)` and `@patch("app.api.routes.runs.active_task_name", return_value="financial_sentiment")` to **every** existing test that patches `runs.load_arms` and posts to `/runs` (the decorator order adds args left-to-right bottom-up — update each signature accordingly). Then new tests:

```python
@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
@patch("app.api.routes.runs.active_task_name", return_value="financial_sentiment")
@patch("app.api.routes.runs.load_task", return_value=FAKE_TASK)
def test_create_run_records_task_and_threads_task_name(mock_task_cfg, mock_active, mock_arms, mock_call):
    example_id = _insert_example()
    run_id = None
    try:
        resp = TestClient(app).post("/runs", json={"repeats": 1, "sample_size": 1, "seed": 1})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        # task_name in every enqueue
        for call in mock_call.apply_async.call_args_list:
            assert call.kwargs["kwargs"]["task_name"] == "financial_sentiment"
        # persisted on the Run
        status = TestClient(app).get(f"/runs/{run_id}").json()
        assert status["task"] == "financial_sentiment"
    finally:
        if run_id is not None:
            _delete_run(run_id)
        _delete_example(example_id)


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
def test_create_run_rejects_unknown_task(mock_arms, mock_call):
    from app.config.tasks import UnknownTaskError
    with patch("app.api.routes.runs.load_task", side_effect=UnknownTaskError("nope")):
        resp = TestClient(app).post("/runs", json={"repeats": 1, "task": "bogus"})
    assert resp.status_code == 422


@patch("app.api.routes.runs.run_single_call")
@patch("app.api.routes.runs.load_arms", return_value=FAKE_ARMS)
@patch("app.api.routes.runs.active_task_name", return_value="financial_sentiment")
def test_create_run_400_when_task_has_no_seeded_examples(mock_active, mock_arms, mock_call):
    other_task = SimpleNamespace(**{**FAKE_TASK.__dict__, "source": "nothing_seeded_here"})
    with patch("app.api.routes.runs.load_task", return_value=other_task):
        resp = TestClient(app).post("/runs", json={"repeats": 1, "sample_size": 1})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: new tests FAIL; some existing tests may error on the added decorators until signatures are updated — update them in Step 1 so only the *new* assertions fail.

- [ ] **Step 3: Implement `backend/app/api/routes/runs.py`**

Imports:
```python
from app.config.tasks import UnknownTaskError, active_task_name, load_task
```

`RunCreateRequest`: add `task: str | None = None`.

In `create_run`, before loading arms:
```python
task_name = payload.task or active_task_name(str(ARMS_PATH))
try:
    task_cfg = load_task(task_name)
except UnknownTaskError as exc:
    from app.config.tasks import list_tasks
    raise HTTPException(status_code=422, detail=f"unknown task {task_name!r}; available: {list_tasks()}") from exc

available_arms = load_arms(str(ARMS_PATH), task=task_cfg)
```

Change the example query:
```python
result = await session.execute(
    select(EvalExample.id, EvalExample.text)
    .where(EvalExample.source == task_cfg.source)
    .order_by(EvalExample.id)
)
all_examples = result.all()
if not all_examples:
    raise HTTPException(
        status_code=400,
        detail=f"No eval examples for task {task_name!r}; run `pe seed --task {task_name}` first",
    )
```

`Run(...)`: add `task=task_name`.

In `_enqueue_all`, add `"task_name": task_name` to the `kwargs` dict passed to `run_single_call.apply_async`.

`RunStatusResponse`, `RunSummary`: add `task: str`. In `list_runs` populate `task=run.task`; in `get_run_status` populate `task=run.task`.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: PASS (all — existing + new).

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/runs.py backend/tests/api/test_runs.py
git commit -m "feat(runs): task selection, source-filtered sampling, task_name in enqueue

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 7: Worker — thread `task_name` through call + judge tasks

**Files:**
- Modify: `backend/app/tasks/worker.py`
- Modify: `backend/tests/tasks/test_execute_call.py`, `backend/tests/tasks/test_execute_judge_call.py`
- Reference: Tasks 1, 3, 4, 6

**Interfaces:**
- Consumes: `load_task` (Task 1); `load_arms(..., task=)` (Task 4); `score_output(..., rubric_template=, description=)` (Task 3); `"task_name"` kwarg from `POST /runs` enqueue (Task 6).
- Produces:
  - `run_single_call(self, run_id, example_id, example_text, arm_name, repeat_index, task_name: str = "financial_sentiment")`
  - `execute_call(*, ..., task_name: str = "financial_sentiment", ...)` — loads the task, calls `load_arms(str(ARMS_PATH), task=task_cfg)`, and passes `"task_name": task_name` into `run_judge_call.apply_async`.
  - `run_judge_call(self, run_result_id, task_name: str = "financial_sentiment")`
  - `execute_judge_call(*, run_result_id, task_name: str = "financial_sentiment", ...)` — loads the task, passes `rubric_template=task_cfg.rubric, description=task_cfg.description` into `score_output`.

- [ ] **Step 1: Write failing tests**

`backend/tests/tasks/test_execute_judge_call.py` — add:

```python
def test_execute_judge_call_uses_task_rubric(monkeypatch):
    seen = {}

    class _Adapter:
        def generate(self, prompt):
            seen["prompt"] = prompt
            from app.adapters.base import ModelResponse
            return ModelResponse(text="SCORE: 4\nRATIONALE: fine", latency_ms=1.0,
                                 prompt_tokens=1, completion_tokens=1, cost_estimate_usd=None)

    monkeypatch.setattr(worker, "load_judge_arm", lambda path: _Adapter())
    monkeypatch.setattr(
        worker, "_load_run_result_for_judging",
        lambda rid: _coro(("Chip firm posts record quarter.", "Business", "Business")),
    )
    fake_task = SimpleNamespace(rubric="TOPICR {input_text}|{gold_label}|{model_output}", description="a topic task")
    monkeypatch.setattr(worker, "load_task", lambda name: fake_task)
    saved = {}
    monkeypatch.setattr(worker, "_persist_judge_result", lambda **kw: _capture(saved, kw))

    worker.execute_judge_call(run_result_id=1, task_name="ag_news")
    assert seen["prompt"].startswith("TOPICR ")
    assert saved["status"] == "completed" and saved["score"] == 4
```

Adapt the helper style (`_coro`, `_capture`) to whatever the existing test module already uses for async-return and capture — check the top of `test_execute_judge_call.py` and reuse its patterns. Add `from types import SimpleNamespace`.

`backend/tests/tasks/test_execute_call.py` — the ~15 `monkeypatch.setattr(worker, "load_arms", lambda path: {...})` lines must widen to `lambda path, task=None: {...}`. Add one new test:

```python
def test_execute_call_threads_task_name_to_judge(monkeypatch):
    adapter = _StubAdapter("positive")
    monkeypatch.setattr(worker, "load_arms", lambda path, task=None: {"fake-arm": Arm("fake-arm", adapter)})
    monkeypatch.setattr(worker, "load_task", lambda name: SimpleNamespace(
        eval_prompt="x {text}", rubric="r", description="d", source="s", labels=("positive",)))
    captured = {}
    monkeypatch.setattr(worker.run_judge_call, "apply_async", lambda **kw: captured.update(kw))
    monkeypatch.setattr(worker, "_persist_run_result", lambda **kw: _coro(123))
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: SimpleNamespace(celery_queue="celery"))

    worker.execute_call(run_id=1, example_id=1, example_text="hi", arm_name="fake-arm",
                        repeat_index=0, task_name="ag_news")
    assert captured["kwargs"]["task_name"] == "ag_news"
```

Reuse the module's existing stub/coro helpers rather than inventing new ones — match names.

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/tasks/ -v`
Expected: new tests FAIL; the `load_arms` lambda-signature edits should already be in place from Step 1 so pre-existing tests still pass.

- [ ] **Step 3: Implement `backend/app/tasks/worker.py`**

Add import: `from app.config.tasks import load_task`.

`execute_call` signature: add `task_name: str = "financial_sentiment"` (keyword, before `celery_task_id`/`max_retries` is fine since all are keyword).

Inside `execute_call`, replace the config-load block:
```python
try:
    task_cfg = load_task(task_name)
    arms = load_arms(str(ARMS_PATH), task=task_cfg)
    arm = arms[arm_name]
except Exception as exc:
    ...
    _persist_failure(f"Could not resolve arm '{arm_name}' for task '{task_name}': {exc!r}")
    return
```

At the judge enqueue:
```python
run_judge_call.apply_async(
    kwargs={"run_result_id": result_id, "task_name": task_name}, queue=judge_queue
)
```

`run_single_call`: add `task_name: str = "financial_sentiment"` param and pass it to `execute_call(..., task_name=task_name)`.

`execute_judge_call`: add `task_name: str = "financial_sentiment"`. After loading `(input_text, gold_label, model_output)`:
```python
try:
    task_cfg = load_task(task_name)
except Exception as exc:
    logger.error("Could not resolve task %r for judging (run_result_id=%s): %s", task_name, run_result_id, exc)
    _persist_failure(f"Could not resolve task {task_name!r}: {exc!r}")
    return
```
Then:
```python
judge_result = _retry_model_call(
    lambda: score_output(
        judge_adapter, input_text, gold_label, model_output,
        rubric_template=task_cfg.rubric, description=task_cfg.description,
    ),
    standard_max_retries=max_retries,
)
```

`run_judge_call`: add `task_name: str = "financial_sentiment"` and pass to `execute_judge_call`.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/tasks/ -v`
Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `cd backend && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/app/tasks/worker.py backend/tests/tasks/test_execute_call.py backend/tests/tasks/test_execute_judge_call.py
git commit -m "feat(worker): thread task_name through call + judge tasks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 8: Seed script + `pe seed --task` + `pe tasks` + `GET /tasks`

**Files:**
- Modify: `backend/scripts/seed_eval_examples.py`
- Modify: `backend/app/cli/__init__.py`
- Create: `backend/app/api/routes/tasks.py`
- Modify: wherever routers register (search `include_router` — likely `backend/app/main.py`)
- Modify: `backend/arms.yaml` (add `task: financial_sentiment`)
- Modify: `docker-compose.yml` (bind-mount tasks dir)
- Modify: `backend/tests/scripts/test_seed_eval_examples.py`
- Create: `backend/tests/api/test_tasks.py`
- Modify: `backend/tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `load_task`, `list_tasks`, `active_task_name` (Task 1); `load_task_examples` (Task 2).
- Produces:
  - `scripts/seed_eval_examples.py`: `--task <name>` (default `financial_sentiment`); seeds `EvalExample(text, gold_label, source=task.source)`; idempotent per `source`. `main()` prints `Inserted N new eval examples (task=<name>, source=<source>).`
  - `GET /tasks` → `list[TaskInfo]` where `TaskInfo = {name: str, description: str, labels: list[str], active: bool, seeded_count: int}`.
  - `pe tasks` — table of `name`, `active`, `seeded` (calls `GET /tasks`).
  - `pe seed --task <name>` — passes `--task` to the container script.

- [ ] **Step 1: Write failing tests**

`backend/tests/api/test_tasks.py`:
```python
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import postgres_reachable

pytestmark = pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running")


def test_get_tasks_lists_financial_sentiment_active_by_default():
    resp = TestClient(app).get("/tasks")
    assert resp.status_code == 200
    by_name = {t["name"]: t for t in resp.json()}
    assert "financial_sentiment" in by_name
    fs = by_name["financial_sentiment"]
    assert fs["active"] is True
    assert set(fs["labels"]) == {"positive", "negative", "neutral"}
    assert fs["seeded_count"] >= 0
```

`backend/tests/scripts/test_seed_eval_examples.py` — check the existing test's shape and add one that seeds the fixture task pack (`tests/data/fixtures/task_sample`) via `--task` and asserts rows land under `source="task_sample"`. If the existing test calls `seed()` directly, add a `task` param path: `seed(task_name="task_sample", tasks_dir=...)`.

`backend/tests/cli/test_cli.py` — add a test that `pe tasks` calls `api_get("/tasks")` and renders; and `pe seed --task ag_news` includes `--task ag_news` in the compose args (mock `compose`).

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/api/test_tasks.py tests/scripts/test_seed_eval_examples.py tests/cli/test_cli.py -v`
Expected: FAIL (`/tasks` 404; seed has no `--task`).

- [ ] **Step 3: Implement**

`backend/app/api/routes/tasks.py`:
```python
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pydantic import BaseModel

from app.config.tasks import active_task_name, list_tasks, load_task
from app.db.models import EvalExample
from app.db.session import get_session

router = APIRouter(prefix="/tasks", tags=["tasks"])

ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class TaskInfo(BaseModel):
    name: str
    description: str
    labels: list[str]
    active: bool
    seeded_count: int


@router.get("", response_model=list[TaskInfo])
async def list_task_packs(session: AsyncSession = Depends(get_session)) -> list[TaskInfo]:
    active = active_task_name(str(ARMS_PATH))
    counts = dict(
        (await session.execute(
            select(EvalExample.source, func.count()).group_by(EvalExample.source)
        )).all()
    )
    out = []
    for name in list_tasks():
        cfg = load_task(name)
        out.append(TaskInfo(
            name=name, description=cfg.description, labels=list(cfg.labels),
            active=(name == active), seeded_count=counts.get(cfg.source, 0),
        ))
    return out
```

Register it: in `app/main.py` (or the router aggregator) add `app.include_router(tasks.router)` next to the others.

`backend/scripts/seed_eval_examples.py` — rewrite to be task-driven:
```python
import argparse
import asyncio

from dotenv import load_dotenv
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.tasks import DEFAULT_TASK, load_task
from app.data.loader import load_task_examples
from app.db.models import EvalExample
from app.db.session import engine

load_dotenv()


async def seed(task_name: str = DEFAULT_TASK, tasks_dir=None) -> tuple[int, str]:
    task = load_task(task_name, tasks_dir=tasks_dir)
    examples = load_task_examples(task)
    inserted = 0
    async with AsyncSession(engine) as session:
        existing = set((await session.execute(
            select(EvalExample.text).where(EvalExample.source == task.source)
        )).scalars().all())
        for ex in examples:
            if ex.text in existing:
                continue
            session.add(EvalExample(text=ex.text, gold_label=ex.gold_label, source=task.source))
            inserted += 1
        await session.commit()
    return inserted, task.source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    inserted, source = asyncio.run(seed(args.task))
    print(f"Inserted {inserted} new eval examples (task={args.task}, source={source}).")


if __name__ == "__main__":
    main()
```

`load_task` needs a `tasks_dir` param already (added in Task 1). If `load_task_examples`'s signature can't reach `tasks_dir`, that's fine — `load_task` resolves `data_path` to absolute at load, so `load_task_examples` just needs the config.

`backend/app/cli/__init__.py`:
- `seed` command: add `task: str = typer.Option("financial_sentiment", "--task", help="Task pack to seed.")` and append `"--task", task` to the compose exec args.
- New `tasks` command:
  ```python
  @app.command()
  def tasks():
      """List configured task packs (active + seeded counts)."""
      rows = api_get("/tasks")
      _render.table(
          [{"name": r["name"], "active": "*" if r["active"] else "", "seeded": r["seeded_count"]} for r in rows],
          columns=["name", "active", "seeded"],
      )
  ```

`backend/arms.yaml` — add at the top, above `arms:`:
```yaml
# The active evaluation task (a pack under backend/tasks/). Default:
# financial_sentiment. `pe tasks` lists them; `pe seed --task <name>` seeds one.
task: financial_sentiment
```

`docker-compose.yml` — extend the `&arms-config` anchor so api + worker see live task-pack edits:
```yaml
    volumes: &arms-config
      - ./backend/arms.yaml:/app/arms.yaml:ro
      - ./backend/tasks:/app/tasks:ro
```
The `migrate` service (used for `pe seed`) does **not** use `*arms-config` — add the same two mounts to `migrate` explicitly, or seeding reads the image's baked-in copy (acceptable for `financial_sentiment`, stale for a freshly-added pack). Add them to `migrate` for consistency.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/api/test_tasks.py tests/scripts/test_seed_eval_examples.py tests/cli/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke**

Run:
```bash
cd /home/shreyash/projects/prompt_experimentation
docker compose up -d --build api worker
docker compose run --rm migrate uv run python -m scripts.seed_eval_examples --task financial_sentiment
curl -s localhost:8000/tasks | python3 -m json.tool
cd backend && uv run pe tasks
```
Expected: seed reports `Inserted 0` (already seeded) or a positive count; `/tasks` shows `financial_sentiment` active with `seeded_count > 2000`.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/seed_eval_examples.py backend/app/cli/__init__.py backend/app/api/routes/tasks.py backend/app/main.py backend/arms.yaml docker-compose.yml backend/tests/api/test_tasks.py backend/tests/scripts/test_seed_eval_examples.py backend/tests/cli/test_cli.py
git commit -m "feat(tasks): pe seed --task, pe tasks, GET /tasks; mount task packs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 9: Generalise the MCP judge server

**Files:**
- Modify: `backend/app/mcp_judge_server.py`
- Modify: `.mcp.json`
- Modify: `backend/scripts/judge_tool_dryrun.py`
- Modify: `backend/tests/test_mcp_judge_server.py`, `backend/tests/scripts/test_judge_tool_dryrun.py`
- Reference: Tasks 1, 3

**Interfaces:**
- Consumes: `active_task_name`, `load_task` (Task 1); `score_output(..., rubric_template=, description=)` (Task 3).
- Produces:
  - MCP server name `rubric-judge`.
  - Tool `score_output_against_gold(input_text: str, gold_label: str, model_output: str) -> ScoreResult` where `ScoreResult = {score: int, rationale: str, judge_model: str, task: str}`.
  - Validates `gold_label` against the active task's `labels`; renders the active task's `rubric`.

- [ ] **Step 1: Write failing tests**

Update `backend/tests/test_mcp_judge_server.py`:
- Rename references `score_financial_sentiment` → `score_output_against_gold`, `_score_financial_sentiment` → `_score_output_against_gold`.
- The monkeypatch targets change: also patch `app.mcp_judge_server.load_task` to return a `SimpleNamespace(labels=("positive","negative","neutral"), rubric=RUBRIC_PROMPT_TEMPLATE, description="a financial-sentiment", name="financial_sentiment")` and `app.mcp_judge_server.active_task_name` to return `"financial_sentiment"`.
- New assertion: result has `task == "financial_sentiment"`.
- The "out-of-domain gold_label" test: `gold_label="bullish"` → `ValueError` mentioning the allowed set from the task.

Update `backend/tests/scripts/test_judge_tool_dryrun.py` for the renamed `TOOL_NAME`.

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py tests/scripts/test_judge_tool_dryrun.py -v`
Expected: FAIL (name errors).

- [ ] **Step 3: Implement `backend/app/mcp_judge_server.py`**

```python
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


load_dotenv(ENV_PATH)

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
        raise ValueError(f"gold_label must be one of {list(task.labels)} for task {task.name!r}, got {gold_label!r}")
    if adapter is None:
        adapter = load_judge_arm(str(ARMS_PATH))
    result = score_output(
        adapter, input_text, gold_label, model_output,
        rubric_template=task.rubric, description=task.description,
    )
    return {
        "score": result.score,
        "rationale": result.rationale,
        "judge_model": getattr(adapter, "model", "unknown"),
        "task": task.name,
    }


@mcp.tool()
def score_output_against_gold(input_text: str, gold_label: str, model_output: str) -> ScoreResult:
    """Score a candidate response (1-5) against a gold label using this platform's active evaluation task rubric. The valid gold labels depend on the configured task (see GET /tasks or `pe tasks`). Returns the score, a one-sentence rationale, the judge model, and the task name."""
    return _score_output_against_gold(input_text, gold_label, model_output)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

`.mcp.json` — rename the key:
```json
{
  "mcpServers": {
    "rubric-judge": {
      "command": "uv",
      "args": ["run", "--directory", "backend", "python", "-m", "app.mcp_judge_server"]
    }
  }
}
```

`backend/scripts/judge_tool_dryrun.py`:
- `TOOL_NAME = "score_output_against_gold"`.
- Replace the hardcoded financial framing: build the candidate prompt from `load_task(DEFAULT_TASK).eval_prompt` and use `load_task(DEFAULT_TASK).labels` for the "label not in set" negative case (change `"bullish"` → any string not in `labels`).
- The dataset rows still come from `financial_phrasebank` (the dryrun is a financial smoke test) — that's fine; only the tool contract changed.
- Update the module docstring.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/test_mcp_judge_server.py tests/scripts/test_judge_tool_dryrun.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/mcp_judge_server.py .mcp.json backend/scripts/judge_tool_dryrun.py backend/tests/test_mcp_judge_server.py backend/tests/scripts/test_judge_tool_dryrun.py
git commit -m "feat(mcp): generalise judge tool to the active task (rubric-judge)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 10: Generalise the training dataset builder

**Files:**
- Modify: `backend/app/training/config.py` (`TrainingConfig`)
- Modify: `backend/training.yaml`
- Modify: `backend/app/training/dataset.py`
- Modify: `backend/tests/training/test_dataset.py`
- Check: `backend/tests/scripts/test_finetune_scripts.py`, `backend/tests/scripts/test_finetune_report.py`
- Reference: Tasks 1, 3

**Interfaces:**
- Consumes: `load_task` (Task 1); `render_eval_prompt(text, template=)` (Task 3).
- Produces:
  - `TrainingConfig.task: str` (new required key in `training.yaml`; value `financial_sentiment`).
  - `dataset.py` reads label map + valid labels + eval prompt from `load_task(cfg.task)` instead of module constants `LABEL_NAMES` / `_VALID_LABELS` / bare `render_eval_prompt`.
  - `_balance_neutral` gated: only rebalances when `"neutral"` is in the task's labels **and** `cfg.balance_neutral` is true (keep the existing `balance_neutral` bool key — do not rename it).

- [ ] **Step 1: Write failing tests**

`backend/tests/training/test_dataset.py`:
- `test_record_format_matches_render_eval_prompt` currently asserts against `render_eval_prompt("Shares fell 4 percent.")`. Change it to assert against `render_eval_prompt(sentence, template=load_task("financial_sentiment").eval_prompt)` — which is byte-identical (the financial pack's `eval_prompt == EVAL_PROMPT_TEMPLATE`), so the record content is unchanged. This proves the generalization didn't shift the financial output.
- Add `test_build_uses_task_label_names`: monkeypatch `dataset.load_task` to return a `SimpleNamespace(labels=("World","Sports"), label_names=("World","Sports"), eval_prompt="topic: {text}", ...)`, monkeypatch `load_source_examples` to yield int-labelled rows, assert the written JSONL `assistant` content is the mapped word and the `user` content uses `"topic: "`.

- [ ] **Step 2: Run to verify fail**

Run: `cd backend && uv run pytest tests/training/ -v`
Expected: FAIL — `TrainingConfig` has no `task`; `dataset.load_task` undefined.

- [ ] **Step 3: Implement**

`backend/app/training/config.py` — add `task: str` to the `TrainingConfig` dataclass (place it after `run_name`). The generic `_check_type` loop handles a `str` field automatically; `_REQUIRED_KEYS`-style checks are derived from `fields()`, so no other change.

`backend/training.yaml` — add near the top:
```yaml
# Which task pack (backend/tasks/<name>/) this fine-tune targets. Its
# label set + eval prompt drive the SFT records; the leakage guard still
# checks against every seeded eval_example regardless of task.
task: financial_sentiment
```

`backend/app/training/dataset.py`:
- Add `from app.config.tasks import load_task`.
- Delete module constants `LABEL_NAMES` / `_VALID_LABELS`; instead, inside `build_sft_dataset` (and `load_source_examples`, which takes `cfg`), resolve:
  ```python
  task = load_task(cfg.task)
  valid_labels = set(task.labels)
  int_to_word = dict(enumerate(task.label_names)) if task.label_names else {}
  ```
- `load_source_examples`: replace `LABEL_NAMES[int(raw_label)]` with `int_to_word[int(raw_label)]` and raise `ValueError` if `int_to_word` is empty but an int label appears ("task.yaml needs `label_names` to train from this source").
- `_write_jsonl`: take an extra `eval_prompt: str` param, call `render_eval_prompt(sentence, template=eval_prompt)`.
- `_balance_neutral`: at its call site, guard `if cfg.balance_neutral and "neutral" in valid_labels:`.
- `LICENSE_TEXT` stays financial-specific — it is only written for the financial source. If a non-financial task is ever trained, a follow-up adds a per-task license note; out of scope here. Add a one-line comment saying so.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/training/ tests/scripts/test_finetune_scripts.py tests/scripts/test_finetune_report.py -v`
Expected: PASS. (These need the `training` extra — if `uv run pytest` skips them for missing deps, run `uv sync --extra training` first, or confirm they were already skipped before this task and remain so.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/training/config.py backend/training.yaml backend/app/training/dataset.py backend/tests/training/test_dataset.py
git commit -m "feat(training): SFT dataset builder reads labels + eval prompt from the task pack

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 11: Frontend — task dropdown + run header

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/NewRunForm.tsx`
- Modify: the run detail header component (find it: `grep -rl "run_id\|arm_names" frontend/src/components` — likely `RunDetail.tsx` or `RunView.tsx`)
- Modify/create: `frontend/src/components/NewRunForm.test.tsx` (if the project has component tests — check `frontend/` for `*.test.tsx` and the test runner in `frontend/package.json`)

**Interfaces:**
- Consumes: `GET /tasks` → `TaskInfo[]` (Task 8); `RunCreateRequest.task` (Task 6); `RunStatusResponse.task` (Task 6).
- Produces: `fetchTasks(): Promise<TaskInfo[]>`; `RunCreateRequest.task?: string`; a `<select>` in `NewRunForm`; the run's `task` shown in the run header.

- [ ] **Step 1: Check the frontend test setup**

Run: `cd frontend && cat package.json | grep -A3 '"scripts"' && ls src/**/*.test.* 2>/dev/null`
If there is a test runner (vitest/jest) and existing component tests, write a failing test first (Step 2). If there is **no** frontend test infrastructure, skip the test steps and do a manual browser check in Step 5 — do not stand up a test framework for this.

- [ ] **Step 2: (if tests exist) Write a failing test**

```tsx
// NewRunForm.test.tsx
it("defaults the task select to the active task and posts it", async () => {
  // mock fetchTasks -> [{name:'financial_sentiment', active:true, seeded_count:10, ...},
  //                     {name:'ag_news', active:false, seeded_count:0, ...}]
  // mock createRun, open the form, submit, assert createRun called with task: 'financial_sentiment'
});
```

- [ ] **Step 3: Implement**

`frontend/src/api/types.ts`:
```ts
export interface TaskInfo {
  name: string;
  description: string;
  labels: string[];
  active: boolean;
  seeded_count: number;
}
// RunCreateRequest: add
  task?: string;
// the run status/summary type: add
  task: string;
```

`frontend/src/api/client.ts`:
```ts
export async function fetchTasks(): Promise<TaskInfo[]> {
  const res = await fetch(`${API_BASE}/tasks`);
  if (!res.ok) throw new Error(`GET /tasks failed: ${res.status}`);
  return res.json();
}
```
(Match the file's existing fetch-helper style — reuse whatever wrapper `fetchArms` uses.)

`frontend/src/components/NewRunForm.tsx`:
- `const tasksQuery = useQuery({ queryKey: ['tasks'], queryFn: fetchTasks, enabled: open });`
- `const [task, setTask] = useState<string>('');`
- When `tasksQuery.data` arrives and `task === ''`, set it to the `active` task's name (a `useEffect`).
- Render a `<select>` above the Arms fieldset listing `tasksQuery.data` (label: `${t.name} (${t.seeded_count} seeded)`).
- If the chosen task's `seeded_count === 0`: show `Run \`pe seed --task <name>\` first.` and set the submit button `disabled`.
- In `handleSubmit`: `if (task) body.task = task;`.

Run header: add `Task: {run.task}` next to the arm names / created-at line.

- [ ] **Step 4: Build check**

Run: `cd frontend && npm run build`
Expected: no TypeScript errors.

- [ ] **Step 5: Manual browser check**

Run: `docker compose up -d --build frontend api` then open `http://localhost:5173`, click **New run**, confirm the task dropdown shows `financial_sentiment` selected and a run starts. Confirm the run detail page shows `Task: financial_sentiment`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src
git commit -m "feat(frontend): task dropdown in New Run form + task on run header

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 12: Docs — CLAUDE.md + backend/README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `backend/README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `CLAUDE.md`**

- In the **Architecture** section, under "Eval dataset", add that the dataset is now one *task pack* among others: `backend/tasks/<name>/task.yaml` bundles the dataset pointer, label set, default eval prompt, and judge rubric; `arms.yaml`'s top-level `task:` key selects the active one (default `financial_sentiment`, byte-identical to the former hardcoded behaviour). BYO dataset = a JSONL file + a `task.yaml`.
- Under "Judge layer", note the rubric is now per-task (from the task pack), still fixed 1–5.
- Add a **Build phases** entry:
  ```
  8. **Task-agnostic eval** ✅ **Done.** The eval loop is no longer hardwired
     to financial sentiment: a task is a pack under `backend/tasks/<name>/`
     (`task.yaml` + JSONL data) declaring labels, eval prompt, and judge
     rubric. `arms.yaml` `task:` selects the active one; `Run.task` records
     it and it is threaded through the judge task. MCP judge server and the
     training dataset builder read the active/configured task too. Score
     scale stays 1–5 so the stats/calibration layers are unchanged. Spec:
     `docs/superpowers/specs/2026-08-30-task-agnostic-eval-design.md`; plan:
     `docs/superpowers/plans/2026-08-30-task-agnostic-eval.md`. Demo:
     Deliverable 2 (AG News prompt A/B) — see
     `docs/superpowers/reports/` once run.
  ```
- In **Open decisions**, update the "dataset stays a swap-in point" note to say the swap-in is now concrete (task packs), not just aspirational.

- [ ] **Step 2: Update `backend/README.md`**

- Add a "Bring your own task" subsection near the dataset/"Data & license" notes: the three files a task pack needs, the `task.yaml` schema, `pe seed --task <name>`, `pe tasks`, and setting `arms.yaml` `task:`.
- Note the `./backend/tasks` bind mount (edit packs without a rebuild).
- Note that the MCP tool is now `score_output_against_gold` on server `rubric-judge` and its valid labels follow the active task.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md backend/README.md
git commit -m "docs(phase8): task-agnostic eval — task packs, pe tasks, rubric-judge

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

- [ ] **Step 4: Full regression + lint**

Run: `cd backend && uv run pytest -q && uv run ruff check .`
Expected: all green. Fix any lint from the new files.

---

## Task 13: AG News task pack (data plumbing for Deliverable 2)

**Files:**
- Create: `backend/tasks/ag_news/task.yaml`
- Create: `backend/tasks/ag_news/fetch_ag_news.py`
- Create: `backend/tasks/ag_news/data.jsonl` (generated by the fetch script, then committed)
- Create: `backend/tasks/ag_news/LICENSE.txt`
- Modify: `backend/README.md` ("Data & license" — add AG News attribution)
- Create: `backend/tests/tasks/test_ag_news_pack.py`

**Interfaces:**
- Consumes: `load_task`, `load_task_examples` (Tasks 1–2).
- Produces: a seedable `ag_news` task pack; `source: ag_news_sample`.

- [ ] **Step 1: Write `backend/tasks/ag_news/fetch_ag_news.py`**

```python
"""Regenerates backend/tasks/ag_news/data.jsonl: a stratified sample of AG
News (topic classification) for the prompt-A/B demo.

    uv run --with datasets python backend/tasks/ag_news/fetch_ag_news.py --per-class 30

AG News is the AG corpus (ComeToMyHead news aggregator), redistributed for
research via Hugging Face. Only this small sample is vendored; see LICENSE.txt.
"""
import argparse
import json
import random
from pathlib import Path

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
OUT = Path(__file__).resolve().parent / "data.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="test")
    by_label: dict[int, list[str]] = {k: [] for k in LABELS}
    for row in ds:
        by_label[row["label"]].append(row["text"].replace("\n", " ").strip())

    rng = random.Random(args.seed)
    rows = []
    for label_int, word in LABELS.items():
        picks = rng.sample(by_label[label_int], args.per_class)
        rows.extend({"text": t, "gold_label": word} for t in picks)
    rng.shuffle(rows)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `backend/tasks/ag_news/task.yaml`**

```yaml
name: ag_news
description: a news-topic classification
labels: [World, Sports, Business, Sci/Tech]
source: ag_news_sample
data: data.jsonl
format: jsonl
label_names: [World, Sports, Business, Sci/Tech]
eval_prompt: |-
  Classify the topic of the news snippet below as exactly one of: World, Sports, Business, Sci/Tech. Answer with only the label.

  Snippet: {text}
rubric: |
  You are grading a news-topic classification model's response.

  Input text: {input_text}
  Correct label: {gold_label}
  Model's response: {model_output}

  Score the response 1-5:
  5 = correctly identifies the label as {gold_label}, clearly and directly
  4 = correct label, with minor clarity/formatting issues
  3 = ambiguous, hedged, or only partially matches the correct label
  2 = identifies the wrong label but the response is otherwise coherent/on-topic
  1 = wrong label, off-topic, malformed, or non-responsive

  Respond in exactly this format:
  SCORE: <1-5>
  RATIONALE: <one sentence>
```

- [ ] **Step 3: Write `backend/tasks/ag_news/LICENSE.txt`**

Cite: AG News derived from "AG's corpus of news articles" (ComeToMyHead), popularized by Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for Text Classification", NeurIPS. Redistributed for non-commercial research. Only a 120-row sample is vendored here.

- [ ] **Step 4: Generate the data**

Run: `cd /home/shreyash/projects/prompt_experimentation && uv run --with datasets --project backend python backend/tasks/ag_news/fetch_ag_news.py --per-class 30`
Expected: `wrote 120 rows to .../data.jsonl`.

- [ ] **Step 5: Write the test** — `backend/tests/tasks/test_ag_news_pack.py`

```python
from app.config.tasks import load_task
from app.data.loader import load_task_examples


def test_ag_news_pack_loads_and_is_balanced():
    task = load_task("ag_news")
    examples = load_task_examples(task)
    assert len(examples) == 120
    counts = {}
    for e in examples:
        counts[e.gold_label] = counts.get(e.gold_label, 0) + 1
    assert set(counts) == {"World", "Sports", "Business", "Sci/Tech"}
    assert all(c == 30 for c in counts.values())
```

- [ ] **Step 6: Run tests**

Run: `cd backend && uv run pytest tests/tasks/test_ag_news_pack.py tests/config/test_tasks.py -v`
Expected: PASS. `list_tasks()` now returns `["ag_news", "financial_sentiment"]`.

- [ ] **Step 7: Commit**

```bash
git add backend/tasks/ag_news backend/tests/tasks/test_ag_news_pack.py backend/README.md
git commit -m "feat(tasks): AG News task pack (topic classification) for the prompt A/B demo

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Deliverable 2 — the executed prompt A/B (manual follow-up, not a code task)

After Tasks 1–13 are merged, run the comparison and write it up. This is
operational work (a live model run + human labeling + prose), tracked here
so it is not forgotten — do **not** try to automate it into a task above.

1. Add two AG News arms to `arms.yaml` (same `model: qwen3:8b`, differing
   `prompt_template` only):
   - `ag-news-terse`: `"Classify the topic (World, Sports, Business, Sci/Tech) of this snippet. /no_think Answer with only the label.\n\n{text}"`
   - `ag-news-cot`: a step-by-step template ending `Label: <one of World, Sports, Business, Sci/Tech>` (keep `/no_think` so the *arm* prompt, not Qwen3 thinking mode, is the manipulated variable).
2. `pe seed --task ag_news`
3. `pe run --task ag_news --arm ag-news-terse --arm ag-news-cot -n 120 -r 3`
4. `pe calibrate select --run-id <id> --size 30` → hand-label → `pe calibrate import` → `pe calibrate report`. Record judge/human agreement for AG News **before** trusting judge scores.
5. `pe stats compare --run-id <id>`, `pe stats equivalence --run-id <id>`, `pe stats power --run-id <id>`.
6. Build the frontier scatter (reuse the finetune report's plotting approach).
7. Write `docs/superpowers/reports/2026-08-31-prompt-ab-comparison.md` +
   `2026-08-31-prompt-ab-frontier.png`, mirroring
   `docs/superpowers/reports/2026-08-30-finetune-comparison.md`. Report the
   result honestly whichever way it falls; the framing is that the paired
   test resolves a small effect an unpaired win-rate would miss.
8. Update `CLAUDE.md`: flip the "feature exists but no run has used it" gap
   in the Phase 8 note; link the report.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| Task pack directory + `task.yaml` schema | 1, 13 |
| `app/config/tasks.py` (`TaskConfig`, `load_task`, `list_tasks`, `active_task_name`) | 1 |
| `app/data/loader.py` (jsonl + phrasebank) | 2 |
| `financial_sentiment` byte-identical + regression test | 1 (Step 2/5) |
| `eval_prompt.py` optional template | 3 |
| `arms.py` `Arm.prompt_template: str \| None`, `load_arms(task=)` | 4 |
| `GET /arms` resolves active task | 4 |
| `Run.task` column + migration 0003 | 5 |
| `POST /runs` task param, source filter, enqueue `task_name`, response fields | 6 |
| worker `task_name` threading (call + judge) | 7 |
| `score_output(rubric_template=)` | 3 (helper), 7 (wired) |
| judge rubric `RUBRIC_PROMPT_TEMPLATE` → optional template | 3 |
| seed script `--task` | 8 |
| `pe seed --task`, `pe tasks` | 8 |
| `GET /tasks` endpoint | 8 |
| MCP server rename + tool rename + active-task validation + `.mcp.json` | 9 |
| `judge_tool_dryrun.py` update | 9 |
| training builder generalization (`training.yaml` `task:`, label map, eval prompt, balance guard) | 10 |
| frontend task dropdown + `GET /tasks` client + run header | 11 |
| `docker-compose.yml` tasks bind mount | 8 |
| `arms.yaml` `task: financial_sentiment` | 8 |
| CLAUDE.md + README docs | 12 |
| Deliverable 2 (AG News pack) | 13 + manual follow-up |
| Error handling: unknown task 422, zero-seeded 400, malformed task.yaml, bad jsonl line, MCP bad label, in-flight Celery defaults | 1, 2, 6, 7, 9 |
| Regression: stats/calibration untouched | verified by `uv run pytest -q` in 4/7/12 |

No gaps.

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases"/"similar to Task N". Every code step has real code. Frontend Task 11 explicitly branches on "tests exist or not" rather than hand-waving — acceptable because the frontend test setup is unknown until inspected, and the step says exactly what to do in each case.

**3. Type consistency:**
- `TaskConfig` fields (`name`, `description`, `labels`, `source`, `data_path`, `data_format`, `eval_prompt`, `rubric`, `label_names`) — used consistently in Tasks 2, 4, 6, 7, 8, 9, 10.
- `load_task(name, tasks_dir=None)` — the `tasks_dir` kwarg is defined in Task 1 and used in Task 1's tests and Task 8's `seed(...)`. Consistent.
- `load_arms(config_path, *, task=None)` — Task 4 defines; Tasks 6, 7 call with `task=task_cfg`; existing monkeypatches widened to `lambda path, task=None:` in Task 7. Consistent.
- `score_output(..., *, rubric_template=..., description=...)` — Task 3 defines; Task 7 and Task 9 call with both. Consistent.
- `render_prompt(..., template=..., description=...)` — Task 3 defines; Task 3 tests use `template=`/`description=`. Consistent.
- `TaskExample(text, gold_label)` — Task 2 defines; Tasks 2, 13 tests use positionally. Consistent.
- Celery task defaults: `task_name: str = "financial_sentiment"` on `run_single_call`, `execute_call`, `run_judge_call`, `execute_judge_call` — Task 7, matches Global Constraints literal.
- `TaskInfo` (`name`, `description`, `labels`, `active`, `seeded_count`) — Task 8 (backend), Task 11 (frontend `types.ts`). Consistent.
- `Run.task` — Task 5 model field, Task 6 reads `run.task`, migration column name `task`. Consistent.

No inconsistencies found.
