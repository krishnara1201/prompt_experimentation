# Judge Layer & Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 3 of the platform — a rubric-based LLM-as-judge that
automatically scores every completed `RunResult`, plus a calibration
workflow (stratified sample → hand-labeled JSON → import → Spearman/Cohen's
kappa report) that must be run and read before judge scores on a full run
are trusted.

**Architecture:** A new `app/judge/` package (`rubric.py`, `scorer.py`,
`calibration.py`) reuses the existing `ModelAdapter.generate(prompt)`
interface unchanged. A new Celery task `run_judge_call` is chained onto
`execute_call` in `app/tasks/worker.py` immediately after a successful
generation persists, mirroring the existing retry/backoff pattern. Judge
results land on new `run_result.judge_*` columns; a separate
`judge_calibration_label` table holds human scores used only to check
judge/human agreement, never conflated with `judge_score`. Three CLI
scripts under `backend/scripts/` (select / import / report) drive the
calibration workflow, following the existing `seed_eval_examples.py` idiom.

**Tech Stack:** Python, SQLModel, Alembic, Celery, httpx (via existing
adapters), scipy (new dependency, Spearman correlation only — Cohen's kappa
is implemented manually to avoid pulling in sklearn for one metric), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-judge-layer-calibration-design.md`

## Global Constraints

- Judge prompt is fixed and reference-guided (shown the correct
  `gold_label`), scoring 1-5, in exactly this format:
  ```
  SCORE: <1-5>
  RATIONALE: <one sentence>
  ```
- The judge model is configured via a **new top-level `judge:` key** in
  `arms.yaml` — never inside the `arms:` list, so it can never be selected
  as an eval arm.
- Judging is auto-chained: `execute_call` enqueues `run_judge_call` only
  after a successful (`status="completed"`) generation persists. A judge
  failure sets `judge_status="failed"` but must never change
  `RunResult.status`.
- "Correct" for Cohen's kappa binarization is `score >= 4`.
- No new DB table stores the calibration *report* — it's recomputed on
  demand from `judge_calibration_label` + `run_result`.
- All new async DB code follows the existing pattern: production code
  imports `engine` from `app.db.session`; tests monkeypatch that module's
  `engine` reference to `tests.conftest.db_test_engine` (`NullPool`) and are
  skipped via `pytestmark = pytest.mark.skipif(not postgres_reachable(), ...)`
  when Postgres isn't running.
- CLI scripts print human-readable output and expose an `async def` /
  pure-function core that tests call directly (matches
  `scripts/seed_eval_examples.py`'s `seed()` split from `main()`).
- Never write a real email address into committed code — use a placeholder
  like `you@example.com` in docs/tests; the actual `--labeled-by` value is
  supplied at runtime by whoever imports labels.

---

### Task 1: Add scipy dependency

**Files:**
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Produces: `scipy` importable as `from scipy.stats import spearmanr` for Task 5.

- [ ] **Step 1: Add the dependency**

From `backend/`:

```bash
uv add scipy
```

This updates `pyproject.toml` and `uv.lock`.

- [ ] **Step 2: Verify it installed**

Run: `uv run python -c "from scipy.stats import spearmanr; print(spearmanr([1,2,3],[1,2,3]))"`
Expected: prints a `SignificanceResult` with `correlation=1.0`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add scipy dependency for judge calibration stats"
```

---

### Task 2: Data model — judge columns, calibration table, migration

**Files:**
- Modify: `backend/app/db/models.py`
- Modify: `backend/migrations/env.py`
- Create: `backend/migrations/versions/0002_add_judge_layer.py`
- Test: `backend/tests/db/test_models.py`

**Interfaces:**
- Produces: `RunResult.judge_rationale: str | None`,
  `RunResult.judge_status: str` (default `"pending"`),
  `RunResult.judge_error_message: str | None`,
  `RunResult.judge_celery_task_id: str | None`; new model
  `JudgeCalibrationLabel(id, run_result_id, human_score, labeled_by, notes,
  labeled_at)`.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/db/test_models.py`:

```python
def test_run_result_judge_columns_default_correctly():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            await session.commit()
            await session.refresh(example)

            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            assert result.judge_status == "pending"
            assert result.judge_score is None
            assert result.judge_rationale is None
            assert result.judge_error_message is None
            assert result.judge_celery_task_id is None

            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def test_judge_calibration_label_round_trip():
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose sharply.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            result = RunResult(
                run_id=run.id, example_id=example.id, arm_name="fake-arm",
                repeat_index=0, status="completed", judge_score=4, judge_status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            label = JudgeCalibrationLabel(
                run_result_id=result.id, human_score=4, labeled_by="you@example.com", notes="agrees"
            )
            session.add(label)
            await session.commit()
            await session.refresh(label)

            fetched = await session.get(JudgeCalibrationLabel, label.id)
            assert fetched is not None
            assert fetched.human_score == 4
            assert fetched.labeled_by == "you@example.com"
            assert fetched.labeled_at is not None

            await session.delete(label)
            await session.delete(result)
            await session.delete(run)
            await session.delete(example)
            await session.commit()

    asyncio.run(_run())
```

Update the import line at the top of the file to add `JudgeCalibrationLabel`:

```python
from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'JudgeCalibrationLabel'`
(or skipped if Postgres isn't running — start it first: `docker compose up -d postgres` from the repo root).

- [ ] **Step 3: Add the columns and model**

In `backend/app/db/models.py`, modify `RunResult` — add these fields right after the existing `judge_score` field:

```python
    judge_score: Optional[float] = Field(default=None)
    judge_rationale: Optional[str] = Field(default=None)
    judge_status: str = Field(default="pending")
    judge_error_message: Optional[str] = Field(default=None)
    judge_celery_task_id: Optional[str] = Field(default=None)
```

Add a new model at the end of the file:

```python
class JudgeCalibrationLabel(SQLModel, table=True):
    __tablename__ = "judge_calibration_label"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_result_id: int = Field(foreign_key="run_result.id", index=True)
    human_score: int
    labeled_by: str
    notes: Optional[str] = Field(default=None)
    labeled_at: datetime = Field(default_factory=_utcnow)
```

- [ ] **Step 4: Create the migration**

Create `backend/migrations/versions/0002_add_judge_layer.py`:

```python
"""add judge columns to run_result, create judge_calibration_label

Revision ID: 0002_add_judge_layer
Revises: 0001_create_initial_tables
Create Date: 2026-08-27

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0002_add_judge_layer"
down_revision = "0001_create_initial_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("run_result", sa.Column("judge_rationale", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column(
        "run_result",
        sa.Column("judge_status", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="pending"),
    )
    op.add_column("run_result", sa.Column("judge_error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column("run_result", sa.Column("judge_celery_task_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    op.create_table(
        "judge_calibration_label",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_result_id", sa.Integer(), nullable=False),
        sa.Column("human_score", sa.Integer(), nullable=False),
        sa.Column("labeled_by", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("labeled_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_result_id"], ["run_result.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_judge_calibration_label_run_result_id", "judge_calibration_label", ["run_result_id"])


def downgrade() -> None:
    op.drop_index("ix_judge_calibration_label_run_result_id", table_name="judge_calibration_label")
    op.drop_table("judge_calibration_label")
    op.drop_column("run_result", "judge_celery_task_id")
    op.drop_column("run_result", "judge_error_message")
    op.drop_column("run_result", "judge_status")
    op.drop_column("run_result", "judge_rationale")
```

Update `backend/migrations/env.py`'s import line so `SQLModel.metadata` registers the new table:

```python
from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult  # noqa: F401 -- registers metadata
```

Apply it: `cd backend && uv run alembic upgrade head`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: PASS (or skipped if Postgres isn't running).

- [ ] **Step 6: Commit**

```bash
git add app/db/models.py migrations/env.py migrations/versions/0002_add_judge_layer.py tests/db/test_models.py
git commit -m "feat: add judge columns and judge_calibration_label table"
```

---

### Task 3: Judge rubric prompt

**Files:**
- Create: `backend/app/judge/__init__.py` (empty)
- Create: `backend/app/judge/rubric.py`
- Test: `backend/tests/judge/__init__.py` (empty)
- Test: `backend/tests/judge/test_rubric.py`

**Interfaces:**
- Produces: `render_prompt(input_text: str, gold_label: str, model_output: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/judge/test_rubric.py`:

```python
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
```

Create empty `backend/app/judge/__init__.py` and `backend/tests/judge/__init__.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/judge/test_rubric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.judge.rubric'`

- [ ] **Step 3: Write the rubric module**

Create `backend/app/judge/rubric.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/judge/test_rubric.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/judge/__init__.py app/judge/rubric.py tests/judge/__init__.py tests/judge/test_rubric.py
git commit -m "feat: add judge rubric prompt template"
```

---

### Task 4: Judge scorer (calls the judge adapter, parses its response)

**Files:**
- Create: `backend/app/judge/scorer.py`
- Test: `backend/tests/judge/test_scorer.py`

**Interfaces:**
- Consumes: `render_prompt` from Task 3 (`app.judge.rubric`); `ModelAdapter`,
  `ModelResponse` from `app.adapters.base` (existing, unchanged).
- Produces: `JudgeResult(score: int, rationale: str)` (dataclass);
  `JudgeParseError(ValueError)`; `parse_judge_response(text: str) ->
  JudgeResult`; `score_output(adapter: ModelAdapter, input_text: str,
  gold_label: str, model_output: str) -> JudgeResult` — used by Task 7's
  Celery task.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/judge/test_scorer.py`:

```python
import pytest

from app.adapters.base import ModelResponse
from app.judge.scorer import JudgeParseError, JudgeResult, parse_judge_response, score_output


def test_parses_well_formed_response():
    text = "SCORE: 4\nRATIONALE: Correct sentiment but a bit terse."
    result = parse_judge_response(text)
    assert result == JudgeResult(score=4, rationale="Correct sentiment but a bit terse.")


def test_parses_response_with_extra_whitespace():
    text = "  SCORE:   5  \n  RATIONALE:   Clear and direct.  \n"
    result = parse_judge_response(text)
    assert result.score == 5
    assert result.rationale == "Clear and direct."


def test_raises_when_score_missing():
    with pytest.raises(JudgeParseError):
        parse_judge_response("RATIONALE: no score given")


def test_raises_when_rationale_missing():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 3")


def test_raises_on_out_of_range_score():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 9\nRATIONALE: out of range")


def test_raises_on_multi_digit_score():
    with pytest.raises(JudgeParseError):
        parse_judge_response("SCORE: 10\nRATIONALE: looks like ten")


class _FakeAdapter:
    def __init__(self, text):
        self._text = text

    def generate(self, prompt):
        return ModelResponse(text=self._text, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)


def test_score_output_renders_prompt_and_parses_response():
    adapter = _FakeAdapter("SCORE: 5\nRATIONALE: Nailed it.")
    result = score_output(adapter, "Profits rose.", "positive", "The tone is positive.")
    assert result.score == 5
    assert result.rationale == "Nailed it."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/judge/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.judge.scorer'`

- [ ] **Step 3: Write the scorer module**

Create `backend/app/judge/scorer.py`:

```python
import re
from dataclasses import dataclass

from app.adapters.base import ModelAdapter
from app.judge.rubric import render_prompt

SCORE_PATTERN = re.compile(r"SCORE:\s*([1-5])\b", re.IGNORECASE)
RATIONALE_PATTERN = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL)


class JudgeParseError(ValueError):
    pass


@dataclass
class JudgeResult:
    score: int
    rationale: str


def parse_judge_response(text: str) -> JudgeResult:
    score_match = SCORE_PATTERN.search(text)
    if not score_match:
        raise JudgeParseError(f"No SCORE found in judge response: {text!r}")

    rationale_match = RATIONALE_PATTERN.search(text)
    if not rationale_match:
        raise JudgeParseError(f"No RATIONALE found in judge response: {text!r}")

    return JudgeResult(
        score=int(score_match.group(1)),
        rationale=rationale_match.group(1).strip(),
    )


def score_output(
    adapter: ModelAdapter, input_text: str, gold_label: str, model_output: str
) -> JudgeResult:
    prompt = render_prompt(input_text, gold_label, model_output)
    response = adapter.generate(prompt)
    return parse_judge_response(response.text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/judge/test_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/judge/scorer.py tests/judge/test_scorer.py
git commit -m "feat: add judge scorer that calls a judge adapter and parses its response"
```

---

### Task 5: Calibration statistics (Spearman correlation, Cohen's kappa)

**Files:**
- Create: `backend/app/judge/calibration.py`
- Test: `backend/tests/judge/test_calibration.py`

**Interfaces:**
- Consumes: `scipy.stats.spearmanr` (Task 1).
- Produces: `cohens_kappa(pairs: list[tuple[float, int]], threshold: int =
  4) -> float`; `calibration_report(pairs: list[tuple[float, int]]) ->
  dict` with keys `n`, `spearman_r`, `spearman_p`, `cohens_kappa`,
  `mean_abs_diff` — used by Task 10's `calibration_report.py` script.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/judge/test_calibration.py`:

```python
import pytest

from app.judge.calibration import calibration_report, cohens_kappa


def test_cohens_kappa_perfect_agreement():
    pairs = [(5, 5), (1, 1), (5, 5), (1, 1), (3, 2)]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_cohens_kappa_no_agreement():
    pairs = [(5, 1), (1, 5), (5, 1), (1, 5)]
    assert cohens_kappa(pairs) == pytest.approx(-1.0)


def test_cohens_kappa_requires_at_least_one_pair():
    with pytest.raises(ValueError):
        cohens_kappa([])


def test_calibration_report_computes_all_metrics():
    pairs = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
    report = calibration_report(pairs)

    assert report["n"] == 5
    assert report["spearman_r"] == pytest.approx(1.0)
    assert report["cohens_kappa"] == pytest.approx(1.0)
    assert report["mean_abs_diff"] == pytest.approx(0.0)


def test_calibration_report_requires_at_least_one_pair():
    with pytest.raises(ValueError):
        calibration_report([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/judge/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.judge.calibration'`

- [ ] **Step 3: Write the calibration module**

Create `backend/app/judge/calibration.py`:

```python
from scipy.stats import spearmanr

CORRECT_THRESHOLD = 4


def cohens_kappa(pairs: list[tuple[float, int]], threshold: int = CORRECT_THRESHOLD) -> float:
    n = len(pairs)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one pair")

    a = b = c = d = 0
    for judge_score, human_score in pairs:
        judge_correct = judge_score >= threshold
        human_correct = human_score >= threshold
        if judge_correct and human_correct:
            a += 1
        elif judge_correct and not human_correct:
            b += 1
        elif not judge_correct and human_correct:
            c += 1
        else:
            d += 1

    po = (a + d) / n
    judge_correct_rate = (a + b) / n
    human_correct_rate = (a + c) / n
    pe = judge_correct_rate * human_correct_rate + (1 - judge_correct_rate) * (1 - human_correct_rate)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def calibration_report(pairs: list[tuple[float, int]]) -> dict:
    n = len(pairs)
    if n == 0:
        raise ValueError("calibration_report requires at least one (judge_score, human_score) pair")

    judge_scores = [j for j, _ in pairs]
    human_scores = [h for _, h in pairs]

    spearman_r, spearman_p = spearmanr(judge_scores, human_scores)
    mean_abs_diff = sum(abs(j - h) for j, h in pairs) / n

    return {
        "n": n,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "cohens_kappa": cohens_kappa(pairs),
        "mean_abs_diff": mean_abs_diff,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/judge/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/judge/calibration.py tests/judge/test_calibration.py
git commit -m "feat: add Spearman correlation and Cohen's kappa for judge calibration"
```

---

### Task 6: Judge model config (`judge:` key in arms.yaml)

**Files:**
- Modify: `backend/app/config/arms.py`
- Modify: `backend/arms.yaml`
- Test: `backend/tests/config/test_arms.py`

**Interfaces:**
- Produces: `InvalidJudgeConfigError(ValueError)`; `load_judge_arm(config_path:
  str) -> ModelAdapter` — used by Task 7's `execute_judge_call`.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/config/test_arms.py`. First update the import line:

```python
from app.config.arms import (
    InvalidArmConfigError,
    InvalidJudgeConfigError,
    UnknownAdapterError,
    load_arms,
    load_judge_arm,
)
```

Add new fixtures and tests:

```python
VALID_JUDGE_CONFIG = (
    VALID_CONFIG
    + """
judge:
  adapter: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY
"""
)

MISSING_JUDGE_ADAPTER_CONFIG = (
    VALID_CONFIG
    + """
judge:
  model: claude-haiku-4-5-20251001
"""
)

UNKNOWN_JUDGE_ADAPTER_CONFIG = (
    VALID_CONFIG
    + """
judge:
  adapter: telepathy
  model: mind-reader-v1
"""
)


def test_load_judge_arm_builds_correct_adapter(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_JUDGE_CONFIG)

    judge = load_judge_arm(str(config_path))

    assert isinstance(judge, AnthropicAdapter)
    assert judge.model == "claude-haiku-4-5-20251001"


def test_load_judge_arm_raises_when_judge_key_missing(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)  # no judge: key at all

    with pytest.raises(InvalidJudgeConfigError):
        load_judge_arm(str(config_path))


def test_load_judge_arm_raises_when_adapter_missing(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(MISSING_JUDGE_ADAPTER_CONFIG)

    with pytest.raises(InvalidJudgeConfigError):
        load_judge_arm(str(config_path))


def test_load_judge_arm_raises_on_unknown_adapter_type(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(UNKNOWN_JUDGE_ADAPTER_CONFIG)

    with pytest.raises(UnknownAdapterError):
        load_judge_arm(str(config_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_judge_arm'`

- [ ] **Step 3: Implement `load_judge_arm`**

In `backend/app/config/arms.py`, add a new exception class next to the
existing ones:

```python
class InvalidJudgeConfigError(ValueError):
    pass
```

Add the function at the end of the file:

```python
def load_judge_arm(config_path: str) -> ModelAdapter:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not isinstance(raw.get("judge"), dict):
        raise InvalidJudgeConfigError(f"'{config_path}' must have a top-level 'judge' mapping")

    entry = dict(raw["judge"])
    if "adapter" not in entry:
        raise InvalidJudgeConfigError("'judge' entry is missing required key 'adapter'")

    adapter_type = entry.pop("adapter")
    adapter_cls = ADAPTER_TYPES.get(adapter_type)
    if adapter_cls is None:
        raise UnknownAdapterError(f"Unknown adapter type '{adapter_type}' for judge")

    try:
        return adapter_cls(**entry)
    except TypeError as exc:
        raise InvalidJudgeConfigError(
            f"judge config has invalid fields for adapter '{adapter_type}': {exc}"
        ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: PASS

- [ ] **Step 5: Add the judge entry to the real arms.yaml**

Append to `backend/arms.yaml`:

```yaml

judge:
  adapter: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY
  price_per_1m_input: 1.00
  price_per_1m_output: 5.00
```

- [ ] **Step 6: Commit**

```bash
git add app/config/arms.py arms.yaml tests/config/test_arms.py
git commit -m "feat: load the judge model from a separate judge: key in arms.yaml"
```

---

### Task 7: Chain judge scoring onto successful generation calls

**Files:**
- Modify: `backend/app/tasks/worker.py`
- Modify: `backend/tests/tasks/test_execute_call.py`
- Test: `backend/tests/tasks/test_execute_judge_call.py`
- Test: `backend/tests/tasks/test_persist_run_result.py`

**Interfaces:**
- Consumes: `load_judge_arm` (Task 6), `score_output`, `JudgeParseError`,
  `JudgeResult` (Task 4).
- Produces: `_persist_run_result(...) -> int | None` (now returns the new
  row's id, was `-> None`); `run_judge_call` (Celery task,
  `.delay(run_result_id: int)`); `execute_judge_call(*, run_result_id: int,
  celery_task_id: str | None = None, max_retries: int = 3,
  backoff_base_seconds: float = 1.0) -> None`.

- [ ] **Step 1: Write the failing tests for `_persist_run_result`'s new return value**

Add to `backend/tests/tasks/test_persist_run_result.py`, inside
`test_persists_success_response_field_by_field`, after the existing
assertions add:

```python
    result_id = asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name="fake-arm",
            repeat_index=2,
            celery_task_id="task-abc",
            status="completed",
            response=response,
        )
    )
    assert result_id == row.id
```

Note: this duplicates the earlier `_persist_run_result` call in that test —
replace the existing bare `asyncio.run(_persist_run_result(...))` call
(the one with no assignment) with `result_id = asyncio.run(...)` and add
the `assert result_id == row.id` line after fetching `rows`/`row`, rather
than calling it twice. The full updated test body:

```python
def test_persists_success_response_field_by_field(run_and_example):
    run_id, example_id = run_and_example
    response = ModelResponse(
        text="positive",
        latency_ms=123.5,
        prompt_tokens=17,
        completion_tokens=3,
        cost_estimate_usd=0.000123,
        finish_reason="stop",
    )

    result_id = asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name="fake-arm",
            repeat_index=2,
            celery_task_id="task-abc",
            status="completed",
            response=response,
        )
    )

    rows = _fetch_rows(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert result_id == row.id
    assert row.run_id == run_id
    assert row.example_id == example_id
    assert row.arm_name == "fake-arm"
    assert row.repeat_index == 2
    assert row.celery_task_id == "task-abc"
    assert row.status == "completed"
    assert row.output_text == "positive"
    assert row.latency_ms == pytest.approx(123.5)
    assert row.prompt_tokens == 17
    assert row.completion_tokens == 3
    assert row.cost_estimate_usd == pytest.approx(0.000123)
    assert row.error_message is None
    assert row.judge_score is None
    assert row.created_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/tasks/test_persist_run_result.py -v`
Expected: FAIL — `assert result_id == row.id` fails because `_persist_run_result`
currently returns `None`.

- [ ] **Step 3: Make `_persist_run_result` return the new row's id**

In `backend/app/tasks/worker.py`, change the signature and body:

```python
async def _persist_run_result(
    *,
    run_id: int,
    example_id: int,
    arm_name: str,
    repeat_index: int,
    celery_task_id: str | None,
    status: str,
    response: ModelResponse | None = None,
    error_message: str | None = None,
) -> int | None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            run_result = RunResult(
                run_id=run_id,
                example_id=example_id,
                arm_name=arm_name,
                repeat_index=repeat_index,
                celery_task_id=celery_task_id,
                status=status,
                output_text=response.text if response else None,
                latency_ms=response.latency_ms if response else None,
                prompt_tokens=response.prompt_tokens if response else None,
                completion_tokens=response.completion_tokens if response else None,
                cost_estimate_usd=response.cost_estimate_usd if response else None,
                error_message=error_message,
            )
            session.add(run_result)
            await session.commit()
            await session.refresh(run_result)
            return run_result.id
    finally:
        await worker_engine.dispose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/tasks/test_persist_run_result.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for judge chaining in `execute_call`**

In `backend/tests/tasks/test_execute_call.py`, change the import line to
add `MagicMock`:

```python
from unittest.mock import AsyncMock, MagicMock
```

Add two new tests:

```python
def test_enqueues_judge_call_after_successful_persist(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock(return_value=42)
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    judge_task_mock = MagicMock()
    monkeypatch.setattr(worker, "run_judge_call", judge_task_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    judge_task_mock.delay.assert_called_once_with(run_result_id=42)


def test_does_not_enqueue_judge_call_on_failure(monkeypatch):
    adapter = FakeAdapter([RuntimeError("boom")] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    monkeypatch.setattr(worker, "_persist_run_result", AsyncMock())
    judge_task_mock = MagicMock()
    monkeypatch.setattr(worker, "run_judge_call", judge_task_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0, max_retries=3)

    judge_task_mock.delay.assert_not_called()
```

Also add `monkeypatch.setattr(worker, "run_judge_call", MagicMock())` to the
three existing tests whose success path reaches the final
`status="completed"` persist call, so they don't attempt a real Celery
publish: `test_succeeds_on_first_try`, `test_retries_then_succeeds`,
`test_still_retries_429`.

Also add a case to the parametrized `test_is_retryable_classification`.
First add the import:

```python
from app.judge.scorer import JudgeParseError
```

Then add to the `@pytest.mark.parametrize` list:

```python
        (JudgeParseError("bad format"), False),
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v`
Expected: FAIL — `worker` has no attribute `run_judge_call` (monkeypatch
target doesn't exist yet), and the two new chaining tests fail similarly.

- [ ] **Step 7: Implement judge chaining, `execute_judge_call`, and the `run_judge_call` task**

In `backend/app/tasks/worker.py`, update the imports:

```python
from sqlmodel import select

from app.adapters.base import ModelResponse
from app.config.arms import load_arms, load_judge_arm
from app.db.models import EvalExample, RunResult
from app.db.session import DATABASE_URL
from app.judge.scorer import JudgeParseError, JudgeResult, score_output
```

Extend `is_retryable` — add this check before the final `return True`:

```python
    if isinstance(exc, JudgeParseError):
        return False
```

At the end of `execute_call`, replace the final block:

```python
    asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name=arm_name,
            repeat_index=repeat_index,
            celery_task_id=celery_task_id,
            status="completed",
            response=response,
        )
    )
```

with:

```python
    result_id = asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name=arm_name,
            repeat_index=repeat_index,
            celery_task_id=celery_task_id,
            status="completed",
            response=response,
        )
    )
    run_judge_call.delay(run_result_id=result_id)
```

Add the following after `_persist_run_result` and before `is_retryable`:

```python
async def _load_run_result_for_judging(run_result_id: int) -> tuple[str, str, str] | None:
    """Returns (input_text, gold_label, model_output), or None if the row is missing."""
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            result = await session.execute(
                select(EvalExample.text, EvalExample.gold_label, RunResult.output_text)
                .join(RunResult, RunResult.example_id == EvalExample.id)
                .where(RunResult.id == run_result_id)
            )
            row = result.first()
            return tuple(row) if row else None
    finally:
        await worker_engine.dispose()


async def _persist_judge_result(
    *,
    run_result_id: int,
    celery_task_id: str | None,
    status: str,
    score: int | None = None,
    rationale: str | None = None,
    error_message: str | None = None,
) -> None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            run_result = await session.get(RunResult, run_result_id)
            if run_result is None:
                logger.error("Cannot persist judge result: RunResult %s not found", run_result_id)
                return
            run_result.judge_status = status
            run_result.judge_score = score
            run_result.judge_rationale = rationale
            run_result.judge_error_message = error_message
            run_result.judge_celery_task_id = celery_task_id
            session.add(run_result)
            await session.commit()
    finally:
        await worker_engine.dispose()
```

Add the following after `execute_call` and its `@celery_app.task` wrapper,
at the end of the file:

```python
def execute_judge_call(
    *,
    run_result_id: int,
    celery_task_id: str | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
) -> None:
    def _persist_failure(error_message: str) -> None:
        try:
            asyncio.run(
                _persist_judge_result(
                    run_result_id=run_result_id,
                    celery_task_id=celery_task_id,
                    status="failed",
                    error_message=error_message,
                )
            )
        except Exception:
            logger.error(
                "Failed to persist failed judge result (run_result_id=%s): %s",
                run_result_id,
                error_message,
                exc_info=True,
            )

    try:
        loaded = asyncio.run(_load_run_result_for_judging(run_result_id))
    except Exception as exc:
        logger.error(
            "Could not load RunResult for judging (run_result_id=%s): %s", run_result_id, exc, exc_info=True
        )
        _persist_failure(f"Could not load RunResult: {exc!r}")
        return

    if loaded is None:
        logger.error("RunResult %s not found for judging", run_result_id)
        _persist_failure(f"RunResult {run_result_id} not found")
        return

    input_text, gold_label, model_output = loaded

    try:
        judge_adapter = load_judge_arm(str(ARMS_PATH))
    except Exception as exc:
        logger.error("Could not resolve judge arm (run_result_id=%s): %s", run_result_id, exc, exc_info=True)
        _persist_failure(f"Could not resolve judge arm: {exc!r}")
        return

    attempt = 0
    last_exc: Exception | None = None
    judge_result: JudgeResult | None = None
    while attempt <= max_retries:
        try:
            judge_result = score_output(judge_adapter, input_text, gold_label, model_output)
            break
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc):
                logger.warning("Non-retryable judge error (run_result_id=%s): %s", run_result_id, exc)
                break
            attempt += 1
            if attempt <= max_retries:
                logger.warning(
                    "Judge call failed, retrying (attempt %s/%s, run_result_id=%s): %s",
                    attempt,
                    max_retries,
                    run_result_id,
                    exc,
                )
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))

    if judge_result is None:
        logger.error("Judge call gave up (run_result_id=%s): %s", run_result_id, last_exc)
        _persist_failure(str(last_exc))
        return

    asyncio.run(
        _persist_judge_result(
            run_result_id=run_result_id,
            celery_task_id=celery_task_id,
            status="completed",
            score=judge_result.score,
            rationale=judge_result.rationale,
        )
    )


@celery_app.task(bind=True)
def run_judge_call(self, run_result_id: int) -> None:
    execute_judge_call(run_result_id=run_result_id, celery_task_id=self.request.id)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/tasks/test_execute_call.py -v`
Expected: PASS

- [ ] **Step 9: Write tests for `execute_judge_call` itself**

Create `backend/tests/tasks/test_execute_judge_call.py`:

```python
from unittest.mock import AsyncMock

import httpx
import pytest

from app.adapters.base import ModelResponse
from app.tasks import worker

GOOD_RESPONSE = ModelResponse(
    text="SCORE: 5\nRATIONALE: Correctly identifies positive sentiment.",
    latency_ms=10.0,
    prompt_tokens=40,
    completion_tokens=8,
)
MALFORMED_RESPONSE = ModelResponse(text="not a score", latency_ms=5.0, prompt_tokens=10, completion_tokens=2)


class FakeJudgeAdapter:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, prompt):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scores_and_persists_on_success(monkeypatch):
    adapter = FakeJudgeAdapter([GOOD_RESPONSE])
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(
        worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "The tone is positive."))
    )
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 1
    persist_mock.assert_awaited_once()
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
    assert kwargs["score"] == 5
    assert "positive sentiment" in kwargs["rationale"]


def test_run_result_not_found_persists_failure(monkeypatch):
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=None))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)

    worker.execute_judge_call(run_result_id=999)

    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "not found" in kwargs["error_message"]


def test_malformed_judge_response_does_not_retry(monkeypatch):
    adapter = FakeJudgeAdapter([MALFORMED_RESPONSE] * 4)
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "hmm")))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 1
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"


def test_retries_transient_judge_errors(monkeypatch):
    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(500, request=request)
    transient = httpx.HTTPStatusError("boom", request=request, response=response)

    adapter = FakeJudgeAdapter([transient, GOOD_RESPONSE])
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "ok")))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 2
    assert sleep_calls == [1.0]
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
```

- [ ] **Step 10: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/tasks/test_execute_judge_call.py -v`
Expected: PASS

- [ ] **Step 11: Run the full worker + config test suite**

Run: `cd backend && uv run pytest tests/tasks/ tests/config/ -v`
Expected: PASS (all worker, judge-call, and config tests green)

- [ ] **Step 12: Commit**

```bash
git add app/tasks/worker.py tests/tasks/test_execute_call.py tests/tasks/test_execute_judge_call.py tests/tasks/test_persist_run_result.py
git commit -m "feat: chain judge scoring onto successful generation calls"
```

---

### Task 8: `select_calibration_sample.py` script

**Files:**
- Create: `backend/scripts/select_calibration_sample.py`
- Test: `backend/tests/scripts/test_select_calibration_sample.py`

**Interfaces:**
- Consumes: `app.db.models.EvalExample`, `app.db.models.RunResult`;
  `app.db.session.engine` (module-level, monkeypatched by DB tests).
- Produces: `stratified_sample(rows: list[dict], n: int, seed: int | None =
  None) -> list[dict]` (pure, unit-testable without Postgres);
  `select_sample(run_id: int, n: int, seed: int | None = None) -> list[dict]`
  (async); used standalone via CLI, and by Task 9/10 only indirectly (they
  read its JSON output, not its code).

- [ ] **Step 1: Write the failing tests for the pure sampling function**

Create `backend/tests/scripts/test_select_calibration_sample.py`:

```python
from scripts.select_calibration_sample import stratified_sample


def _row(run_result_id, arm_name, gold_label):
    return {
        "run_result_id": run_result_id,
        "arm_name": arm_name,
        "gold_label": gold_label,
        "input_text": f"text {run_result_id}",
        "model_output": f"output {run_result_id}",
        "judge_score": 4,
        "judge_rationale": "ok",
        "human_score": None,
    }


def test_returns_everything_when_n_exceeds_available():
    rows = [_row(1, "arm-a", "positive"), _row(2, "arm-b", "negative")]
    sample = stratified_sample(rows, n=10)
    assert [r["run_result_id"] for r in sample] == [1, 2]


def test_samples_across_all_strata():
    rows = (
        [_row(i, "arm-a", "positive") for i in range(1, 11)]
        + [_row(i, "arm-a", "negative") for i in range(11, 21)]
        + [_row(i, "arm-b", "positive") for i in range(21, 31)]
        + [_row(i, "arm-b", "negative") for i in range(31, 41)]
    )
    sample = stratified_sample(rows, n=8, seed=42)

    assert len(sample) == 8
    strata = {(r["arm_name"], r["gold_label"]) for r in sample}
    assert strata == {("arm-a", "positive"), ("arm-a", "negative"), ("arm-b", "positive"), ("arm-b", "negative")}


def test_is_deterministic_given_a_seed():
    rows = [_row(i, "arm-a", "positive") for i in range(1, 21)]
    first = stratified_sample(rows, n=5, seed=7)
    second = stratified_sample(rows, n=5, seed=7)
    assert [r["run_result_id"] for r in first] == [r["run_result_id"] for r in second]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/scripts/test_select_calibration_sample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.select_calibration_sample'`

- [ ] **Step 3: Write the script**

Create `backend/scripts/select_calibration_sample.py`:

```python
"""Selects a stratified sample of judged RunResults for human calibration
labeling. Writes a JSON file — fill in each row's `human_score` by hand,
then run import_calibration_labels.py.

Run from inside backend/:
uv run python -m scripts.select_calibration_sample --run-id 1 --n 40 --out calibration_sample.json
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, RunResult
from app.db.session import engine


async def _fetch_judged_rows(run_id: int) -> list[dict]:
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(RunResult, EvalExample)
            .join(EvalExample, RunResult.example_id == EvalExample.id)
            .where(RunResult.run_id == run_id, RunResult.judge_status == "completed")
            .order_by(RunResult.id)
        )
        rows = []
        for run_result, example in result.all():
            rows.append(
                {
                    "run_result_id": run_result.id,
                    "arm_name": run_result.arm_name,
                    "input_text": example.text,
                    "gold_label": example.gold_label,
                    "model_output": run_result.output_text,
                    "judge_score": run_result.judge_score,
                    "judge_rationale": run_result.judge_rationale,
                    "human_score": None,
                }
            )
        return rows


def stratified_sample(rows: list[dict], n: int, seed: int | None = None) -> list[dict]:
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["run_result_id"])

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["arm_name"], row["gold_label"])
        buckets.setdefault(key, []).append(row)

    ordered_keys = sorted(buckets.keys())
    for key in ordered_keys:
        rng.shuffle(buckets[key])

    quota = max(1, n // len(ordered_keys))
    selected: list[dict] = []
    leftovers: list[dict] = []
    for key in ordered_keys:
        bucket = buckets[key]
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    return sorted(selected, key=lambda r: r["run_result_id"])


async def select_sample(run_id: int, n: int, seed: int | None = None) -> list[dict]:
    rows = await _fetch_judged_rows(run_id)
    return stratified_sample(rows, n, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sample = asyncio.run(select_sample(args.run_id, args.n, seed=args.seed))
    args.out.write_text(json.dumps(sample, indent=2))
    print(f"Wrote {len(sample)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/scripts/test_select_calibration_sample.py -v`
Expected: PASS

- [ ] **Step 5: Write and run a DB-backed test for the fetch query**

Add to the same test file:

```python
import asyncio

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from scripts.select_calibration_sample import _fetch_judged_rows
from tests.conftest import db_test_engine, postgres_reachable


@pytest.mark.skipif(not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)")
def test_fetch_judged_rows_only_includes_judge_completed(monkeypatch):
    monkeypatch.setattr("scripts.select_calibration_sample.engine", db_test_engine)

    async def _setup():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=2)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            judged = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
                judge_status="completed",
                judge_score=5,
                judge_rationale="Good.",
            )
            pending = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=1,
                output_text="positive",
                status="completed",
                judge_status="pending",
            )
            session.add(judged)
            session.add(pending)
            await session.commit()
            await session.refresh(judged)
            await session.refresh(pending)
            return run.id, example.id, judged.id, pending.id

    run_id, example_id, judged_id, pending_id = asyncio.run(_setup())

    try:
        rows = asyncio.run(_fetch_judged_rows(run_id))
        assert [r["run_result_id"] for r in rows] == [judged_id]
        assert rows[0]["gold_label"] == "positive"
        assert rows[0]["judge_score"] == 5
    finally:
        async def _teardown():
            async with AsyncSession(db_test_engine) as session:
                for rid in (judged_id, pending_id):
                    obj = await session.get(RunResult, rid)
                    if obj:
                        await session.delete(obj)
                run = await session.get(Run, run_id)
                if run:
                    await session.delete(run)
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
                await session.commit()

        asyncio.run(_teardown())
```

Run: `cd backend && uv run pytest tests/scripts/test_select_calibration_sample.py -v`
Expected: PASS (or skipped if Postgres isn't running).

- [ ] **Step 6: Commit**

```bash
git add scripts/select_calibration_sample.py tests/scripts/test_select_calibration_sample.py
git commit -m "feat: add select_calibration_sample.py for stratified calibration sampling"
```

---

### Task 9: `import_calibration_labels.py` script

**Files:**
- Create: `backend/scripts/import_calibration_labels.py`
- Test: `backend/tests/scripts/test_import_calibration_labels.py`

**Interfaces:**
- Consumes: `app.db.models.JudgeCalibrationLabel` (Task 2);
  `app.db.session.engine`.
- Produces: `CalibrationImportError(ValueError)`; `import_labels(rows:
  list[dict], labeled_by: str) -> int` (async, upsert count) — reads JSON
  written by Task 8's script.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/scripts/test_import_calibration_labels.py`:

```python
import asyncio

import pytest
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
from scripts.import_calibration_labels import CalibrationImportError, import_labels
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


@pytest.fixture
def run_result_id():
    async def _setup():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
                judge_score=5,
                judge_status="completed",
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)
            return result.id, run.id, example.id

    result_id, run_id, example_id = asyncio.run(_setup())
    yield result_id

    async def _teardown():
        async with AsyncSession(db_test_engine) as session:
            await session.execute(delete(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == result_id))
            rr = await session.get(RunResult, result_id)
            if rr:
                await session.delete(rr)
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            example = await session.get(EvalExample, example_id)
            if example:
                await session.delete(example)
            await session.commit()

    asyncio.run(_teardown())


def test_import_is_idempotent_upsert(monkeypatch, run_result_id):
    monkeypatch.setattr("scripts.import_calibration_labels.engine", db_test_engine)
    rows = [{"run_result_id": run_result_id, "human_score": 4, "notes": "seems right"}]

    first_count = asyncio.run(import_labels(rows, "you@example.com"))
    rows[0]["human_score"] = 5
    second_count = asyncio.run(import_labels(rows, "you@example.com"))

    assert first_count == 1
    assert second_count == 1

    async def _fetch():
        async with AsyncSession(db_test_engine) as session:
            result = await session.execute(
                select(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == run_result_id)
            )
            return result.scalars().all()

    labels = asyncio.run(_fetch())
    assert len(labels) == 1
    assert labels[0].human_score == 5


def test_rejects_missing_human_score(run_result_id):
    rows = [{"run_result_id": run_result_id, "human_score": None}]
    with pytest.raises(CalibrationImportError):
        asyncio.run(import_labels(rows, "you@example.com"))


def test_rejects_out_of_range_human_score(run_result_id):
    rows = [{"run_result_id": run_result_id, "human_score": 7}]
    with pytest.raises(CalibrationImportError):
        asyncio.run(import_labels(rows, "you@example.com"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/scripts/test_import_calibration_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.import_calibration_labels'`
(or skipped if Postgres isn't running).

- [ ] **Step 3: Write the script**

Create `backend/scripts/import_calibration_labels.py`:

```python
"""Imports hand-filled human_score values from a calibration sample JSON
file (produced by select_calibration_sample.py) into judge_calibration_label.
Idempotent: re-running upserts by run_result_id rather than duplicating.

Run from inside backend/:
uv run python -m scripts.import_calibration_labels --in calibration_sample.json --labeled-by you@example.com
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import JudgeCalibrationLabel
from app.db.session import engine


class CalibrationImportError(ValueError):
    pass


def _validate(rows: list[dict]) -> None:
    for row in rows:
        score = row.get("human_score")
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise CalibrationImportError(
                f"run_result_id {row.get('run_result_id')} has invalid human_score: {score!r}"
            )


async def import_labels(rows: list[dict], labeled_by: str) -> int:
    _validate(rows)
    upserted = 0
    async with AsyncSession(engine) as session:
        for row in rows:
            result = await session.execute(
                select(JudgeCalibrationLabel).where(
                    JudgeCalibrationLabel.run_result_id == row["run_result_id"]
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.human_score = row["human_score"]
                existing.labeled_by = labeled_by
                existing.notes = row.get("notes")
                session.add(existing)
            else:
                session.add(
                    JudgeCalibrationLabel(
                        run_result_id=row["run_result_id"],
                        human_score=row["human_score"],
                        labeled_by=labeled_by,
                        notes=row.get("notes"),
                    )
                )
            upserted += 1
        await session.commit()
    return upserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--labeled-by", required=True)
    args = parser.parse_args()

    rows = json.loads(args.in_path.read_text())
    count = asyncio.run(import_labels(rows, args.labeled_by))
    print(f"Upserted {count} calibration labels.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/scripts/test_import_calibration_labels.py -v`
Expected: PASS (or skipped if Postgres isn't running).

- [ ] **Step 5: Commit**

```bash
git add scripts/import_calibration_labels.py tests/scripts/test_import_calibration_labels.py
git commit -m "feat: add import_calibration_labels.py for idempotent human-label import"
```

---

### Task 10: `calibration_report.py` script

**Files:**
- Create: `backend/scripts/calibration_report.py`
- Test: `backend/tests/scripts/test_calibration_report.py`

**Interfaces:**
- Consumes: `calibration_report` (Task 5, `app.judge.calibration`);
  `app.db.models.JudgeCalibrationLabel`, `RunResult`; `app.db.session.engine`.
- Produces: `build_report(run_id: int) -> dict` (async) — the final
  deliverable of the calibration workflow.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/scripts/test_calibration_report.py`:

```python
import asyncio

import pytest
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
from scripts.calibration_report import build_report
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


def test_build_report_joins_judge_and_human_scores(monkeypatch):
    monkeypatch.setattr("scripts.calibration_report.engine", db_test_engine)

    async def _setup():
        async with AsyncSession(db_test_engine) as session:
            example = EvalExample(text="Profits rose.", gold_label="positive", source="test")
            session.add(example)
            run = Run(arm_names=["fake-arm"], sample_size=None, repeats=1, seed=None, total_calls=1)
            session.add(run)
            await session.commit()
            await session.refresh(example)
            await session.refresh(run)

            result = RunResult(
                run_id=run.id,
                example_id=example.id,
                arm_name="fake-arm",
                repeat_index=0,
                output_text="positive",
                status="completed",
                judge_status="completed",
                judge_score=4,
            )
            session.add(result)
            await session.commit()
            await session.refresh(result)

            label = JudgeCalibrationLabel(run_result_id=result.id, human_score=4, labeled_by="you@example.com")
            session.add(label)
            await session.commit()
            return run.id, example.id, result.id

    run_id, example_id, result_id = asyncio.run(_setup())

    try:
        report = asyncio.run(build_report(run_id))
        assert report["n"] == 1
        assert report["mean_abs_diff"] == pytest.approx(0.0)
    finally:
        async def _teardown():
            async with AsyncSession(db_test_engine) as session:
                await session.execute(
                    delete(JudgeCalibrationLabel).where(JudgeCalibrationLabel.run_result_id == result_id)
                )
                rr = await session.get(RunResult, result_id)
                if rr:
                    await session.delete(rr)
                run = await session.get(Run, run_id)
                if run:
                    await session.delete(run)
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
                await session.commit()

        asyncio.run(_teardown())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/scripts/test_calibration_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.calibration_report'`
(or skipped if Postgres isn't running).

- [ ] **Step 3: Write the script**

Create `backend/scripts/calibration_report.py`:

```python
"""Computes judge/human agreement for a run's calibration sample. Run this
and read the result before trusting judge_score on the rest of the run.

Run from inside backend/:
uv run python -m scripts.calibration_report --run-id 1
"""
import argparse
import asyncio

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import JudgeCalibrationLabel, RunResult
from app.db.session import engine
from app.judge.calibration import calibration_report


async def _fetch_pairs(run_id: int) -> list[tuple[float, int]]:
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(RunResult.judge_score, JudgeCalibrationLabel.human_score)
            .join(JudgeCalibrationLabel, JudgeCalibrationLabel.run_result_id == RunResult.id)
            .where(RunResult.run_id == run_id)
        )
        return [(judge_score, human_score) for judge_score, human_score in result.all()]


async def build_report(run_id: int) -> dict:
    pairs = await _fetch_pairs(run_id)
    return calibration_report(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    report = asyncio.run(build_report(args.run_id))
    print(f"n = {report['n']}")
    print(f"Spearman r = {report['spearman_r']:.3f} (p = {report['spearman_p']:.3f})")
    print(f"Cohen's kappa (score>=4 as correct) = {report['cohens_kappa']:.3f}")
    print(f"Mean absolute difference = {report['mean_abs_diff']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/scripts/test_calibration_report.py -v`
Expected: PASS (or skipped if Postgres isn't running).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration_report.py tests/scripts/test_calibration_report.py
git commit -m "feat: add calibration_report.py for judge/human agreement reporting"
```

---

### Task 11: Docs, full test run, mark Phase 3 done

**Files:**
- Modify: `backend/README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- None — documentation and final verification only.

- [ ] **Step 1: Document the judge + calibration workflow in the backend README**

Add a new `## Phase 3: Judge layer + calibration` section to
`backend/README.md`, after the existing `## Phase 2: Orchestration`
section (before `## Tests`):

```markdown
## Phase 3: Judge layer + calibration

Every successfully completed `RunResult` is automatically scored 1-5 by a
rubric-based LLM judge (configured via the separate `judge:` key in
`arms.yaml`, never as an eval arm). Judge scores land on `judge_score` /
`judge_rationale`; `judge_status` tracks `pending` / `completed` / `failed`
independently of the generation call's own `status`.

**Before trusting `judge_score` on a full run**, run the calibration
workflow — CLAUDE.md's differentiator is that judge calibration is
reported, not assumed:

### 1. Select a stratified sample to hand-label

```bash
uv run python -m scripts.select_calibration_sample --run-id 1 --n 40 --out calibration_sample.json
```

Stratifies by `(arm_name, gold_label)` so every arm and sentiment class is
represented. Open the file and fill in each row's `human_score` (1-5) by
hand.

### 2. Import your labels

```bash
uv run python -m scripts.import_calibration_labels --in calibration_sample.json --labeled-by you@example.com
```

Idempotent — re-running with updated scores upserts rather than duplicates.

### 3. Read the calibration report

```bash
uv run python -m scripts.calibration_report --run-id 1
```

Prints Spearman correlation and Cohen's kappa (score >= 4 treated as
"correct") between judge and human scores, plus mean absolute difference.
```

- [ ] **Step 2: Run the full backend test suite**

Run: `cd backend && uv run pytest -v`
Expected: PASS (Postgres-dependent tests pass if `docker compose up -d
postgres redis` is running from the repo root, otherwise they skip
cleanly — none should FAIL or ERROR).

- [ ] **Step 3: Mark Phase 3 done in CLAUDE.md**

In `CLAUDE.md`, change:

```markdown
3. **Judge layer + calibration** — implement rubric-based LLM-as-judge; score
   the gold subset; report agreement with human labels before proceeding.
   Spec: `docs/superpowers/specs/2026-08-27-judge-layer-calibration-design.md`.
```

to:

```markdown
3. **Judge layer + calibration** ✅ **Done.** Rubric-based LLM-as-judge
   (`backend/app/judge/`) auto-scores every completed `RunResult` via a
   chained Celery task. Calibration workflow (`backend/scripts/select_
   calibration_sample.py`, `import_calibration_labels.py`,
   `calibration_report.py`) reports Spearman correlation and Cohen's kappa
   between judge and human scores before judge scores are trusted on a
   full run. Spec: `docs/superpowers/specs/2026-08-27-judge-layer-calibration-design.md`.
```

- [ ] **Step 4: Commit**

```bash
git add backend/README.md CLAUDE.md
git commit -m "docs: document judge layer + calibration workflow, mark Phase 3 done"
```
