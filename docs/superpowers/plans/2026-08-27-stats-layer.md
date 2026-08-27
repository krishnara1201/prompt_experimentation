# Stats Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paired significance testing, a Bayesian equivalence test, multiple-comparison correction, and a sample-size/power calculator, exposed via a new `/runs/{run_id}/compare|equivalence|power` API router.

**Architecture:** A `backend/app/stats/` package with four pure-logic modules (`aggregation.py` fetches and groups `RunResult` rows by `(example_id, arm_name)`; `paired_tests.py` runs a hierarchical paired bootstrap + Wilcoxon + Holm-Bonferroni correction; `power.py` estimates required/achieved sample size; `bayesian.py` runs a PyMC paired-difference model), wired together by a new FastAPI router in `backend/app/api/routes/stats.py`.

**Tech Stack:** Python, FastAPI, SQLModel, scipy, numpy, PyMC (new dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-stats-layer-design.md`

## Global Constraints

- Minimum 5 eligible paired examples for any comparison (`MIN_PAIRED_EXAMPLES = 5`) — below that, raise `InsufficientDataError`, surfaced by the API as HTTP 422. No silent partial results.
- Paired bootstrap default `B = 10,000` replicates, overridable via `bootstrap_samples` query param.
- Multiple-comparison correction is Holm-Bonferroni (family-wise), not Benjamini-Hochberg — applied only when a `/compare` response contains more than one pair.
- `epsilon` (equivalence margin) has no default anywhere in the stack — the API rejects a request that omits it.
- Wilcoxon and the Bayesian model both operate on **repeat-averaged** per-example diffs; the hierarchical bootstrap is the only place repeat-level (within-arm) variance is propagated, per the spec's explicit simplification.
- `metric` is validated against an explicit allowlist (`ALLOWED_METRICS` in `aggregation.py`) before ever reaching `getattr(RunResult, metric)` — it is user-controlled API input.
- Only `pymc` is a new dependency; everything else (bootstrap, Wilcoxon, Holm, power formula) stays on `scipy`/`numpy`, already present.
- Follow existing repo conventions: dataclasses for pure-module results, Pydantic `BaseModel`s for API request/response shapes, `HTTPException` for API errors, `AsyncSession`/`get_session` for DB access, Postgres-`skipif` + `TestClient` + local helper functions for API-level tests (see `tests/api/test_runs.py`).

---

### Task 1: Aggregation — group `RunResult` rows by `(example_id, arm_name)`

**Files:**
- Create: `backend/app/stats/__init__.py`
- Create: `backend/app/stats/aggregation.py`
- Create: `backend/tests/stats/__init__.py`
- Test: `backend/tests/stats/test_aggregation.py`

**Interfaces:**
- Produces: `ALLOWED_METRICS: set[str]` = `{"judge_score", "latency_ms", "cost_estimate_usd", "prompt_tokens", "completion_tokens"}`. `async def load_metric_by_example(session: AsyncSession, run_id: int, metric: str, arm_names: list[str]) -> dict[tuple[int, str], list[float]]` — keys are `(example_id, arm_name)`, values are that cell's repeat values (unaveraged). Raises `ValueError` if `metric not in ALLOWED_METRICS`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/stats/test_aggregation.py
import asyncio

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.stats.aggregation import load_metric_by_example
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


def _insert_examples(n: int) -> list[int]:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            examples = [EvalExample(text=f"text {i}", gold_label="positive", source="test") for i in range(n)]
            session.add_all(examples)
            await session.commit()
            for example in examples:
                await session.refresh(example)
            return [e.id for e in examples]

    return asyncio.run(_run())


def _insert_run(arm_names: list[str]) -> int:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=arm_names, sample_size=None, repeats=1, seed=None, total_calls=0)
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    return asyncio.run(_run())


def _insert_result(
    run_id: int,
    example_id: int,
    arm_name: str,
    repeat_index: int,
    status: str = "completed",
    judge_status: str = "completed",
    judge_score: float | None = None,
    latency_ms: float | None = None,
) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    status=status,
                    judge_status=judge_status,
                    judge_score=judge_score,
                    latency_ms=latency_ms,
                    output_text="x" if status == "completed" else None,
                    error_message=None if status == "completed" else "boom",
                )
            )
            await session.commit()

    asyncio.run(_run())


def _cleanup(run_id: int, example_ids: list[int]) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            from sqlmodel import delete

            await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            for example_id in example_ids:
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def test_groups_completed_results_by_example_and_arm():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=100.0)
        _insert_result(run_id, example_ids[0], "arm-a", 1, latency_ms=120.0)
        _insert_result(run_id, example_ids[0], "arm-b", 0, latency_ms=200.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a", "arm-b"])

        result = asyncio.run(_query())
        assert result[(example_ids[0], "arm-a")] == [100.0, 120.0]
        assert result[(example_ids[0], "arm-b")] == [200.0]
    finally:
        _cleanup(run_id, example_ids)


def test_excludes_non_completed_status_rows():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, status="failed", latency_ms=999.0)
        _insert_result(run_id, example_ids[0], "arm-a", 1, status="completed", latency_ms=50.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        result = asyncio.run(_query())
        assert result[(example_ids[0], "arm-a")] == [50.0]
    finally:
        _cleanup(run_id, example_ids)


def test_judge_score_metric_requires_completed_judge_status():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(
            run_id, example_ids[0], "arm-a", 0,
            status="completed", judge_status="pending", judge_score=None, latency_ms=10.0,
        )
        _insert_result(
            run_id, example_ids[0], "arm-a", 1,
            status="completed", judge_status="completed", judge_score=4.0, latency_ms=20.0,
        )

        async def _query_judge():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "judge_score", ["arm-a"])

        async def _query_latency():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        judge_result = asyncio.run(_query_judge())
        latency_result = asyncio.run(_query_latency())
        assert judge_result[(example_ids[0], "arm-a")] == [4.0]
        assert latency_result[(example_ids[0], "arm-a")] == [10.0, 20.0]
    finally:
        _cleanup(run_id, example_ids)


def test_only_includes_requested_arms():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=1.0)
        _insert_result(run_id, example_ids[0], "arm-b", 0, latency_ms=2.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await load_metric_by_example(session, run_id, "latency_ms", ["arm-a"])

        result = asyncio.run(_query())
        assert list(result.keys()) == [(example_ids[0], "arm-a")]
    finally:
        _cleanup(run_id, example_ids)


def test_rejects_unknown_metric():
    async def _query():
        async with AsyncSession(db_test_engine) as session:
            return await load_metric_by_example(session, 1, "not_a_real_column", ["arm-a"])

    with pytest.raises(ValueError):
        asyncio.run(_query())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/stats/test_aggregation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stats'`

- [ ] **Step 3: Write the aggregation module**

```python
# backend/app/stats/__init__.py
```
(empty file)

```python
# backend/app/stats/aggregation.py
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import RunResult

ALLOWED_METRICS = {"judge_score", "latency_ms", "cost_estimate_usd", "prompt_tokens", "completion_tokens"}


async def load_metric_by_example(
    session: AsyncSession,
    run_id: int,
    metric: str,
    arm_names: list[str],
) -> dict[tuple[int, str], list[float]]:
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"unknown metric {metric!r}; must be one of {sorted(ALLOWED_METRICS)}")

    column = getattr(RunResult, metric)
    stmt = select(RunResult.example_id, RunResult.arm_name, column).where(
        RunResult.run_id == run_id,
        RunResult.arm_name.in_(arm_names),
        RunResult.status == "completed",
        column.is_not(None),
    )
    if metric == "judge_score":
        stmt = stmt.where(RunResult.judge_status == "completed")

    result = await session.execute(stmt)
    repeats_by_cell: dict[tuple[int, str], list[float]] = {}
    for example_id, arm_name, value in result.all():
        repeats_by_cell.setdefault((example_id, arm_name), []).append(float(value))
    return repeats_by_cell
```

```python
# backend/tests/stats/__init__.py
```
(empty file)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/stats/test_aggregation.py -v`
Expected: PASS (all 5 tests) — skipped instead if Postgres isn't running locally.

- [ ] **Step 5: Commit**

```bash
git add backend/app/stats/__init__.py backend/app/stats/aggregation.py backend/tests/stats/__init__.py backend/tests/stats/test_aggregation.py
git commit -m "feat: add RunResult aggregation for the stats layer"
```

---

### Task 2: Paired frequentist tests — hierarchical bootstrap, Wilcoxon, Holm-Bonferroni

**Files:**
- Create: `backend/app/stats/errors.py`
- Create: `backend/app/stats/paired_tests.py`
- Test: `backend/tests/stats/test_paired_tests.py`

**Interfaces:**
- Consumes: nothing from Task 1 (operates on the `dict[tuple[int, str], list[float]]` shape `load_metric_by_example` produces, but takes it as a plain argument).
- Produces: `MIN_PAIRED_EXAMPLES = 5`, `class InsufficientDataError(ValueError)` in `app/stats/errors.py`. `@dataclass class PairedComparisonResult` with fields `arm_a: str, arm_b: str, metric: str, n_examples: int, n_excluded: int, mean_diff: float, ci_lower: float, ci_upper: float, wilcoxon_statistic: float, p_value: float, p_value_corrected: float | None = None`. `def paired_diffs(repeats_by_cell, arm_a: str, arm_b: str) -> tuple[list[float], int]` — repeat-averaged per-example diffs for the eligible examples, and the excluded count. `def compare_pair(repeats_by_cell, arm_a: str, arm_b: str, metric: str, bootstrap_samples: int = 10_000, seed: int | None = None) -> PairedComparisonResult`, raises `InsufficientDataError`. `def correct_pairwise_pvalues(results: list[PairedComparisonResult]) -> list[PairedComparisonResult]` — mutates and returns the same list, setting `p_value_corrected` on each.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/stats/test_paired_tests.py
import pytest

from app.stats.errors import InsufficientDataError
from app.stats.paired_tests import PairedComparisonResult, compare_pair, correct_pairwise_pvalues, paired_diffs


def _cell(*values: float) -> list[float]:
    return list(values)


def test_paired_diffs_computes_repeat_averaged_diffs_for_shared_examples():
    repeats_by_cell = {
        (1, "a"): _cell(4.0, 6.0),  # mean 5.0
        (1, "b"): _cell(3.0),
        (2, "a"): _cell(2.0),
        (2, "b"): _cell(1.0),
        (3, "a"): _cell(9.0),  # arm b missing for example 3
    }
    diffs, n_excluded = paired_diffs(repeats_by_cell, "a", "b")
    assert diffs == [5.0 - 3.0, 2.0 - 1.0]
    assert n_excluded == 1


def test_compare_pair_raises_when_fewer_than_min_examples():
    repeats_by_cell = {
        (i, arm): _cell(float(i))
        for i in range(3)
        for arm in ("a", "b")
    }
    with pytest.raises(InsufficientDataError):
        compare_pair(repeats_by_cell, "a", "b", "latency_ms")


def test_compare_pair_computes_mean_diff_and_valid_ci():
    # arm "a" is always exactly 2.0 higher than arm "b" for every example.
    repeats_by_cell = {}
    for i in range(10):
        repeats_by_cell[(i, "a")] = _cell(float(i) + 2.0)
        repeats_by_cell[(i, "b")] = _cell(float(i))

    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=500, seed=42)
    assert isinstance(result, PairedComparisonResult)
    assert result.n_examples == 10
    assert result.n_excluded == 0
    assert result.mean_diff == pytest.approx(2.0)
    assert result.ci_lower <= result.mean_diff <= result.ci_upper
    assert result.p_value < 0.05


def test_compare_pair_excludes_examples_missing_from_one_arm():
    repeats_by_cell = {}
    for i in range(6):
        repeats_by_cell[(i, "a")] = _cell(1.0)
        if i != 5:
            repeats_by_cell[(i, "b")] = _cell(1.0)

    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=200, seed=1)
    assert result.n_examples == 5
    assert result.n_excluded == 1


def test_compare_pair_all_identical_diffs_returns_p_value_one_without_raising():
    repeats_by_cell = {(i, arm): _cell(1.0) for i in range(6) for arm in ("a", "b")}
    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=200, seed=1)
    assert result.p_value == 1.0


def test_compare_pair_reproducible_with_seed():
    repeats_by_cell = {}
    for i in range(8):
        repeats_by_cell[(i, "a")] = _cell(float(i) * 1.3, float(i) * 1.1)
        repeats_by_cell[(i, "b")] = _cell(float(i))

    first = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=300, seed=7)
    second = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=300, seed=7)
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper


def test_correct_pairwise_pvalues_applies_holm_bonferroni():
    results = [
        PairedComparisonResult("a", "b", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.01),
        PairedComparisonResult("a", "c", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.02),
        PairedComparisonResult("b", "c", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.03),
    ]
    corrected = correct_pairwise_pvalues(results)
    assert corrected is results
    assert [r.p_value_corrected for r in results] == pytest.approx([0.03, 0.04, 0.04])


def test_correct_pairwise_pvalues_is_noop_for_single_pair():
    results = [PairedComparisonResult("a", "b", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.03)]
    correct_pairwise_pvalues(results)
    assert results[0].p_value_corrected == 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/stats/test_paired_tests.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stats.errors'`

- [ ] **Step 3: Write the errors and paired_tests modules**

```python
# backend/app/stats/errors.py
MIN_PAIRED_EXAMPLES = 5


class InsufficientDataError(ValueError):
    """Raised when a metric/arm-pair has fewer than MIN_PAIRED_EXAMPLES eligible paired examples."""
```

```python
# backend/app/stats/paired_tests.py
from dataclasses import dataclass
import random

from scipy import stats as scipy_stats

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class PairedComparisonResult:
    arm_a: str
    arm_b: str
    metric: str
    n_examples: int
    n_excluded: int
    mean_diff: float
    ci_lower: float
    ci_upper: float
    wilcoxon_statistic: float
    p_value: float
    p_value_corrected: float | None = None


def _eligible_examples(
    repeats_by_cell: dict[tuple[int, str], list[float]], arm_a: str, arm_b: str
) -> tuple[list[int], int]:
    example_ids = sorted({example_id for (example_id, arm_name) in repeats_by_cell if arm_name in (arm_a, arm_b)})
    eligible = [
        example_id
        for example_id in example_ids
        if (example_id, arm_a) in repeats_by_cell and (example_id, arm_b) in repeats_by_cell
    ]
    return eligible, len(example_ids) - len(eligible)


def _repeat_mean(repeats_by_cell: dict[tuple[int, str], list[float]], example_id: int, arm: str) -> float:
    values = repeats_by_cell[(example_id, arm)]
    return sum(values) / len(values)


def paired_diffs(
    repeats_by_cell: dict[tuple[int, str], list[float]], arm_a: str, arm_b: str
) -> tuple[list[float], int]:
    eligible, n_excluded = _eligible_examples(repeats_by_cell, arm_a, arm_b)
    diffs = [
        _repeat_mean(repeats_by_cell, example_id, arm_a) - _repeat_mean(repeats_by_cell, example_id, arm_b)
        for example_id in eligible
    ]
    return diffs, n_excluded


def compare_pair(
    repeats_by_cell: dict[tuple[int, str], list[float]],
    arm_a: str,
    arm_b: str,
    metric: str,
    bootstrap_samples: int = 10_000,
    seed: int | None = None,
) -> PairedComparisonResult:
    eligible, n_excluded = _eligible_examples(repeats_by_cell, arm_a, arm_b)
    if len(eligible) < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(
            f"only {len(eligible)} paired examples for {arm_a!r} vs {arm_b!r} on {metric!r}; "
            f"need at least {MIN_PAIRED_EXAMPLES}"
        )

    diffs = [
        _repeat_mean(repeats_by_cell, example_id, arm_a) - _repeat_mean(repeats_by_cell, example_id, arm_b)
        for example_id in eligible
    ]
    n = len(eligible)
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    replicate_means = []
    for _ in range(bootstrap_samples):
        total = 0.0
        for _ in range(n):
            example_id = eligible[rng.randrange(n)]
            a_repeats = repeats_by_cell[(example_id, arm_a)]
            b_repeats = repeats_by_cell[(example_id, arm_b)]
            total += a_repeats[rng.randrange(len(a_repeats))] - b_repeats[rng.randrange(len(b_repeats))]
        replicate_means.append(total / n)
    replicate_means.sort()
    ci_lower = replicate_means[int(0.025 * bootstrap_samples)]
    ci_upper = replicate_means[min(int(0.975 * bootstrap_samples), bootstrap_samples - 1)]

    if all(d == 0.0 for d in diffs):
        # scipy.stats.wilcoxon raises when every paired difference is exactly
        # zero -- e.g. two arms that always agree. Report the (correct) null
        # result directly instead of catching the exception.
        wilcoxon_statistic, p_value = 0.0, 1.0
    else:
        wilcoxon_statistic, p_value = scipy_stats.wilcoxon(diffs)

    return PairedComparisonResult(
        arm_a=arm_a,
        arm_b=arm_b,
        metric=metric,
        n_examples=n,
        n_excluded=n_excluded,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        wilcoxon_statistic=float(wilcoxon_statistic),
        p_value=float(p_value),
    )


def correct_pairwise_pvalues(results: list[PairedComparisonResult]) -> list[PairedComparisonResult]:
    m = len(results)
    if m <= 1:
        for r in results:
            r.p_value_corrected = r.p_value
        return results

    order = sorted(range(m), key=lambda i: results[i].p_value)
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * results[idx].p_value
        running_max = max(running_max, adjusted)
        corrected[idx] = min(running_max, 1.0)
    for i, r in enumerate(results):
        r.p_value_corrected = corrected[i]
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/stats/test_paired_tests.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/stats/errors.py backend/app/stats/paired_tests.py backend/tests/stats/test_paired_tests.py
git commit -m "feat: add hierarchical paired bootstrap, Wilcoxon, and Holm-Bonferroni correction"
```

---

### Task 3: Sample-size / power calculator

**Files:**
- Create: `backend/app/stats/power.py`
- Test: `backend/tests/stats/test_power.py`

**Interfaces:**
- Consumes: `MIN_PAIRED_EXAMPLES`, `InsufficientDataError` from `app.stats.errors` (Task 2).
- Produces: `@dataclass class PowerResult` with fields `pilot_n: int, pilot_mean_diff: float, pilot_std_diff: float, effect_size: float, alpha: float, target_power: float, required_n: int, achieved_power: float`. `def estimate_sample_size(pilot_diffs: list[float], effect_size: float | None = None, power: float = 0.8, alpha: float = 0.05) -> PowerResult`, raises `InsufficientDataError` or `ValueError` (zero effect size).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/stats/test_power.py
import math
import statistics

import pytest
from scipy.stats import norm

from app.stats.errors import InsufficientDataError
from app.stats.power import PowerResult, estimate_sample_size


def test_estimate_sample_size_raises_when_fewer_than_min_pilot_examples():
    with pytest.raises(InsufficientDataError):
        estimate_sample_size([1.0, 2.0, 3.0])


def test_estimate_sample_size_matches_closed_form_formula():
    pilot_diffs = [-1.0, 0.0, 1.0, 2.0, 3.0]
    result = estimate_sample_size(pilot_diffs, effect_size=1.0, power=0.8, alpha=0.05)

    mean_diff = statistics.mean(pilot_diffs)
    std_diff = statistics.stdev(pilot_diffs)
    z_alpha = norm.ppf(1 - 0.05 / 2)
    z_power = norm.ppf(0.8)
    expected_required_n = math.ceil(((z_alpha + z_power) ** 2 * std_diff**2) / 1.0**2)

    assert isinstance(result, PowerResult)
    assert result.pilot_n == 5
    assert result.pilot_mean_diff == pytest.approx(mean_diff)
    assert result.pilot_std_diff == pytest.approx(std_diff)
    assert result.required_n == expected_required_n
    assert 0.0 <= result.achieved_power <= 1.0


def test_estimate_sample_size_defaults_effect_size_to_pilot_mean_diff():
    pilot_diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = estimate_sample_size(pilot_diffs)
    assert result.effect_size == pytest.approx(statistics.mean(pilot_diffs))


def test_estimate_sample_size_raises_on_zero_effect_size():
    pilot_diffs = [-2.0, -1.0, 0.0, 1.0, 2.0]  # mean 0.0
    with pytest.raises(ValueError):
        estimate_sample_size(pilot_diffs)


def test_estimate_sample_size_smaller_effect_requires_larger_n():
    pilot_diffs = [-1.0, 0.0, 1.0, 2.0, 3.0]
    small_effect = estimate_sample_size(pilot_diffs, effect_size=0.5)
    large_effect = estimate_sample_size(pilot_diffs, effect_size=2.0)
    assert small_effect.required_n > large_effect.required_n
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/stats/test_power.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stats.power'`

- [ ] **Step 3: Write the power module**

```python
# backend/app/stats/power.py
from dataclasses import dataclass
import math

from scipy.stats import norm

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class PowerResult:
    pilot_n: int
    pilot_mean_diff: float
    pilot_std_diff: float
    effect_size: float
    alpha: float
    target_power: float
    required_n: int
    achieved_power: float


def estimate_sample_size(
    pilot_diffs: list[float],
    effect_size: float | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> PowerResult:
    n = len(pilot_diffs)
    if n < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(f"only {n} pilot paired examples; need at least {MIN_PAIRED_EXAMPLES}")

    mean_diff = sum(pilot_diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in pilot_diffs) / (n - 1)
    std_diff = math.sqrt(variance) or 1e-6

    delta = effect_size if effect_size is not None else mean_diff
    if delta == 0:
        raise ValueError("effect_size cannot be zero -- no detectable difference to power for")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    required_n = math.ceil(((z_alpha + z_power) ** 2 * std_diff**2) / delta**2)

    z_achieved = abs(delta) * math.sqrt(n) / std_diff - z_alpha
    achieved_power = float(norm.cdf(z_achieved))

    return PowerResult(
        pilot_n=n,
        pilot_mean_diff=mean_diff,
        pilot_std_diff=std_diff,
        effect_size=delta,
        alpha=alpha,
        target_power=power,
        required_n=required_n,
        achieved_power=achieved_power,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/stats/test_power.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/stats/power.py backend/tests/stats/test_power.py
git commit -m "feat: add sample-size/power calculator"
```

---

### Task 4: Bayesian equivalence test (PyMC)

**Files:**
- Modify: `backend/pyproject.toml` (add `pymc` dependency)
- Create: `backend/app/stats/bayesian.py`
- Test: `backend/tests/stats/test_bayesian.py`

**Interfaces:**
- Consumes: `MIN_PAIRED_EXAMPLES`, `InsufficientDataError` from `app.stats.errors` (Task 2).
- Produces: `@dataclass class EquivalenceResult` with fields `epsilon: float, posterior_mean: float, ci_lower: float, ci_upper: float, p_equivalent: float`. `def equivalence_probability(diffs: list[float], epsilon: float, draws: int = 2000, tune: int = 1000, chains: int = 2, cores: int = 1, random_seed: int | None = None) -> EquivalenceResult`, raises `InsufficientDataError`.

- [ ] **Step 1: Add the pymc dependency**

```bash
cd backend && uv add pymc
```

- [ ] **Step 2: Verify it installed**

Run: `cd backend && uv run python -c "import pymc; print(pymc.__version__)"`
Expected: prints a version string, no import error.

- [ ] **Step 3: Write the failing tests**

```python
# backend/tests/stats/test_bayesian.py
import pytest

from app.stats.bayesian import EquivalenceResult, equivalence_probability
from app.stats.errors import InsufficientDataError

SAMPLE_KWARGS = dict(draws=200, tune=200, chains=2, cores=1, random_seed=0)


def test_equivalence_probability_raises_when_fewer_than_min_examples():
    with pytest.raises(InsufficientDataError):
        equivalence_probability([0.1, 0.2, 0.3], epsilon=0.5, **SAMPLE_KWARGS)


def test_equivalence_probability_near_one_when_arms_effectively_identical():
    diffs = [0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.02, -0.02, 0.0]
    result = equivalence_probability(diffs, epsilon=0.5, **SAMPLE_KWARGS)
    assert isinstance(result, EquivalenceResult)
    assert result.p_equivalent > 0.9


def test_equivalence_probability_near_zero_when_local_much_worse():
    diffs = [-5.0, -4.8, -5.2, -4.9, -5.1, -5.0, -4.7, -5.3, -5.0, -4.9]
    result = equivalence_probability(diffs, epsilon=0.5, **SAMPLE_KWARGS)
    assert result.p_equivalent < 0.1


def test_equivalence_probability_result_fields_are_internally_consistent():
    diffs = [-1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.3, -0.4, 0.6, -0.1]
    result = equivalence_probability(diffs, epsilon=1.0, **SAMPLE_KWARGS)
    assert result.epsilon == 1.0
    assert result.ci_lower <= result.posterior_mean <= result.ci_upper
    assert 0.0 <= result.p_equivalent <= 1.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/stats/test_bayesian.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stats.bayesian'`

- [ ] **Step 5: Write the bayesian module**

```python
# backend/app/stats/bayesian.py
from dataclasses import dataclass

import numpy as np
import pymc as pm

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class EquivalenceResult:
    epsilon: float
    posterior_mean: float
    ci_lower: float
    ci_upper: float
    p_equivalent: float


def equivalence_probability(
    diffs: list[float],
    epsilon: float,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 1,
    random_seed: int | None = None,
) -> EquivalenceResult:
    if len(diffs) < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(f"only {len(diffs)} paired examples; need at least {MIN_PAIRED_EXAMPLES}")

    diffs_arr = np.asarray(diffs, dtype=float)
    # Weakly-informative priors scaled off the data's own spread, not a
    # hardcoded per-metric range -- this is what keeps the function generic
    # across bounded (judge_score) and unbounded (latency_ms, cost) metrics.
    # The floor guards the degenerate all-identical-diffs case.
    scale = max(float(diffs_arr.std()), 1e-6)

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=10 * scale)
        sigma = pm.HalfNormal("sigma", sigma=10 * scale)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=diffs_arr)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=random_seed, progressbar=False)

    mu_draws = idata.posterior["mu"].values.reshape(-1)
    return EquivalenceResult(
        epsilon=epsilon,
        posterior_mean=float(mu_draws.mean()),
        ci_lower=float(np.percentile(mu_draws, 2.5)),
        ci_upper=float(np.percentile(mu_draws, 97.5)),
        p_equivalent=float(np.mean(mu_draws >= -epsilon)),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/stats/test_bayesian.py -v`
Expected: PASS (4 tests). This runs actual MCMC sampling, so expect several seconds of runtime rather than instant.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/stats/bayesian.py backend/tests/stats/test_bayesian.py
git commit -m "feat: add PyMC-based Bayesian equivalence test"
```

---

### Task 5: `/runs/{run_id}/compare|equivalence|power` API router

**Files:**
- Create: `backend/app/api/routes/stats.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/api/test_stats.py`

**Interfaces:**
- Consumes: `ALLOWED_METRICS`, `load_metric_by_example` (Task 1); `paired_diffs`, `compare_pair`, `correct_pairwise_pvalues` (Task 2); `estimate_sample_size` (Task 3); `equivalence_probability` (Task 4); `InsufficientDataError` (Task 2/`app.stats.errors`); `Run`, `get_session` (existing).
- Produces: FastAPI `router` mounted at `/runs`, adding `GET /runs/{run_id}/compare`, `GET /runs/{run_id}/equivalence`, `GET /runs/{run_id}/power`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/api/test_stats.py
import asyncio
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.main import app
from tests.conftest import db_test_engine, postgres_reachable

pytestmark = pytest.mark.skipif(
    not postgres_reachable(), reason="Postgres not running (see docker-compose.yml)"
)


async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(db_test_engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _use_test_engine_session():
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.pop(get_session, None)


def _insert_examples(n: int) -> list[int]:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            examples = [EvalExample(text=f"text {i}", gold_label="positive", source="test") for i in range(n)]
            session.add_all(examples)
            await session.commit()
            for example in examples:
                await session.refresh(example)
            return [e.id for e in examples]

    return asyncio.run(_run())


def _insert_run(arm_names: list[str]) -> int:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            run = Run(arm_names=arm_names, sample_size=None, repeats=1, seed=None, total_calls=0)
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    return asyncio.run(_run())


def _insert_result(run_id: int, example_id: int, arm_name: str, latency_ms: float) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=0,
                    status="completed",
                    judge_status="completed",
                    latency_ms=latency_ms,
                    output_text="x",
                )
            )
            await session.commit()

    asyncio.run(_run())


def _cleanup(run_id: int, example_ids: list[int]) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            await session.execute(delete(RunResult).where(RunResult.run_id == run_id))
            run = await session.get(Run, run_id)
            if run:
                await session.delete(run)
            for example_id in example_ids:
                example = await session.get(EvalExample, example_id)
                if example:
                    await session.delete(example)
            await session.commit()

    asyncio.run(_run())


def _seed_two_arm_run(n_examples: int = 6, offset: float = 2.0) -> tuple[int, list[int]]:
    example_ids = _insert_examples(n_examples)
    run_id = _insert_run(["arm-a", "arm-b"])
    for i, example_id in enumerate(example_ids):
        _insert_result(run_id, example_id, "arm-a", latency_ms=float(i) + offset)
        _insert_result(run_id, example_id, "arm-b", latency_ms=float(i))
    return run_id, example_ids


def test_compare_arms_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999/compare?metric=latency_ms")
    assert response.status_code == 404


def test_compare_arms_422_for_unknown_metric():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=not_a_metric")
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_400_for_unknown_arm():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=not-an-arm"
        )
        assert response.status_code == 400
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_returns_paired_result_for_explicit_pair():
    run_id, example_ids = _seed_two_arm_run()
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=arm-b&bootstrap_samples=200"
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["n_examples"] == 6
        assert body[0]["mean_diff"] == pytest.approx(2.0)
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_all_pairs_with_holm_correction_for_three_arms():
    example_ids = _insert_examples(6)
    run_id = _insert_run(["arm-a", "arm-b", "arm-c"])
    try:
        for i, example_id in enumerate(example_ids):
            _insert_result(run_id, example_id, "arm-a", latency_ms=float(i) + 3.0)
            _insert_result(run_id, example_id, "arm-b", latency_ms=float(i))
            _insert_result(run_id, example_id, "arm-c", latency_ms=float(i) + 1.0)

        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=latency_ms&bootstrap_samples=200")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3
        for row in body:
            assert row["p_value_corrected"] >= row["p_value"]
    finally:
        _cleanup(run_id, example_ids)


def test_compare_arms_422_when_insufficient_paired_examples():
    run_id, example_ids = _seed_two_arm_run(n_examples=3)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/compare?metric=latency_ms&arm_a=arm-a&arm_b=arm-b")
        assert response.status_code == 422
    finally:
        _cleanup(run_id, example_ids)


def test_equivalence_returns_probability_between_zero_and_one():
    run_id, example_ids = _seed_two_arm_run(n_examples=10, offset=0.1)
    try:
        response = TestClient(app).get(
            f"/runs/{run_id}/equivalence?metric=latency_ms&arm_local=arm-a&arm_api=arm-b&epsilon=1.0"
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["p_equivalent"] <= 1.0
        assert body["ci_lower"] <= body["posterior_mean"] <= body["ci_upper"]
    finally:
        _cleanup(run_id, example_ids)


def test_power_returns_required_n_and_achieved_power():
    run_id, example_ids = _seed_two_arm_run(n_examples=10, offset=2.0)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/power?metric=latency_ms&arm_a=arm-a&arm_b=arm-b")
        assert response.status_code == 200
        body = response.json()
        assert body["required_n"] > 0
        assert 0.0 <= body["achieved_power"] <= 1.0
    finally:
        _cleanup(run_id, example_ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.routes.stats'`

- [ ] **Step 3: Write the stats router**

```python
# backend/app/api/routes/stats.py
from itertools import combinations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import Run
from app.db.session import get_session
from app.stats.aggregation import ALLOWED_METRICS, load_metric_by_example
from app.stats.bayesian import equivalence_probability
from app.stats.errors import InsufficientDataError
from app.stats.paired_tests import compare_pair, correct_pairwise_pvalues, paired_diffs
from app.stats.power import estimate_sample_size

router = APIRouter(prefix="/runs", tags=["stats"])


class PairedComparisonResponse(BaseModel):
    arm_a: str
    arm_b: str
    metric: str
    n_examples: int
    n_excluded: int
    mean_diff: float
    ci_lower: float
    ci_upper: float
    wilcoxon_statistic: float
    p_value: float
    p_value_corrected: float | None


class EquivalenceResponse(BaseModel):
    arm_local: str
    arm_api: str
    metric: str
    epsilon: float
    n_examples: int
    posterior_mean: float
    ci_lower: float
    ci_upper: float
    p_equivalent: float


class PowerResponse(BaseModel):
    arm_a: str
    arm_b: str
    metric: str
    pilot_n: int
    pilot_mean_diff: float
    pilot_std_diff: float
    effect_size: float
    alpha: float
    target_power: float
    required_n: int
    achieved_power: float


async def _load_run_or_404(run_id: int, session: AsyncSession) -> Run:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _validate_metric(metric: str) -> None:
    if metric not in ALLOWED_METRICS:
        raise HTTPException(
            status_code=422, detail=f"Unknown metric {metric!r}; must be one of {sorted(ALLOWED_METRICS)}"
        )


def _validate_arms(run: Run, *arm_names: str) -> None:
    for name in arm_names:
        if name not in run.arm_names:
            raise HTTPException(status_code=400, detail=f"Unknown arm for this run: {name}")


def _resolve_pairs(run: Run, arm_a: str | None, arm_b: str | None) -> list[tuple[str, str]]:
    if (arm_a is None) != (arm_b is None):
        raise HTTPException(status_code=422, detail="arm_a and arm_b must be supplied together, or both omitted")
    if arm_a is not None:
        _validate_arms(run, arm_a, arm_b)
        return [(arm_a, arm_b)]
    return list(combinations(run.arm_names, 2))


@router.get("/{run_id}/compare", response_model=list[PairedComparisonResponse])
async def compare_arms(
    run_id: int,
    metric: str,
    arm_a: str | None = None,
    arm_b: str | None = None,
    bootstrap_samples: int = Query(default=10_000, gt=0),
    session: AsyncSession = Depends(get_session),
):
    _validate_metric(metric)
    run = await _load_run_or_404(run_id, session)
    pairs = _resolve_pairs(run, arm_a, arm_b)

    repeats_by_cell = await load_metric_by_example(session, run_id, metric, run.arm_names)

    try:
        results = [
            compare_pair(repeats_by_cell, a, b, metric, bootstrap_samples=bootstrap_samples) for a, b in pairs
        ]
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    correct_pairwise_pvalues(results)
    return [PairedComparisonResponse(**vars(r)) for r in results]


@router.get("/{run_id}/equivalence", response_model=EquivalenceResponse)
async def equivalence(
    run_id: int,
    metric: str,
    arm_local: str,
    arm_api: str,
    epsilon: float,
    session: AsyncSession = Depends(get_session),
):
    _validate_metric(metric)
    run = await _load_run_or_404(run_id, session)
    _validate_arms(run, arm_local, arm_api)

    repeats_by_cell = await load_metric_by_example(session, run_id, metric, [arm_local, arm_api])
    diffs, _n_excluded = paired_diffs(repeats_by_cell, arm_local, arm_api)

    try:
        result = equivalence_probability(diffs, epsilon)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return EquivalenceResponse(
        arm_local=arm_local,
        arm_api=arm_api,
        metric=metric,
        epsilon=epsilon,
        n_examples=len(diffs),
        posterior_mean=result.posterior_mean,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        p_equivalent=result.p_equivalent,
    )


@router.get("/{run_id}/power", response_model=PowerResponse)
async def power_estimate(
    run_id: int,
    metric: str,
    arm_a: str,
    arm_b: str,
    power: float = Query(default=0.8, gt=0, lt=1),
    alpha: float = Query(default=0.05, gt=0, lt=1),
    effect_size: float | None = None,
    session: AsyncSession = Depends(get_session),
):
    _validate_metric(metric)
    run = await _load_run_or_404(run_id, session)
    _validate_arms(run, arm_a, arm_b)

    repeats_by_cell = await load_metric_by_example(session, run_id, metric, [arm_a, arm_b])
    diffs, _n_excluded = paired_diffs(repeats_by_cell, arm_a, arm_b)

    try:
        result = estimate_sample_size(diffs, effect_size=effect_size, power=power, alpha=alpha)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PowerResponse(
        arm_a=arm_a,
        arm_b=arm_b,
        metric=metric,
        pilot_n=result.pilot_n,
        pilot_mean_diff=result.pilot_mean_diff,
        pilot_std_diff=result.pilot_std_diff,
        effect_size=result.effect_size,
        alpha=result.alpha,
        target_power=result.target_power,
        required_n=result.required_n,
        achieved_power=result.achieved_power,
    )
```

```python
# backend/app/main.py
from fastapi import FastAPI

from app.api.routes.runs import router as runs_router
from app.api.routes.stats import router as stats_router
from app.db.session import lifespan

app = FastAPI(title="Prompt Experimentation API", lifespan=lifespan)
app.include_router(runs_router)
app.include_router(stats_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_stats.py -v`
Expected: PASS (8 tests) — skipped instead if Postgres isn't running locally. The equivalence test runs real MCMC sampling, so expect several seconds of runtime.

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && uv run pytest -v`
Expected: PASS (all tests, including the pre-existing suite — nothing in this plan modifies existing modules other than `main.py`'s router registration).

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routes/stats.py backend/app/main.py backend/tests/api/test_stats.py
git commit -m "feat: expose stats layer via /runs/{run_id}/compare|equivalence|power"
```

---

## Post-plan follow-up (not part of this plan)

Update `CLAUDE.md`'s "Build phases" section to mark Phase 4 done, mirroring how Phases 1-3 were marked, once all tasks above are complete and verified.
