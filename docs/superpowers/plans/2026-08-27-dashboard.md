# Phase 5 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 5 dashboard — win-rate table with CIs, cost/latency/quality frontier scatter, and judge calibration report — as a new read-only React frontend backed by three new FastAPI endpoints.

**Architecture:** Three new read-only backend endpoints (`GET /runs`, `GET /runs/{run_id}/summary`, `GET /runs/{run_id}/calibration`) added to the existing `runs.py`/`stats.py` route files and `aggregation.py` module, following their existing patterns exactly. A new `frontend/` Vite + React + TypeScript app consumes those plus the existing `GET /runs/{run_id}/compare` endpoint, with a run list page and a per-run tabbed dashboard page.

**Tech Stack:** Backend: FastAPI, SQLModel, pytest (existing). Frontend (new): Vite, React 19, TypeScript, Tailwind CSS v4, `@tanstack/react-query` v5, `react-router-dom` v7, Recharts v3.

**Spec:** `docs/superpowers/specs/2026-08-27-dashboard-design.md`

## Global Constraints

- Read-only dashboard — no run creation/editing from the UI (spec Non-goals).
- No auth (spec Non-goals) — matches the existing unauthenticated FastAPI app.
- No `equivalence`/`power` views in this phase (spec Non-goals).
- No dedicated frontend automated test suite — verify manually in-browser (spec Non-goals, Testing plan).
- Backend tests use the existing pattern exactly: `fastapi.testclient.TestClient`, `app.dependency_overrides[get_session]` pointed at the `NullPool` `db_test_engine` from `tests/conftest.py`, manual insert/cleanup helpers, `pytestmark = pytest.mark.skipif(not postgres_reachable(), ...)`.
- Frontend calls same-origin `/api/...` paths; a Vite dev-server proxy forwards `/api/*` to `http://localhost:8000/*`. No CORS middleware is added to the backend (spec: "Dev proxy").
- Pin major versions: `tailwindcss@^4`, `@tailwindcss/vite@^4`, `@tanstack/react-query@^5`, `react-router-dom@^7`, `recharts@^3`.

---

## Task 1: Extract run-status helper and add `GET /runs` (list)

**Files:**
- Modify: `backend/app/api/routes/runs.py`
- Test: `backend/tests/api/test_runs.py`

**Interfaces:**
- Produces: `RunSummary` (Pydantic model: `run_id: int, created_at: datetime, arm_names: list[str], status: str, total_calls: int, completed: int, failed: int, pending: int`), `GET /runs` returning `list[RunSummary]`, ordered newest-first by `created_at`.
- Produces (internal helper, used by Task 1 only in this plan but available to future callers): `_status_from_counts(total_calls: int, completed: int, failed: int) -> tuple[str, int]` returning `(status, pending)`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/api/test_runs.py`, after the existing `test_get_run_status_404_for_missing_run` test:

```python
def test_list_runs_includes_status_and_counts():
    example_id = _insert_example()
    run_id = _insert_run(total_calls=3)
    try:
        _insert_results(run_id, example_id, ["completed", "failed"])

        response = TestClient(app).get("/runs")
        assert response.status_code == 200
        rows = {row["run_id"]: row for row in response.json()}
        assert run_id in rows
        row = rows[run_id]
        assert row["status"] == "running"
        assert row["total_calls"] == 3
        assert row["completed"] == 1
        assert row["failed"] == 1
        assert row["pending"] == 1
        assert row["arm_names"] == ["fake-arm"]
    finally:
        _delete_run(run_id)
        _delete_example(example_id)


def test_list_runs_ordered_newest_first():
    run_id_a = _insert_run(total_calls=1)
    run_id_b = _insert_run(total_calls=1)
    try:
        rows = TestClient(app).get("/runs").json()
        ids = [row["run_id"] for row in rows]
        assert ids.index(run_id_b) < ids.index(run_id_a)
    finally:
        _delete_run(run_id_a)
        _delete_run(run_id_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_runs.py -k test_list_runs -v`
Expected: FAIL with `404 Not Found` for `GET /runs` (route doesn't exist yet) — the two new tests fail, all others still pass.

- [ ] **Step 3: Extract the status helper and wire up `get_run_status` to use it**

In `backend/app/api/routes/runs.py`, add the helper near the top (after the `RunStatusResponse` class):

```python
def _status_from_counts(total_calls: int, completed: int, failed: int) -> tuple[str, int]:
    done = completed + failed
    pending = total_calls - done
    if done == 0:
        status = "pending"
    elif done < total_calls:
        status = "running"
    elif failed == 0:
        status = "completed"
    else:
        status = "completed_with_errors"
    return status, pending
```

Replace the body of `get_run_status` (the `if done == 0: ... status = "completed_with_errors"` block) to call it:

```python
@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    counts_result = await session.execute(
        select(RunResult.status, func.count()).where(RunResult.run_id == run_id).group_by(RunResult.status)
    )
    counts = dict(counts_result.all())
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    status, pending = _status_from_counts(run.total_calls, completed, failed)

    return RunStatusResponse(
        run_id=run.id,
        status=status,
        total_calls=run.total_calls,
        completed=completed,
        failed=failed,
        pending=pending,
    )
```

- [ ] **Step 4: Add the `RunSummary` model and `GET /runs` route**

Add `from datetime import datetime` to the imports at the top of `runs.py`. Add the model near `RunStatusResponse`:

```python
class RunSummary(BaseModel):
    run_id: int
    created_at: datetime
    arm_names: list[str]
    status: str
    total_calls: int
    completed: int
    failed: int
    pending: int
```

Add the route, placed before `create_run` (so `GET ""` and `POST ""` sit together at the top of the file):

```python
@router.get("", response_model=list[RunSummary])
async def list_runs(session: AsyncSession = Depends(get_session)) -> list[RunSummary]:
    runs_result = await session.execute(select(Run).order_by(Run.created_at.desc()))
    runs = runs_result.scalars().all()

    counts_result = await session.execute(
        select(RunResult.run_id, RunResult.status, func.count()).group_by(RunResult.run_id, RunResult.status)
    )
    counts_by_run: dict[int, dict[str, int]] = {}
    for run_id, status, count in counts_result.all():
        counts_by_run.setdefault(run_id, {})[status] = count

    summaries = []
    for run in runs:
        counts = counts_by_run.get(run.id, {})
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        status, pending = _status_from_counts(run.total_calls, completed, failed)
        summaries.append(
            RunSummary(
                run_id=run.id,
                created_at=run.created_at,
                arm_names=run.arm_names,
                status=status,
                total_calls=run.total_calls,
                completed=completed,
                failed=failed,
                pending=pending,
            )
        )
    return summaries
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v`
Expected: PASS — all tests in the file, including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routes/runs.py backend/tests/api/test_runs.py
git commit -m "feat: add GET /runs list endpoint"
```

---

## Task 2: Add `summarize_arms` aggregation and `GET /runs/{run_id}/summary`

**Files:**
- Modify: `backend/app/stats/aggregation.py`
- Modify: `backend/app/api/routes/stats.py`
- Test: `backend/tests/stats/test_aggregation.py`
- Test: `backend/tests/api/test_stats.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `ArmSummary` dataclass (`arm_name: str, n: int, mean_judge_score: float | None, mean_latency_ms: float | None, mean_cost_estimate_usd: float | None, mean_prompt_tokens: float | None, mean_completion_tokens: float | None`) and `async def summarize_arms(session, run_id, arm_names) -> list[ArmSummary]` in `app.stats.aggregation`. `n` is the count of `status == "completed"` rows for that arm (not metric-specific). `ArmSummaryResponse` (same fields as `ArmSummary`) and `GET /runs/{run_id}/summary` returning `list[ArmSummaryResponse]`, in the same order as `run.arm_names`.

- [ ] **Step 1: Write the failing aggregation-level tests**

Add to `backend/tests/stats/test_aggregation.py`:

```python
from app.stats.aggregation import load_metric_by_example, summarize_arms
```

(replace the existing single-name import line with the two-name version above), then add at the end of the file:

```python
def test_summarize_arms_computes_means_and_completed_count():
    example_ids = _insert_examples(2)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, judge_score=4.0, latency_ms=100.0)
        _insert_result(run_id, example_ids[1], "arm-a", 0, judge_score=2.0, latency_ms=200.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a"])

        result = asyncio.run(_query())
        assert len(result) == 1
        summary = result[0]
        assert summary.arm_name == "arm-a"
        assert summary.n == 2
        assert summary.mean_judge_score == pytest.approx(3.0)
        assert summary.mean_latency_ms == pytest.approx(150.0)
        assert summary.mean_cost_estimate_usd is None
    finally:
        _cleanup(run_id, example_ids)


def test_summarize_arms_excludes_non_completed_rows():
    example_ids = _insert_examples(2)
    run_id = _insert_run(["arm-a"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, status="failed", latency_ms=999.0)
        _insert_result(run_id, example_ids[1], "arm-a", 0, status="completed", latency_ms=50.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a"])

        result = asyncio.run(_query())
        summary = result[0]
        assert summary.n == 1
        assert summary.mean_latency_ms == pytest.approx(50.0)
    finally:
        _cleanup(run_id, example_ids)


def test_summarize_arms_returns_arms_in_requested_order_with_zero_n_when_absent():
    example_ids = _insert_examples(1)
    run_id = _insert_run(["arm-a", "arm-b"])
    try:
        _insert_result(run_id, example_ids[0], "arm-a", 0, latency_ms=10.0)

        async def _query():
            async with AsyncSession(db_test_engine) as session:
                return await summarize_arms(session, run_id, ["arm-a", "arm-b"])

        result = asyncio.run(_query())
        assert [s.arm_name for s in result] == ["arm-a", "arm-b"]
        assert result[1].n == 0
        assert result[1].mean_latency_ms is None
    finally:
        _cleanup(run_id, example_ids)
```

The `_insert_result` helper already in this file accepts `judge_score`/`latency_ms`/`status` kwargs — no changes needed there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/stats/test_aggregation.py -k test_summarize_arms -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_arms'`.

- [ ] **Step 3: Implement `summarize_arms` in `aggregation.py`**

Add to `backend/app/stats/aggregation.py` (after `load_metric_by_example`):

```python
from dataclasses import dataclass

from sqlalchemy import func

_SUMMARY_METRICS = ["judge_score", "latency_ms", "cost_estimate_usd", "prompt_tokens", "completion_tokens"]


@dataclass
class ArmSummary:
    arm_name: str
    n: int
    mean_judge_score: float | None
    mean_latency_ms: float | None
    mean_cost_estimate_usd: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None


async def summarize_arms(session: AsyncSession, run_id: int, arm_names: list[str]) -> list[ArmSummary]:
    n_result = await session.execute(
        select(RunResult.arm_name, func.count())
        .where(RunResult.run_id == run_id, RunResult.arm_name.in_(arm_names), RunResult.status == "completed")
        .group_by(RunResult.arm_name)
    )
    n_by_arm = dict(n_result.all())

    means: dict[str, dict[str, float]] = {name: {} for name in arm_names}
    for metric in _SUMMARY_METRICS:
        column = getattr(RunResult, metric)
        stmt = (
            select(RunResult.arm_name, func.avg(column))
            .where(
                RunResult.run_id == run_id,
                RunResult.arm_name.in_(arm_names),
                RunResult.status == "completed",
                column.is_not(None),
            )
            .group_by(RunResult.arm_name)
        )
        if metric == "judge_score":
            stmt = stmt.where(RunResult.judge_status == "completed")

        result = await session.execute(stmt)
        for arm_name, avg_value in result.all():
            means[arm_name][metric] = float(avg_value)

    return [
        ArmSummary(
            arm_name=arm_name,
            n=n_by_arm.get(arm_name, 0),
            mean_judge_score=means[arm_name].get("judge_score"),
            mean_latency_ms=means[arm_name].get("latency_ms"),
            mean_cost_estimate_usd=means[arm_name].get("cost_estimate_usd"),
            mean_prompt_tokens=means[arm_name].get("prompt_tokens"),
            mean_completion_tokens=means[arm_name].get("completion_tokens"),
        )
        for arm_name in arm_names
    ]
```

(`select` is already imported at the top of `aggregation.py`; add the `from dataclasses import dataclass` and `from sqlalchemy import func` imports alongside the existing ones.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/stats/test_aggregation.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Write the failing API-level test**

Add to `backend/tests/api/test_stats.py`, after `test_compare_arms_422_for_bootstrap_samples_over_cap`:

```python
def test_run_summary_returns_per_arm_means():
    run_id, example_ids = _seed_two_arm_run_judge_score(n_examples=4, offset=1.0)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/summary")
        assert response.status_code == 200
        body = {row["arm_name"]: row for row in response.json()}
        assert body["arm-a"]["n"] == 4
        assert body["arm-a"]["mean_judge_score"] == pytest.approx(2.5)
        assert body["arm-b"]["mean_judge_score"] == pytest.approx(1.5)
        assert body["arm-a"]["mean_cost_estimate_usd"] is None
    finally:
        _cleanup(run_id, example_ids)


def test_run_summary_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999/summary")
    assert response.status_code == 404
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_stats.py -k test_run_summary -v`
Expected: FAIL with `404 Not Found` for the summary route (route doesn't exist yet — note `test_run_summary_404_for_missing_run` will pass "by accident" with the wrong status reason; check the first test fails, that's the meaningful one).

- [ ] **Step 7: Add the endpoint to `stats.py`**

Add `summarize_arms` to the existing import line:

```python
from app.stats.aggregation import ALLOWED_METRICS, load_metric_by_example, summarize_arms
```

Add the response model and route (after the `PowerResponse` class, before `_load_run_or_404`, or anywhere alongside the other response models — place it near `PairedComparisonResponse` for readability):

```python
class ArmSummaryResponse(BaseModel):
    arm_name: str
    n: int
    mean_judge_score: float | None
    mean_latency_ms: float | None
    mean_cost_estimate_usd: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
```

Add the route (after `compare_arms`, before `equivalence`):

```python
@router.get("/{run_id}/summary", response_model=list[ArmSummaryResponse])
async def run_summary(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await _load_run_or_404(run_id, session)
    summaries = await summarize_arms(session, run_id, run.arm_names)
    return [ArmSummaryResponse(**vars(s)) for s in summaries]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_stats.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 9: Commit**

```bash
git add backend/app/stats/aggregation.py backend/app/api/routes/stats.py backend/tests/stats/test_aggregation.py backend/tests/api/test_stats.py
git commit -m "feat: add GET /runs/{run_id}/summary per-arm aggregate endpoint"
```

---

## Task 3: Add `GET /runs/{run_id}/calibration`

**Files:**
- Modify: `backend/app/api/routes/stats.py`
- Test: `backend/tests/api/test_stats.py`

**Interfaces:**
- Consumes: `app.judge.calibration.calibration_report(pairs: list[tuple[float, int]]) -> dict` (existing; raises `ValueError` on empty `pairs`), `app.db.models.JudgeCalibrationLabel` (existing: `run_result_id: int, human_score: int, labeled_by: str`).
- Produces: `CalibrationResponse` (`run_id: int, n: int, spearman_r: float, spearman_p: float, cohens_kappa: float, mean_abs_diff: float`), `GET /runs/{run_id}/calibration` — 404 `"Run not found"` for an unknown run, 404 `"No calibration labels for this run"` for a run with zero labeled results.

- [ ] **Step 1: Write the failing tests**

Add `JudgeCalibrationLabel` and `select` to the existing top-of-file imports in `backend/tests/api/test_stats.py`:

```python
from sqlmodel import delete, select
...
from app.db.models import EvalExample, JudgeCalibrationLabel, Run, RunResult
```

(`RunResult` is already imported there; extend that line rather than duplicating it.)

Then add a small insert helper for calibration labels, after `_insert_result`:

```python
def _insert_calibration_label(run_result_id: int, human_score: int) -> None:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            session.add(
                JudgeCalibrationLabel(run_result_id=run_result_id, human_score=human_score, labeled_by="test")
            )
            await session.commit()

    asyncio.run(_run())


def _completed_run_result_ids(run_id: int) -> list[int]:
    async def _run():
        async with AsyncSession(db_test_engine) as session:
            result = await session.execute(select(RunResult.id).where(RunResult.run_id == run_id).order_by(RunResult.id))
            return list(result.scalars().all())

    return asyncio.run(_run())
```

Then add the tests, after `test_run_summary_404_for_missing_run`:

```python
def test_calibration_returns_agreement_stats():
    run_id, example_ids = _seed_two_arm_run_judge_score(n_examples=4, offset=0.0)
    try:
        result_ids = _completed_run_result_ids(run_id)
        for result_id in result_ids:
            _insert_calibration_label(result_id, human_score=3)

        response = TestClient(app).get(f"/runs/{run_id}/calibration")
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        assert body["n"] == len(result_ids)
        assert "spearman_r" in body
        assert "cohens_kappa" in body
    finally:
        _cleanup(run_id, example_ids)


def test_calibration_404_when_no_labels():
    run_id, example_ids = _seed_two_arm_run_judge_score(n_examples=4, offset=0.0)
    try:
        response = TestClient(app).get(f"/runs/{run_id}/calibration")
        assert response.status_code == 404
        assert "No calibration labels" in response.json()["detail"]
    finally:
        _cleanup(run_id, example_ids)


def test_calibration_404_for_missing_run():
    response = TestClient(app).get("/runs/999999999/calibration")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_stats.py -k test_calibration -v`
Expected: FAIL — `test_calibration_returns_agreement_stats` and `test_calibration_404_when_no_labels` fail with 404 (route doesn't exist), `test_calibration_404_for_missing_run` fails on the `detail` assertion (generic 404, wrong body).

- [ ] **Step 3: Add the endpoint**

Add imports at the top of `stats.py`:

```python
from sqlmodel import select

from app.db.models import JudgeCalibrationLabel, Run, RunResult
from app.judge.calibration import calibration_report
```

(`Run` is already imported; extend that line to include `JudgeCalibrationLabel, RunResult`.)

Add the response model near the others:

```python
class CalibrationResponse(BaseModel):
    run_id: int
    n: int
    spearman_r: float
    spearman_p: float
    cohens_kappa: float
    mean_abs_diff: float
```

Add the route (after `run_summary`):

```python
@router.get("/{run_id}/calibration", response_model=CalibrationResponse)
async def calibration(run_id: int, session: AsyncSession = Depends(get_session)):
    await _load_run_or_404(run_id, session)

    result = await session.execute(
        select(RunResult.judge_score, JudgeCalibrationLabel.human_score)
        .join(JudgeCalibrationLabel, JudgeCalibrationLabel.run_result_id == RunResult.id)
        .where(RunResult.run_id == run_id)
    )
    pairs = [(judge_score, human_score) for judge_score, human_score in result.all()]
    if not pairs:
        raise HTTPException(status_code=404, detail="No calibration labels for this run")

    report = calibration_report(pairs)
    return CalibrationResponse(
        run_id=run_id,
        n=report["n"],
        spearman_r=report["spearman_r"],
        spearman_p=report["spearman_p"],
        cohens_kappa=report["cohens_kappa"],
        mean_abs_diff=report["mean_abs_diff"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_stats.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && uv run pytest -v`
Expected: PASS — no regressions from Tasks 1–3.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routes/stats.py backend/tests/api/test_stats.py
git commit -m "feat: add GET /runs/{run_id}/calibration endpoint"
```

---

## Task 4: Scaffold the frontend app

**Files:**
- Create: `frontend/` (via `npm create vite@latest`)
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/index.css`
- Modify: `frontend/src/main.tsx`

**Interfaces:**
- Consumes: nothing (this task has no dependency on Tasks 1–3's endpoints existing at runtime — it's scaffolding only).
- Produces: a running Vite dev server at `http://localhost:5173` serving a blank React page, with Tailwind's utility classes available and `/api/*` requests proxied to `http://localhost:8000`. Later tasks build on this.

Note: Tailwind v4 configures itself via the `@tailwindcss/vite` plugin and a single `@import "tailwindcss";` — no `tailwind.config.ts` file is needed (v3's separate-config-file approach doesn't apply here).

- [ ] **Step 1: Scaffold the Vite React-TS app**

Run from the repo root:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Install the runtime dependencies**

```bash
npm install @tanstack/react-query@^5 react-router-dom@^7 recharts@^3
npm install -D tailwindcss@^4 @tailwindcss/vite@^4
```

- [ ] **Step 3: Wire up Tailwind v4 via the Vite plugin**

Replace the contents of `frontend/vite.config.ts` with:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

Replace the contents of `frontend/src/index.css` with just:

```css
@import "tailwindcss";
```

Delete `frontend/src/App.css` (no longer used — Task 7 replaces `App.tsx`'s contents entirely).

- [ ] **Step 4: Verify the dev server runs with Tailwind active**

Temporarily replace `frontend/src/App.tsx`'s returned JSX with `<h1 className="text-3xl font-bold text-blue-600">Dashboard scaffold OK</h1>` and run:

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173` in a browser and confirm the heading renders in bold blue text (proves Tailwind classes are applied, not just default browser styling). Stop the dev server (Ctrl+C) once confirmed.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "chore: scaffold frontend app (Vite, React, TS, Tailwind)"
```

---

## Task 5: API client and shared types

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/components/QueryState.tsx`
- Create: `frontend/src/components/StatusBadge.tsx`

**Interfaces:**
- Consumes: the three new backend endpoints from Tasks 1–3, and the existing `GET /runs/{run_id}/compare`.
- Produces: `fetchRuns()`, `fetchRunSummary(runId)`, `fetchCompare(runId, metric)`, `fetchCalibration(runId)` (all `Promise`-returning, throwing `Error` with the backend's `detail` message on non-2xx) from `src/api/client.ts`; `RunSummary`, `ArmSummaryResponse`, `PairedComparisonResponse`, `CalibrationResponse` types from `src/api/types.ts`; `<QueryState isLoading error onRetry>{children}</QueryState>` and `<StatusBadge status />` components, used by every later page/component.

- [ ] **Step 1: Add response types**

Create `frontend/src/api/types.ts`:

```ts
export interface RunSummary {
  run_id: number;
  created_at: string;
  arm_names: string[];
  status: string;
  total_calls: number;
  completed: number;
  failed: number;
  pending: number;
}

export interface ArmSummaryResponse {
  arm_name: string;
  n: number;
  mean_judge_score: number | null;
  mean_latency_ms: number | null;
  mean_cost_estimate_usd: number | null;
  mean_prompt_tokens: number | null;
  mean_completion_tokens: number | null;
}

export interface PairedComparisonResponse {
  arm_a: string;
  arm_b: string;
  metric: string;
  n_examples: number;
  n_excluded: number;
  mean_diff: number;
  ci_lower: number;
  ci_upper: number;
  wilcoxon_statistic: number;
  p_value: number;
  p_value_corrected: number | null;
}

export interface CalibrationResponse {
  run_id: number;
  n: number;
  spearman_r: number;
  spearman_p: number;
  cohens_kappa: number;
  mean_abs_diff: number;
}
```

- [ ] **Step 2: Add the API client**

Create `frontend/src/api/client.ts`:

```ts
import type { ArmSummaryResponse, CalibrationResponse, PairedComparisonResponse, RunSummary } from './types';

const BASE = '/api';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchRuns(): Promise<RunSummary[]> {
  return getJson<RunSummary[]>('/runs');
}

export function fetchRunSummary(runId: number): Promise<ArmSummaryResponse[]> {
  return getJson<ArmSummaryResponse[]>(`/runs/${runId}/summary`);
}

export function fetchCompare(runId: number, metric: string): Promise<PairedComparisonResponse[]> {
  return getJson<PairedComparisonResponse[]>(`/runs/${runId}/compare?metric=${metric}`);
}

export function fetchCalibration(runId: number): Promise<CalibrationResponse> {
  return getJson<CalibrationResponse>(`/runs/${runId}/calibration`);
}
```

- [ ] **Step 3: Add the shared loading/error wrapper**

Create `frontend/src/components/QueryState.tsx`:

```tsx
import type { ReactNode } from 'react';

interface QueryStateProps {
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  children: ReactNode;
}

export function QueryState({ isLoading, error, onRetry, children }: QueryStateProps) {
  if (isLoading) {
    return <div className="p-4 text-sm text-gray-500">Loading…</div>;
  }
  if (error) {
    const message = error instanceof Error ? error.message : 'Something went wrong.';
    return (
      <div className="p-4 text-sm text-red-600">
        <p>{message}</p>
        <button onClick={onRetry} className="mt-2 rounded border px-2 py-1 text-xs">
          Retry
        </button>
      </div>
    );
  }
  return <>{children}</>;
}
```

- [ ] **Step 4: Add the status badge**

Create `frontend/src/components/StatusBadge.tsx`:

```tsx
const STATUS_STYLES: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-700',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  completed_with_errors: 'bg-yellow-100 text-yellow-800',
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-700';
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${style}`}>{status}</span>;
}
```

- [ ] **Step 5: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors (these files aren't wired into the app yet, but must compile standalone).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/ frontend/src/components/
git commit -m "feat: add frontend API client and shared query-state components"
```

---

## Task 6: Add `@tanstack/react-query` provider and app routing shell

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`
- Create: `frontend/src/pages/RunListPage.tsx` (placeholder)
- Create: `frontend/src/pages/RunDashboardPage.tsx` (placeholder)

**Interfaces:**
- Consumes: nothing new yet — this task wires routing and the query client so Tasks 7–9 have somewhere to plug in.
- Produces: two routes, `/` → `RunListPage` and `/runs/:runId` → `RunDashboardPage`, both wrapped in a `QueryClientProvider`.

- [ ] **Step 1: Add placeholder pages**

Create `frontend/src/pages/RunListPage.tsx`:

```tsx
export function RunListPage() {
  return <div className="p-6">Run list placeholder</div>;
}
```

Create `frontend/src/pages/RunDashboardPage.tsx`:

```tsx
import { useParams } from 'react-router-dom';

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  return <div className="p-6">Run dashboard placeholder for run #{runId}</div>;
}
```

- [ ] **Step 2: Wire up routing and the query client in `App.tsx`**

Replace the full contents of `frontend/src/App.tsx` with:

```tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { RunListPage } from './pages/RunListPage';
import { RunDashboardPage } from './pages/RunDashboardPage';

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<RunListPage />} />
          <Route path="/runs/:runId" element={<RunDashboardPage />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 3: Update `main.tsx` to use the named `App` export**

Replace the contents of `frontend/src/main.tsx` with:

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 4: Verify both routes render**

Run: `cd frontend && npm run dev`

In a browser: visit `http://localhost:5173/` and confirm "Run list placeholder" renders; visit `http://localhost:5173/runs/42` and confirm "Run dashboard placeholder for run #42" renders. Stop the dev server.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/main.tsx frontend/src/pages/
git commit -m "feat: add frontend routing shell with react-query provider"
```

---

## Task 7: `RunListPage` — run picker with live polling

**Files:**
- Modify: `frontend/src/pages/RunListPage.tsx`

**Interfaces:**
- Consumes: `fetchRuns()` from `src/api/client.ts` (Task 5), `QueryState`, `StatusBadge` (Task 5), `RunSummary` type (Task 5).
- Produces: the real `RunListPage`, linking each row to `/runs/:runId` (consumed by Task 6's routing, already wired).

- [ ] **Step 1: Implement the page**

Replace the contents of `frontend/src/pages/RunListPage.tsx` with:

```tsx
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchRuns } from '../api/client';
import { QueryState } from '../components/QueryState';
import { StatusBadge } from '../components/StatusBadge';
import type { RunSummary } from '../api/types';

const POLL_MS = 3000;
const TERMINAL_STATUSES = new Set(['completed', 'completed_with_errors']);

export function RunListPage() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
    refetchInterval: (query) => {
      const runs = query.state.data as RunSummary[] | undefined;
      const hasActiveRun = runs ? runs.some((run) => !TERMINAL_STATUSES.has(run.status)) : true;
      return hasActiveRun ? POLL_MS : false;
    },
  });

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Runs</h1>
      <QueryState isLoading={isLoading} error={error} onRetry={refetch}>
        {data && data.length === 0 && (
          <p className="text-sm text-gray-500">No runs yet — create one with POST /runs.</p>
        )}
        {data && data.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b text-xs uppercase text-gray-500">
                <th className="py-2">Run</th>
                <th>Created</th>
                <th>Arms</th>
                <th>Status</th>
                <th>Progress</th>
              </tr>
            </thead>
            <tbody>
              {data.map((run) => (
                <tr key={run.run_id} className="border-b hover:bg-gray-50">
                  <td className="py-2">
                    <Link to={`/runs/${run.run_id}`} className="text-blue-600 hover:underline">
                      #{run.run_id}
                    </Link>
                  </td>
                  <td>{new Date(run.created_at).toLocaleString()}</td>
                  <td>{run.arm_names.join(', ')}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td>
                    {run.completed + run.failed} / {run.total_calls}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </QueryState>
    </div>
  );
}
```

- [ ] **Step 2: Verify against the real backend**

Start the backend (`cd backend && uv run fastapi dev app/main.py`) and the frontend dev server (`cd frontend && npm run dev`). Visit `http://localhost:5173/`. If `POST /runs` has never been called, confirm the empty state renders. If runs exist, confirm the table renders with correct status badges and that an in-progress run's row updates without a manual refresh (watch for ~3s). Stop both servers.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/RunListPage.tsx
git commit -m "feat: implement RunListPage with live status polling"
```

---

## Task 8: `RunDashboardPage` — tabbed shell

**Files:**
- Modify: `frontend/src/pages/RunDashboardPage.tsx`
- Create: `frontend/src/components/WinRateTable.tsx` (placeholder)
- Create: `frontend/src/components/FrontierChart.tsx` (placeholder)
- Create: `frontend/src/components/CalibrationReport.tsx` (placeholder)

**Interfaces:**
- Consumes: nothing new — placeholders let this task be verified independently of Tasks 9–11.
- Produces: tab switching between `WinRateTable`, `FrontierChart`, `CalibrationReport`, each receiving `runId: number` as a prop — the exact prop name/type Tasks 9–11 must match.

- [ ] **Step 1: Add placeholder tab components**

Create `frontend/src/components/WinRateTable.tsx`:

```tsx
export function WinRateTable({ runId }: { runId: number }) {
  return <div>Win-rate placeholder for run #{runId}</div>;
}
```

Create `frontend/src/components/FrontierChart.tsx`:

```tsx
export function FrontierChart({ runId }: { runId: number }) {
  return <div>Frontier placeholder for run #{runId}</div>;
}
```

Create `frontend/src/components/CalibrationReport.tsx`:

```tsx
export function CalibrationReport({ runId }: { runId: number }) {
  return <div>Calibration placeholder for run #{runId}</div>;
}
```

- [ ] **Step 2: Implement the tabbed page**

Replace the contents of `frontend/src/pages/RunDashboardPage.tsx` with:

```tsx
import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { WinRateTable } from '../components/WinRateTable';
import { FrontierChart } from '../components/FrontierChart';
import { CalibrationReport } from '../components/CalibrationReport';

type TabKey = 'winrate' | 'frontier' | 'calibration';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'winrate', label: 'Win-rate' },
  { key: 'frontier', label: 'Frontier' },
  { key: 'calibration', label: 'Calibration' },
];

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<TabKey>('winrate');
  const runIdNum = Number(runId);

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Run #{runIdNum}</h1>
      <div className="mb-4 flex gap-2 border-b">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-3 py-2 text-sm ${
              tab === key ? 'border-b-2 border-blue-600 font-medium' : 'text-gray-500'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === 'winrate' && <WinRateTable runId={runIdNum} />}
      {tab === 'frontier' && <FrontierChart runId={runIdNum} />}
      {tab === 'calibration' && <CalibrationReport runId={runIdNum} />}
    </div>
  );
}
```

- [ ] **Step 3: Verify tab switching**

Run `npm run dev`, visit `http://localhost:5173/runs/1`, and click through all three tabs, confirming each placeholder renders and the active tab is visually highlighted. Stop the dev server.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/RunDashboardPage.tsx frontend/src/components/WinRateTable.tsx frontend/src/components/FrontierChart.tsx frontend/src/components/CalibrationReport.tsx
git commit -m "feat: add tabbed RunDashboardPage shell"
```

---

## Task 9: `WinRateTable` — real data

**Files:**
- Modify: `frontend/src/components/WinRateTable.tsx`

**Interfaces:**
- Consumes: `fetchRunSummary(runId)`, `fetchCompare(runId, metric)` (Task 5), `QueryState` (Task 5), `runId: number` prop (Task 8).

- [ ] **Step 1: Implement the component**

Replace the contents of `frontend/src/components/WinRateTable.tsx` with:

```tsx
import { useQuery } from '@tanstack/react-query';
import { fetchCompare, fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';

export function WinRateTable({ runId }: { runId: number }) {
  const summaryQuery = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });
  const compareQuery = useQuery({
    queryKey: ['run-compare', runId, 'judge_score'],
    queryFn: () => fetchCompare(runId, 'judge_score'),
  });

  const isLoading = summaryQuery.isLoading || compareQuery.isLoading;
  const error = summaryQuery.error ?? compareQuery.error;
  const retry = () => {
    summaryQuery.refetch();
    compareQuery.refetch();
  };

  return (
    <QueryState isLoading={isLoading} error={error} onRetry={retry}>
      {summaryQuery.data && (
        <table className="mb-6 w-full text-left text-sm">
          <thead>
            <tr className="border-b text-xs uppercase text-gray-500">
              <th className="py-2">Arm</th>
              <th>n</th>
              <th>Mean quality</th>
              <th>Mean latency (ms)</th>
              <th>Mean cost ($)</th>
            </tr>
          </thead>
          <tbody>
            {summaryQuery.data.map((row) => (
              <tr key={row.arm_name} className="border-b">
                <td className="py-2">{row.arm_name}</td>
                <td>{row.n}</td>
                <td>{row.mean_judge_score?.toFixed(2) ?? '—'}</td>
                <td>{row.mean_latency_ms?.toFixed(0) ?? '—'}</td>
                <td>{row.mean_cost_estimate_usd?.toFixed(4) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {compareQuery.data && (
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-xs uppercase text-gray-500">
              <th className="py-2">Arm A</th>
              <th>Arm B</th>
              <th>Mean diff</th>
              <th>95% CI</th>
              <th>p (corrected)</th>
              <th>Significant</th>
            </tr>
          </thead>
          <tbody>
            {compareQuery.data.map((row) => (
              <tr key={`${row.arm_a}-${row.arm_b}`} className="border-b">
                <td className="py-2">{row.arm_a}</td>
                <td>{row.arm_b}</td>
                <td>{row.mean_diff.toFixed(3)}</td>
                <td>
                  [{row.ci_lower.toFixed(3)}, {row.ci_upper.toFixed(3)}]
                </td>
                <td>{row.p_value_corrected?.toFixed(4) ?? '—'}</td>
                <td>{(row.p_value_corrected ?? 1) < 0.05 ? 'Yes' : 'No'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </QueryState>
  );
}
```

- [ ] **Step 2: Verify against a real completed run**

With the backend running and at least one run that has ≥2 arms and enough completed+judged results to clear `MIN_PAIRED_EXAMPLES` (see `backend/app/stats/errors.py` for the exact threshold), visit that run's Win-rate tab and confirm both tables render with real numbers. Then visit a run with too few paired examples and confirm the backend's 422 detail message renders as an inline notice (via `QueryState`'s error branch), not a blank table.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/WinRateTable.tsx
git commit -m "feat: wire WinRateTable to summary and compare endpoints"
```

---

## Task 10: `FrontierChart` — cost/latency/quality scatter

**Files:**
- Modify: `frontend/src/components/FrontierChart.tsx`

**Interfaces:**
- Consumes: `fetchRunSummary(runId)` (Task 5), `QueryState` (Task 5), `runId: number` prop (Task 8).

- [ ] **Step 1: Implement the component**

Replace the contents of `frontend/src/components/FrontierChart.tsx` with:

```tsx
import { useQuery } from '@tanstack/react-query';
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';

interface FrontierPoint {
  arm_name: string;
  cost: number;
  quality: number;
  latency: number;
  noCost: boolean;
}

export function FrontierChart({ runId }: { runId: number }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });

  const points: FrontierPoint[] = (data ?? []).map((row) => ({
    arm_name: row.arm_name,
    cost: row.mean_cost_estimate_usd ?? 0,
    quality: row.mean_judge_score ?? 0,
    latency: row.mean_latency_ms ?? 0,
    noCost: row.mean_cost_estimate_usd === null,
  }));

  return (
    <QueryState isLoading={isLoading} error={error} onRetry={refetch}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="cost" name="Mean cost ($)" />
          <YAxis type="number" dataKey="quality" name="Mean quality" />
          <ZAxis type="number" dataKey="latency" range={[100, 1000]} name="Mean latency (ms)" />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (!payload || payload.length === 0) return null;
              const point = payload[0].payload as FrontierPoint;
              return (
                <div className="rounded border bg-white p-2 text-xs shadow">
                  <p className="font-medium">{point.arm_name}</p>
                  <p>Quality: {point.quality.toFixed(2)}</p>
                  <p>Cost: {point.noCost ? 'no per-token cost — local compute' : `$${point.cost.toFixed(4)}`}</p>
                  <p>Latency: {point.latency.toFixed(0)} ms</p>
                </div>
              );
            }}
          />
          <Scatter data={points} fill="#2563eb" />
        </ScatterChart>
      </ResponsiveContainer>
    </QueryState>
  );
}
```

- [ ] **Step 2: Verify against a real run**

With the backend running and a run that has at least one API arm (has cost) and, ideally, the local arm (no cost) alongside it, visit the Frontier tab and confirm: one bubble per arm, bubble size visibly differs with latency, hovering shows the tooltip with the "no per-token cost — local compute" note for the local arm.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/FrontierChart.tsx
git commit -m "feat: wire FrontierChart to summary endpoint"
```

---

## Task 11: `CalibrationReport` — judge/human agreement panel

**Files:**
- Modify: `frontend/src/components/CalibrationReport.tsx`

**Interfaces:**
- Consumes: `fetchCalibration(runId)` (Task 5), `runId: number` prop (Task 8). Does not use `QueryState` — the 404 "no calibration" case needs distinct handling from a generic error.

- [ ] **Step 1: Implement the component**

Replace the contents of `frontend/src/components/CalibrationReport.tsx` with:

```tsx
import { useQuery } from '@tanstack/react-query';
import { fetchCalibration } from '../api/client';

export function CalibrationReport({ runId }: { runId: number }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['run-calibration', runId],
    queryFn: () => fetchCalibration(runId),
    retry: false,
  });

  if (isLoading) {
    return <div className="p-4 text-sm text-gray-500">Loading…</div>;
  }

  if (error) {
    const message = error instanceof Error ? error.message : 'Something went wrong.';
    if (message.toLowerCase().includes('no calibration')) {
      return (
        <p className="text-sm text-gray-500">
          No calibration sample recorded for this run — see select_calibration_sample.py.
        </p>
      );
    }
    return (
      <div className="p-4 text-sm text-red-600">
        <p>{message}</p>
        <button onClick={() => refetch()} className="mt-2 rounded border px-2 py-1 text-xs">
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <dl className="grid max-w-md grid-cols-2 gap-3 text-sm">
      <dt className="text-gray-500">n</dt>
      <dd>{data.n}</dd>
      <dt className="text-gray-500">Spearman r</dt>
      <dd>
        {data.spearman_r.toFixed(3)} (p = {data.spearman_p.toFixed(3)})
      </dd>
      <dt className="text-gray-500">Cohen&apos;s kappa</dt>
      <dd>{data.cohens_kappa.toFixed(3)}</dd>
      <dt className="text-gray-500">Mean abs diff</dt>
      <dd>{data.mean_abs_diff.toFixed(3)}</dd>
    </dl>
  );
}
```

- [ ] **Step 2: Verify both states against the real backend**

Visit the Calibration tab for a run with imported calibration labels (via `import_calibration_labels.py`) and confirm the stat panel renders with real numbers. Visit a run with none and confirm the "No calibration sample recorded..." message renders instead of an error.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/CalibrationReport.tsx
git commit -m "feat: wire CalibrationReport to calibration endpoint"
```

---

## Task 12: Full end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full backend test suite**

Run: `cd backend && uv run pytest -v`
Expected: PASS, no regressions across all of `backend/tests/`.

- [ ] **Step 2: Type-check the whole frontend**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Seed data and produce a real run**

Using the project's existing scripts (per `CLAUDE.md`'s Phase 2/3 workflow): run `backend/scripts/seed_eval_examples.py` if the eval table is empty, then `POST /runs` with a small `sample_size` (e.g. 10) across at least two arms with enough `repeats` to exercise the judge pipeline. Wait for it to reach a terminal status via `GET /runs/{run_id}`.

- [ ] **Step 4: Click through the full dashboard**

With both the backend (`uv run fastapi dev app/main.py`) and frontend (`npm run dev`) running:
1. Visit `/`, confirm the seeded run appears with the correct status and progress.
2. Click into the run, confirm the Win-rate tab shows both the per-arm summary and the pairwise comparison table with real numbers.
3. Switch to the Frontier tab, confirm one labeled bubble per arm renders with a sensible cost/quality/latency encoding.
4. Switch to the Calibration tab; if no calibration labels have been imported for this run, confirm the empty-state message renders (this is the expected state unless `select_calibration_sample.py` / `import_calibration_labels.py` were also run for this run_id).
5. Start a second run and confirm its row on `/` transitions from "pending"/"running" to a terminal status without a manual page refresh.

- [ ] **Step 5: Update `CLAUDE.md`**

In `CLAUDE.md`, change the Phase 5 bullet under "Build phases" from:

```
5. **Dashboard** — win-rate table with CIs, cost/latency/quality frontier,
   judge calibration report.
```

to mark it done, following the exact style of the Phase 1–4 entries (✅ **Done.** + one-line summary + file/spec pointers):

```
5. **Dashboard** ✅ **Done.** React app (`frontend/`) — run list with live
   status polling, and a per-run tabbed view: win-rate table (pairwise
   quality diff + CI + corrected p-value), cost/latency/quality frontier
   scatter, and judge calibration report. Backed by three new read-only
   endpoints (`GET /runs`, `/runs/{run_id}/summary`,
   `/runs/{run_id}/calibration`). Spec:
   `docs/superpowers/specs/2026-08-27-dashboard-design.md`; plan:
   `docs/superpowers/plans/2026-08-27-dashboard.md`.
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark Phase 5 dashboard done in CLAUDE.md"
```
