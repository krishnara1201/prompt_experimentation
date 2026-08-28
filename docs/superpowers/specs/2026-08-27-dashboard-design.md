# Phase 5: Dashboard — Design

## Purpose

Give the win-rate/CI table, cost/latency/quality frontier, and judge
calibration report described in `CLAUDE.md`'s Phase 5 a real UI, backed by
the stats layer built in Phase 4. This is the first frontend in the project
(`frontend/` does not exist yet), so it is scoped as a new subsystem: a
small set of new read-only backend endpoints, plus a new React app that
consumes them and the existing `/runs/{run_id}/compare` endpoint.

## Non-goals

- No auth — this is a local developer tool, same trust model as the
  existing unauthenticated FastAPI app.
- No run creation/editing from the UI. Runs are still created via
  `POST /runs` (script or `curl`), matching how the project has been driven
  so far. The dashboard is read-only.
- No cross-run comparison view. Each dashboard page shows one run; comparing
  two runs side by side is out of scope for this phase.
- No `equivalence` or `power` views in the dashboard yet — those endpoints
  exist but take extra required query params (`epsilon`, target arms) that
  don't fit a passive dashboard view well. Left for a later iteration.
- No dedicated frontend automated test suite. Verified manually in-browser
  against a seeded run.

## Backend additions

Three new read-only endpoints, added to the existing route files rather
than a new "dashboard" module, so each stays independently reusable and
consistent with how `compare`/`equivalence`/`power` are already split out
in `stats.py`.

### `GET /runs` — list runs

Added to `backend/app/api/routes/runs.py`, next to the existing
`GET /runs/{run_id}`.

The per-run status computation currently inline in `get_run_status` (lines
121–137 of `runs.py`) is extracted into a helper `_compute_status(run,
counts) -> tuple[str, int, int, int]` (status, completed, failed, pending),
used by both the existing single-run endpoint and the new list endpoint —
avoids duplicating the pending/running/completed/completed_with_errors
logic.

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

@router.get("", response_model=list[RunSummary])
async def list_runs(session: AsyncSession = Depends(get_session)) -> list[RunSummary]: ...
```

Query: select all `Run` rows ordered by `created_at` descending, plus one
grouped count query across all run_ids (`GROUP BY run_id, status`) rather
than N+1 queries per run. No pagination for v1 — acceptable for a
portfolio-scale number of runs; revisit if this becomes a real workload.

### `GET /runs/{run_id}/summary` — per-arm aggregates

Added to `backend/app/api/routes/stats.py`. Powers the frontier scatter
(one point per arm) and gives the win-rate tab a quality/cost/latency
overview above the pairwise table.

New function in `backend/app/stats/aggregation.py`:

```python
@dataclass
class ArmSummary:
    arm_name: str
    n: int
    mean_judge_score: float | None
    mean_latency_ms: float | None
    mean_cost_estimate_usd: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None

async def summarize_arms(session: AsyncSession, run_id: int, arm_names: list[str]) -> list[ArmSummary]: ...
```

Implementation: one query per metric column (mirroring
`load_metric_by_example`'s existing filter: `status == "completed"`,
`column.is_not(None)`, plus `judge_status == "completed"` when the column is
`judge_score`), grouped by `arm_name`, averaged in SQL (`func.avg`). An arm
with zero completed rows for a metric reports `None` for that field rather
than being dropped — the frontend renders that field as "—".

```python
class ArmSummaryResponse(BaseModel):
    arm_name: str
    n: int
    mean_judge_score: float | None
    mean_latency_ms: float | None
    mean_cost_estimate_usd: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None

@router.get("/{run_id}/summary", response_model=list[ArmSummaryResponse])
async def run_summary(run_id: int, session: AsyncSession = Depends(get_session)): ...
```

No `bootstrap_samples`-style tuning params — this is a plain mean, not a
statistical test.

### `GET /runs/{run_id}/calibration` — judge/human agreement

Added to `backend/app/api/routes/stats.py`. Wraps the existing
`app.judge.calibration.calibration_report()` (already used by the
`calibration_report.py` CLI script) behind the API, using the same join
query the script uses (`RunResult.judge_score` ×
`JudgeCalibrationLabel.human_score` for the run's results).

```python
class CalibrationResponse(BaseModel):
    run_id: int
    n: int
    spearman_r: float
    spearman_p: float
    cohens_kappa: float
    mean_abs_diff: float

@router.get("/{run_id}/calibration", response_model=CalibrationResponse)
async def calibration(run_id: int, session: AsyncSession = Depends(get_session)): ...
```

`calibration_report()` raises `ValueError` when given zero pairs (no
calibration labels imported for this run yet). The endpoint catches that
and returns `404` with detail `"No calibration labels for this run"` —
distinct from "run not found" (`404` on the run itself, checked first) so
the frontend can render "no calibration sample for this run" instead of a
generic error.

### Testing

`backend/tests/api/test_runs.py` and `test_stats.py` get new test
functions following the existing pattern (httpx `AsyncClient` against the
app, fixtures seeding `Run`/`RunResult`/`JudgeCalibrationLabel` rows):

- `list_runs`: empty DB → `[]`; multiple runs → correct status/counts per
  run, ordered newest-first.
- `run_summary`: per-arm means computed correctly; an arm with no completed
  rows for a metric reports `None` for that field, not a 500 or a dropped
  row.
- `calibration`: happy path against seeded labels; run with no labels →
  404 with the specific detail message; unknown run_id → 404 "Run not
  found".

## Frontend (`frontend/`, new)

### Stack

- Vite + React + TypeScript (per `CLAUDE.md`).
- Tailwind CSS for styling.
- Recharts for the frontier scatter.
- `@tanstack/react-query` for data fetching; its `refetchInterval` gives
  polling for in-progress runs without hand-rolled polling logic.
- `react-router` for the two routes below.

### Dev proxy (avoids backend CORS changes)

The backend has no CORS middleware today, and adding permissive CORS to a
FastAPI app just to satisfy a dev-server origin mismatch is unnecessary.
Instead, `vite.config.ts` proxies `/api/*` to `http://localhost:8000/*` in
dev, so the frontend always calls same-origin `/api/...` paths. In a future
production build this proxy would be replaced by a reverse-proxy config,
but that's out of scope here — dev-only for now.

### Structure

```
frontend/
  src/
    api/
      client.ts        # typed fetch wrappers, one per endpoint, mirroring the Pydantic response models
      types.ts          # RunSummary, ArmSummaryResponse, PairedComparisonResponse, CalibrationResponse
    pages/
      RunListPage.tsx
      RunDashboardPage.tsx
    components/
      StatusBadge.tsx
      WinRateTable.tsx
      FrontierChart.tsx
      CalibrationReport.tsx
    App.tsx             # routes: "/" -> RunListPage, "/runs/:runId" -> RunDashboardPage
    main.tsx
  vite.config.ts
  tailwind.config.ts
  package.json
```

### `RunListPage`

- Fetches `GET /api/runs` via React Query.
- Table: run id, created_at, arm_names (comma-joined), status badge,
  completed/total_calls.
- Any run whose status is `pending` or `running` polls (`refetchInterval:
  3000`) until it reaches a terminal status (`completed` /
  `completed_with_errors`); terminal-status rows don't poll.
- Row click navigates to `/runs/:runId`.
- Empty state: "No runs yet — create one with `POST /runs`."

### `RunDashboardPage`

Tabbed layout (`Win-rate` / `Frontier` / `Calibration`), all three tabs
fetching for the same `runId` from the URL param.

- **Win-rate tab**: `GET /api/runs/{runId}/compare?metric=judge_score`
  (all pairs, default `bootstrap_samples`) rendered as a table: arm A, arm
  B, mean diff, 95% CI (`ci_lower`–`ci_upper`), corrected p-value, and a
  "significant" flag (`p_value_corrected < 0.05`). Also shows the
  `GET /api/runs/{runId}/summary` means per arm above the table as a quick
  reference. A run with fewer than the stats layer's minimum paired
  examples surfaces the backend's 422 detail message as an inline notice
  instead of an empty table.
- **Frontier tab**: `GET /api/runs/{runId}/summary` rendered as a Recharts
  `ScatterChart` — x = `mean_cost_estimate_usd`, y = `mean_judge_score`,
  bubble radius scaled from `mean_latency_ms`, one labeled point per arm.
  Arms missing a required field (e.g. the local Ollama arm has no cost) are
  plotted at x=0 with a note ("no per-token cost — local compute") rather
  than being dropped, since the local-vs-API cost story is the point of
  this chart per `CLAUDE.md`.
- **Calibration tab**: `GET /api/runs/{runId}/calibration` rendered as a
  small stat panel (n, Spearman r + p, Cohen's kappa, mean abs diff). On
  404 ("no calibration labels"), renders "No calibration sample recorded
  for this run — see `select_calibration_sample.py`" instead of an error
  state.

### Error/loading states

Every fetch goes through the same small `useApiQuery` wrapper (thin layer
over `useQuery`) so loading/error rendering is consistent across tabs:
spinner while loading, backend's `detail` message on 4xx, generic retry
button on network/5xx errors.

## Testing plan

- Backend: pytest additions described above, run via `uv run pytest`.
- Frontend: no automated suite for this MVP. Manual verification via the
  `run` skill: seed eval examples, create a run through completion (small
  sample size for speed), start the Vite dev server, and click through all
  three tabs confirming real data renders, including the empty-calibration
  and insufficient-data-for-compare states.
