# `pe` CLI + Dashboard "New Run" — Design Spec

Date: 2026-08-29
Status: Approved for implementation

## Context

Everything actionable in the platform today is driven by raw `curl` or
by `uv run python -m scripts.*` invocations:

| Step | Today |
| --- | --- |
| Bring up the stack | `docker compose up --build` |
| Seed the eval dataset | `docker compose run --rm migrate uv run python -m scripts.seed_eval_examples` |
| Start a run | `curl -X POST localhost:8000/runs -d '{...}'` |
| Poll status / results | `curl localhost:8000/runs/{id}` … |
| Calibration | three `uv run python -m scripts.*` commands |
| Stats | `curl .../runs/{id}/compare\|equivalence\|power` |
| View results | the React dashboard (read-only) |

The dashboard is view-only — its empty state literally reads *"create one
with POST /runs"*. This spec adds three thin convenience layers over the
flows that already exist. No new orchestration, stats, or judging logic.

## Goals

1. A single `pe` command that covers the whole day-to-day loop: stack
   lifecycle, seeding, starting/watching runs, reading stats, and the
   calibration workflow.
2. A "New Run" form in the dashboard so a run can be started without a
   terminal.
3. One `scripts/demo.sh` that runs the full happy path end to end (a
   portfolio "watch this" command that doubles as a CLI smoke test).

## Non-goals

- No new UI panels for equivalence/power (the `compare` table is already
  in `WinRateTable`; the CLI covers the rest). Easy follow-up.
- The CLI is not packaged for distribution — it runs from source via
  `uv run pe` from `backend/`. `[project.scripts]` is for that, not PyPI.
- No auth. The API is already bound to `127.0.0.1` only.

## 1. `GET /arms` endpoint

New module `backend/app/api/routes/arms.py`, registered in
`app/main.py`. Read-only, no DB.

```
GET /arms  ->  200  [{"name": str, "adapter": str, "model": str | null}, ...]
```

Implementation loads `arms.yaml` via the existing
`app.config.arms.load_arms(str(ARMS_PATH))` (same `ARMS_PATH` constant
pattern as `routes/runs.py`), then for each arm reports its key, the
adapter class (`type(adapter).__name__` mapped back through
`ADAPTER_TYPES`, or simply the adapter's registered type string — see
below), and `getattr(adapter, "model", None)`.

`load_arms` returns `dict[str, ModelAdapter]` and drops the adapter-type
string. Rather than reverse-map the class, `arms.py` re-reads the raw
YAML for the `adapter` field (it is cheap and keeps the endpoint honest
about what the file says). A malformed `arms.yaml` surfaces as a 500 with
the loader's `ValueError` message — acceptable for a localhost dev tool.

Consumed by both the dashboard form (arm checklist) and the CLI (`pe run`
`--arm` validation happens server-side already via `POST /runs`'s
existing unknown-arm 400; the CLI just uses `/arms` for `--list-arms`).

**Test** (`tests/api/test_arms.py`): patch `app.api.routes.arms.load_arms`
with a fake mapping and assert the shape. No Postgres needed, so no
`postgres_reachable` skip.

## 2. The `pe` CLI

### Packaging

- `backend/pyproject.toml`: add `typer>=0.12` to `dependencies`; add

  ```toml
  [project.scripts]
  pe = "app.cli:app"
  ```

- Invoked as `uv run pe …` from `backend/` (uv auto-syncs). Docker-only
  users run it the same way — `uv run` needs only the lockfile, not a
  hand-managed venv.

### Module layout — `backend/app/cli/`

| File | Responsibility |
| --- | --- |
| `__init__.py` | builds the root `typer.Typer()` (`app`), mounts sub-apps, defines top-level commands |
| `_api.py` | `api_get(path, **params)` / `api_post(path, json)` — thin `httpx` wrapper around `$PE_API_URL` (default `http://localhost:8000`); raises `typer.Exit(1)` after printing the API's `detail` on non-2xx or a connection error |
| `_shell.py` | `repo_root()` (walk up for `docker-compose.yml`), `compose(*args)` and `backend_script(*args)` — both call `_run(argv)`, a module-level indirection tests monkeypatch |
| `_render.py` | small helpers to print a list-of-dicts as an aligned table and a dict as key/value lines (no Rich dependency beyond what Typer pulls in) |

### Commands

Stack lifecycle (shell out to `docker compose`, cwd = repo root):

| Command | Runs |
| --- | --- |
| `pe up [--wait/--no-wait]` | `docker compose up -d --build` (+ `--wait`), then polls `GET /runs` until the API answers or 60s elapse |
| `pe down [--volumes]` | `docker compose down [-v]` |
| `pe logs [SERVICE] [--follow]` | `docker compose logs [-f] [service]` |
| `pe seed` | `docker compose run --rm migrate uv run python -m scripts.seed_eval_examples` |

Run lifecycle (HTTP):

| Command | Calls |
| --- | --- |
| `pe run [--sample N] [--repeats N] [--seed N] [--arm A ...] [-q]` | `POST /runs`; prints a summary table, or just the integer `run_id` with `-q` |
| `pe status RUN_ID` | `GET /runs/{id}` |
| `pe watch RUN_ID [--interval S]` | polls `GET /runs/{id}` until status is `completed`/`completed_with_errors`, redrawing a progress line; exit code 1 if any call failed |
| `pe results RUN_ID [--limit N] [--offset N]` | `GET /runs/{id}/results` |
| `pe arms` | `GET /arms` |

Stats (HTTP, `pe stats` sub-app):

| Command | Calls |
| --- | --- |
| `pe stats compare RUN_ID [--metric M]` | `GET /runs/{id}/compare` |
| `pe stats equivalence RUN_ID --local A --api B [--metric judge_score] [--eps E]` | `GET /runs/{id}/equivalence` |
| `pe stats power RUN_ID --arm-a A --arm-b B [--metric M] [--power P] [--alpha A]` | `GET /runs/{id}/power` |

Calibration (`pe calibrate` sub-app) — these need DB access and read/write
a local JSON file you hand-label, so they run on the **host** exactly as
`backend/README.md` Phase 3 documents (`uv run python -m scripts.*` with
cwd = `backend/`, requires a local `.env` with a `localhost` `DATABASE_URL`):

| Command | Runs |
| --- | --- |
| `pe calibrate select --run-id N --out PATH [--n N] [--seed N]` | `scripts.select_calibration_sample` |
| `pe calibrate import --in PATH --labeled-by WHO` | `scripts.import_calibration_labels` |
| `pe calibrate report --run-id N` | `scripts.calibration_report` |

### Config

Single env var `PE_API_URL` (default `http://localhost:8000`). Documented
in `.env.example` as an optional override. No config file.

### Error handling

- Connection refused / DNS failure → `"Cannot reach the API at <url>. Is
  the stack up? Try: pe up"`, exit 1.
- Non-2xx → print `response.json()["detail"]` (falls back to status line),
  exit 1.
- `docker compose` / script subprocess non-zero → its exit code
  propagates; no wrapping.

### Tests — `backend/tests/cli/`

- `respx` (already a dev dep) + Typer's `CliRunner`:
  - `pe run --sample 5 --repeats 2 --seed 1 --arm x` issues one
    `POST /runs` with the expected JSON body; `-q` prints only the id.
  - `pe status` / `pe stats compare` render canned responses without
    error and include key fields.
  - a non-2xx response prints `detail` and exits 1; a connection error
    prints the "is the stack up?" hint and exits 1.
- Shell commands: monkeypatch `app.cli._shell._run` and assert the argv
  (`["docker", "compose", "up", "-d", "--build", "--wait"]`,
  `["docker", "compose", "run", "--rm", "migrate", ...]`, and the
  `uv run python -m scripts.select_calibration_sample --run-id 1 ...`
  form). No Docker, no Postgres.
- `repo_root()` finds the directory containing `docker-compose.yml`.

## 3. Dashboard "New Run" form

- `frontend/src/api/types.ts`: add `ArmInfo`, `RunCreateRequest`,
  `RunCreateResponse`.
- `frontend/src/api/client.ts`: add `fetchArms()` → `GET /api/arms` and
  `createRun(body)` → `POST /api/runs` (new `postJson` helper mirroring
  the existing `getJson`, surfacing `detail` on error).
- `frontend/src/components/NewRunForm.tsx`: a "New run" button on
  `RunListPage` that toggles an inline panel. Fields: sample size
  (number, blank = whole dataset), repeats (number, default 1), seed
  (number, optional), and a checkbox list of arms from `fetchArms()`
  (none checked = all arms, matching `POST /runs` semantics). Submit uses
  a React Query `useMutation`; on success it invalidates `['runs']` and
  `navigate(\`/runs/${run_id}\`)`. Validation errors from the API render
  above the form.
- `RunListPage` empty-state text changes from *"create one with POST
  /runs"* to a prompt pointing at the new button.
- No frontend test runner exists in the repo; verification is
  `npm run build` (`tsc -b`) + `npm run lint` (oxlint) clean, plus a
  manual click-through.

## 4. `scripts/demo.sh`

Repo-root bash script, `set -euo pipefail`:

```
pe up --wait
pe seed
RUN_ID=$(pe run --sample "${DEMO_SAMPLE:-10}" --repeats "${DEMO_REPEATS:-3}" --seed 42 -q)
pe watch "$RUN_ID"
pe stats compare "$RUN_ID"
echo "Dashboard: http://localhost:${FRONTEND_PORT:-5173}/runs/$RUN_ID"
```

It `cd`s to `backend/` and calls `uv run pe …` so a clean checkout needs
only Docker + uv. Overridable sample/repeat counts via env vars.

## 5. Docs

- `README.md`: replace the raw-`curl` "Then kick off a run" block with
  `pe` equivalents; add a one-line `./scripts/demo.sh` mention.
- `backend/README.md`: add a "`pe` CLI" subsection with the command
  table; keep the `curl` examples (they still document the raw API) but
  note `pe` as the friendlier path.
- `CLAUDE.md`: one line under the architecture/run notes pointing at the
  CLI and `demo.sh`.
- `.env.example`: document `PE_API_URL` as an optional override.

## Rejected alternatives

- **Makefile / justfile.** Discoverable but can't parametrize a run
  cleanly (`make run SAMPLE=20 ARMS=a,b` is worse than `pe run`), and
  would still shell to `curl`. `demo.sh` covers the "one command"
  want without it.
- **A standalone repo-root `./pe` uv-script.** Avoids `cd backend` but
  duplicates the API-base and repo-root logic and can't import the arm
  config helpers. Chosen against per the packaging decision above.
- **Surfacing all stats in the dashboard.** More work than the ask;
  the CLI covers equivalence/power today and the UI can grow later.
```