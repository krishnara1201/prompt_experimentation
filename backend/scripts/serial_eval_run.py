"""Run an eval set × arms paired comparison in-process — no Celery/Redis.

A low-memory alternative to the Celery orchestrator (same motivation as
``serial_judge_run.py``) that also **pauses cleanly when the Claude Code
subscription seat hits its usage limit** and resumes once the window
resets. Needs only Postgres reachable at ``DATABASE_URL`` and each arm's
backend reachable (Ollama for a local arm, an authenticated ``claude`` CLI
for a subscription-CLI arm).

    # start a fresh run
    uv run python -m scripts.serial_eval_run new \
        --arms qwen3-8b-local,claude-code-sonnet \
        --task financial_sentiment --sample-size 150 --seed 20260831

    # continue it after the usage window resets (idempotent — skips
    # cells already completed)
    uv run python -m scripts.serial_eval_run resume <run_id>

    # print the resolved arms + cell plan, touch nothing
    uv run python -m scripts.serial_eval_run new --arms ... --dry-run

When done, judge with:  uv run python -m scripts.serial_judge_run <run_id>
"""
import argparse
import asyncio
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.claude_code_cli import UsageLimitError
from app.config.arms import Arm, load_arms
from app.config.tasks import load_task
from app.db.models import Run
from app.db.session import engine
from app.tasks.worker import ARMS_PATH

# One short retry papers over a transient blip; the operator re-runs for
# anything real (this runner is interactive, not a daemon).
TRANSIENT_RETRY_DELAY_S = 2.0


@dataclass
class Cell:
    example_id: int
    example_text: str
    arm_name: str
    repeat_index: int


@dataclass
class Outcome:
    paused: bool
    retry_at: datetime | None
    ok: int
    fail: int
    remaining: int


def plan_cells(
    chosen: list[tuple[int, str]],
    arm_names: list[str],
    repeats: int,
    cli_arms: set[str],
    completed: set[tuple[int, str, int]],
) -> list[Cell]:
    """Every (example, arm, repeat) cell still missing a completed result,
    non-CLI arms first so a mid-run pause never leaves a half-done local arm."""
    ordered = [a for a in arm_names if a not in cli_arms] + [
        a for a in arm_names if a in cli_arms
    ]
    cells: list[Cell] = []
    for arm_name in ordered:
        for example_id, example_text in chosen:
            for repeat_index in range(repeats):
                if (example_id, arm_name, repeat_index) in completed:
                    continue
                cells.append(Cell(example_id, example_text, arm_name, repeat_index))
    return cells


def resolve_arms(task_name: str, arm_names: list[str]) -> tuple[dict[str, Arm], set[str]]:
    task_cfg = load_task(task_name)
    all_arms = load_arms(str(ARMS_PATH), task=task_cfg)
    unknown = [n for n in arm_names if n not in all_arms]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}; have: {', '.join(all_arms)}")
    arms = {n: all_arms[n] for n in arm_names}
    cli_arms = {
        n for n, arm in arms.items()
        if getattr(arm.adapter, "celery_queue", None) == "subscription_cli"
    }
    return arms, cli_arms


async def _sample_examples(source: str, sample_size: int | None, seed: int | None) -> list[tuple[int, str]]:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT id, text FROM eval_example WHERE source = :s ORDER BY id"),
                {"s": source},
            )
        ).all()
    all_examples = [(r[0], r[1]) for r in rows]
    if sample_size is None:
        return all_examples
    return random.Random(seed).sample(all_examples, min(sample_size, len(all_examples)))


async def _completed_cells(run_id: int) -> set[tuple[int, str, int]]:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT example_id, arm_name, repeat_index FROM run_result "
                    "WHERE run_id = :r AND status = 'completed'"
                ),
                {"r": run_id},
            )
        ).all()
    return {(r[0], r[1], r[2]) for r in rows}


async def _persist(run_id: int, cell: Cell, status: str, response=None, error: str | None = None) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO run_result "
                "(run_id, example_id, arm_name, repeat_index, status, output_text, "
                " latency_ms, prompt_tokens, completion_tokens, cost_estimate_usd, "
                " error_message, judge_status, created_at) "
                "VALUES (:run_id, :example_id, :arm_name, :repeat_index, :status, :output_text, "
                " :latency_ms, :prompt_tokens, :completion_tokens, :cost_estimate_usd, "
                " :error_message, 'pending', :created_at)"
            ),
            {
                "run_id": run_id,
                "example_id": cell.example_id,
                "arm_name": cell.arm_name,
                "repeat_index": cell.repeat_index,
                "status": status,
                "output_text": response.text if response else None,
                "latency_ms": response.latency_ms if response else None,
                "prompt_tokens": response.prompt_tokens if response else None,
                "completion_tokens": response.completion_tokens if response else None,
                "cost_estimate_usd": response.cost_estimate_usd if response else None,
                "error_message": error,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )


async def _load_run(run_id: int) -> Run | None:
    async with AsyncSession(engine) as session:
        return await session.get(Run, run_id)


async def run_cells(
    *,
    run_id: int,
    chosen: list[tuple[int, str]],
    arms: dict[str, Arm],
    cli_arms: set[str],
    task_name: str,
    repeats: int = 1,
) -> Outcome:
    completed = await _completed_cells(run_id)
    cells = plan_cells(chosen, list(arms), repeats, cli_arms, completed)
    total = len(cells)
    print(f"run {run_id}: {total} cells outstanding ({len(completed)} already done)")

    ok = fail = 0
    for i, cell in enumerate(cells, 1):
        adapter = arms[cell.arm_name].adapter
        prompt = arms[cell.arm_name].render(cell.example_text)
        try:
            response = _call_with_one_retry(adapter, prompt)
        except UsageLimitError as exc:
            print(f"\nPAUSED: usage limit hit after {i - 1}/{total} outstanding cells.")
            if exc.retry_at is not None:
                delta = exc.retry_at - datetime.now(timezone.utc)
                mins = max(0, int(delta.total_seconds() // 60))
                print(f"  resets at {exc.retry_at:%Y-%m-%d %H:%M %Z} (~{mins} min)")
            print(f"  resume with:  uv run python -m scripts.serial_eval_run resume {run_id}")
            return Outcome(paused=True, retry_at=exc.retry_at, ok=ok, fail=fail, remaining=total - (i - 1))
        except Exception as exc:  # noqa: BLE001 - record and continue
            await _persist(run_id, cell, "failed", error=str(exc))
            fail += 1
            print(f"  [{i}/{total}] {cell.arm_name} ex={cell.example_id} FAILED: {exc!r}")
            continue

        await _persist(run_id, cell, "completed", response=response)
        ok += 1
        if i % 10 == 0:
            print(f"  {i}/{total}  ok={ok} fail={fail}")

    return Outcome(paused=False, retry_at=None, ok=ok, fail=fail, remaining=0)


def _call_with_one_retry(adapter, prompt: str):
    try:
        return adapter.generate(prompt)
    except UsageLimitError:
        raise
    except Exception:
        time.sleep(TRANSIENT_RETRY_DELAY_S)
        return adapter.generate(prompt)


async def _create_run(
    *, task_name: str, arm_names: list[str], sample_size: int | None, seed: int | None, repeats: int
) -> tuple[int, list[tuple[int, str]], int | None]:
    task_cfg = load_task(task_name)
    if sample_size is not None and seed is None:
        seed = random.randrange(2**31)
    chosen = await _sample_examples(task_cfg.source, sample_size, seed)
    if not chosen:
        raise SystemExit(
            f"no eval examples for task {task_name!r}; run `pe seed --task {task_name}` first"
        )
    total_calls = len(chosen) * len(arm_names) * repeats
    async with AsyncSession(engine, expire_on_commit=False) as session:
        run = Run(
            task=task_name,
            arm_names=arm_names,
            sample_size=sample_size,
            repeats=repeats,
            seed=seed,
            total_calls=total_calls,
        )
        session.add(run)
        await session.commit()
        return run.id, chosen, seed


async def _main_new(args) -> int:
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms, cli_arms = resolve_arms(args.task, arm_names)

    if args.dry_run:
        chosen = await _sample_examples(load_task(args.task).source, args.sample_size, args.seed)
        cells = plan_cells(chosen, arm_names, args.repeats, cli_arms, set())
        print(f"task={args.task}  arms={arm_names}  cli_arms={sorted(cli_arms)}")
        print(f"{len(chosen)} examples × {len(arm_names)} arms × {args.repeats} repeats = {len(cells)} cells")
        return 0

    run_id, chosen, seed = await _create_run(
        task_name=args.task, arm_names=arm_names,
        sample_size=args.sample_size, seed=args.seed, repeats=args.repeats,
    )
    print(f"created run {run_id} (seed={seed}, {len(chosen)} examples)")
    return await _drive(run_id, chosen, arms, cli_arms, args.task, args.repeats)


async def _main_resume(args) -> int:
    run = await _load_run(args.run_id)
    if run is None:
        print(f"run {args.run_id} not found", file=sys.stderr)
        return 2
    arms, cli_arms = resolve_arms(run.task, run.arm_names)
    task_cfg = load_task(run.task)
    chosen = await _sample_examples(task_cfg.source, run.sample_size, run.seed)
    return await _drive(args.run_id, chosen, arms, cli_arms, run.task, run.repeats)


async def _drive(run_id, chosen, arms, cli_arms, task_name, repeats) -> int:
    outcome = await run_cells(
        run_id=run_id, chosen=chosen, arms=arms, cli_arms=cli_arms,
        task_name=task_name, repeats=repeats,
    )
    if outcome.paused:
        return 0
    print(
        f"\ndone: run {run_id}  ok={outcome.ok} fail={outcome.fail}\n"
        f"  judge with:  uv run python -m scripts.serial_judge_run {run_id}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new", help="start a fresh run")
    new.add_argument("--arms", required=True, help="comma-separated arm names from arms.yaml")
    new.add_argument("--task", default="financial_sentiment")
    new.add_argument("--sample-size", type=int, default=None)
    new.add_argument("--seed", type=int, default=None)
    new.add_argument("--repeats", type=int, default=1)
    new.add_argument("--dry-run", action="store_true")

    res = sub.add_parser("resume", help="continue an existing run")
    res.add_argument("run_id", type=int)

    args = parser.parse_args(argv)

    async def _amain() -> int:
        try:
            return await (_main_new(args) if args.cmd == "new" else _main_resume(args))
        finally:
            await engine.dispose()

    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
