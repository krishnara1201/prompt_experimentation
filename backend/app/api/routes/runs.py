from datetime import datetime
from pathlib import Path
import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config.arms import load_arms
from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.tasks.worker import run_single_call

router = APIRouter(prefix="/runs", tags=["runs"])

ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class RunCreateRequest(BaseModel):
    # None means "use every configured arm". An explicit empty list would
    # produce total_calls == 0, a run that could never leave "pending", so
    # it is rejected with a 422 rather than accepted as a degenerate run.
    arms: list[str] | None = Field(default=None, min_length=1)
    sample_size: int | None = Field(default=None, gt=0)
    repeats: int = Field(default=1, ge=1)
    seed: int | None = None


class RunCreateResponse(BaseModel):
    run_id: int
    status: str
    total_calls: int


class RunStatusResponse(BaseModel):
    run_id: int
    status: str
    total_calls: int
    completed: int
    failed: int
    pending: int


class RunSummary(BaseModel):
    run_id: int
    created_at: datetime
    arm_names: list[str]
    status: str
    total_calls: int
    completed: int
    failed: int
    pending: int


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


@router.post("", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest, session: AsyncSession = Depends(get_session)):
    available_arms = load_arms(str(ARMS_PATH))
    arm_names = payload.arms if payload.arms is not None else list(available_arms.keys())
    unknown = [name for name in arm_names if name not in available_arms]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown arm(s): {', '.join(unknown)}")

    # ORDER BY is load-bearing: Postgres gives no row-order guarantee
    # without it, so the same seed must sample from the same ordered list
    # for a run to be reproducible.
    result = await session.execute(
        select(EvalExample.id, EvalExample.text).order_by(EvalExample.id)
    )
    all_examples = result.all()
    if not all_examples:
        raise HTTPException(status_code=400, detail="No eval examples found; run the seed script first")

    seed = payload.seed
    if payload.sample_size is not None:
        seed = seed if seed is not None else random.randrange(2**31)
        chosen = random.Random(seed).sample(all_examples, min(payload.sample_size, len(all_examples)))
    else:
        chosen = all_examples

    total_calls = len(chosen) * len(arm_names) * payload.repeats
    run = Run(
        arm_names=arm_names,
        sample_size=payload.sample_size,
        repeats=payload.repeats,
        seed=seed,
        total_calls=total_calls,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    run_id = run.id

    def _enqueue_all() -> None:
        for example_id, example_text in chosen:
            for arm_name in arm_names:
                queue = getattr(available_arms[arm_name], "celery_queue", "celery")
                for repeat_index in range(payload.repeats):
                    run_single_call.apply_async(
                        kwargs={
                            "run_id": run_id,
                            "example_id": example_id,
                            "example_text": example_text,
                            "arm_name": arm_name,
                            "repeat_index": repeat_index,
                        },
                        queue=queue,
                    )

    # .apply_async() is a synchronous Redis round-trip and there may be
    # thousands of them, so it must not run on the event loop.
    try:
        await run_in_threadpool(_enqueue_all)
    except Exception:
        # A partial enqueue would leave a Run row whose total_calls can
        # never be reached — stuck reporting pending/running forever.
        await session.delete(run)
        await session.commit()
        raise

    return RunCreateResponse(run_id=run_id, status="pending", total_calls=total_calls)


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


@router.get("/{run_id}/results", response_model=list[RunResult])
async def get_run_results(
    run_id: int, limit: int = 100, offset: int = 0, session: AsyncSession = Depends(get_session)
):
    # ORDER BY is required for offset/limit paging to be stable — without
    # it pages can duplicate or skip rows.
    result = await session.execute(
        select(RunResult)
        .where(RunResult.run_id == run_id)
        .order_by(RunResult.id)
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()
