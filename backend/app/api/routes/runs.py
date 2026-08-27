from pathlib import Path
import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.arms import load_arms
from app.db.models import EvalExample, Run, RunResult
from app.db.session import get_session
from app.tasks.worker import run_single_call

router = APIRouter(prefix="/runs", tags=["runs"])

ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class RunCreateRequest(BaseModel):
    arms: list[str] | None = None
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


@router.post("", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest, session: AsyncSession = Depends(get_session)):
    available_arms = load_arms(str(ARMS_PATH))
    arm_names = payload.arms if payload.arms is not None else list(available_arms.keys())
    unknown = [name for name in arm_names if name not in available_arms]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown arm(s): {', '.join(unknown)}")

    result = await session.execute(select(EvalExample.id, EvalExample.text))
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

    for example_id, example_text in chosen:
        for arm_name in arm_names:
            for repeat_index in range(payload.repeats):
                run_single_call.delay(
                    run_id=run.id,
                    example_id=example_id,
                    example_text=example_text,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                )

    return RunCreateResponse(run_id=run.id, status="pending", total_calls=total_calls)


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
    done = completed + failed
    pending = run.total_calls - done

    if done == 0:
        status = "pending"
    elif done < run.total_calls:
        status = "running"
    elif failed == 0:
        status = "completed"
    else:
        status = "completed_with_errors"

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
    result = await session.execute(
        select(RunResult).where(RunResult.run_id == run_id).offset(offset).limit(limit)
    )
    return result.scalars().all()
