from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.tasks import active_task_name, list_tasks, load_task
from app.db.models import EvalExample
from app.db.session import get_session

router = APIRouter(prefix="/tasks", tags=["tasks"])

# Same file the arms/run routes resolve config against.
ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class TaskInfo(BaseModel):
    name: str
    description: str
    labels: list[str]
    active: bool
    seeded_count: int


@router.get("", response_model=list[TaskInfo])
async def list_task_packs(
    session: AsyncSession = Depends(get_session),
) -> list[TaskInfo]:
    active = active_task_name(str(ARMS_PATH))
    rows = (
        await session.execute(
            select(EvalExample.source, func.count()).group_by(EvalExample.source)
        )
    ).all()
    counts = {source: count for source, count in rows}

    out: list[TaskInfo] = []
    for name in list_tasks():
        cfg = load_task(name)
        out.append(
            TaskInfo(
                name=name,
                description=cfg.description,
                labels=list(cfg.labels),
                active=(name == active),
                seeded_count=counts.get(cfg.source, 0),
            )
        )
    return out
