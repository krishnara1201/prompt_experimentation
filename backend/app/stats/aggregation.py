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
