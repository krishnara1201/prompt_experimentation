from dataclasses import dataclass

from sqlalchemy import func
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
