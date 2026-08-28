from itertools import combinations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.models import Run
from app.db.session import get_session
from app.stats.aggregation import ALLOWED_METRICS, load_metric_by_example, summarize_arms
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
    n_excluded: int
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
    n_excluded: int


class ArmSummaryResponse(BaseModel):
    arm_name: str
    n: int
    mean_judge_score: float | None
    mean_latency_ms: float | None
    mean_cost_estimate_usd: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None


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
    bootstrap_samples: int = Query(default=10_000, gt=0, le=200_000),
    session: AsyncSession = Depends(get_session),
):
    _validate_metric(metric)
    run = await _load_run_or_404(run_id, session)
    pairs = _resolve_pairs(run, arm_a, arm_b)

    repeats_by_cell = await load_metric_by_example(session, run_id, metric, run.arm_names)

    def _run_compare_pair(a: str, b: str):
        return compare_pair(repeats_by_cell, a, b, metric, bootstrap_samples=bootstrap_samples)

    try:
        # compare_pair runs a pure-Python bootstrap loop that can take
        # seconds at high bootstrap_samples -- it must not run on the event
        # loop, same rationale as run_in_threadpool usage in runs.py.
        results = [await run_in_threadpool(_run_compare_pair, a, b) for a, b in pairs]
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    correct_pairwise_pvalues(results)
    return [PairedComparisonResponse(**vars(r)) for r in results]


@router.get("/{run_id}/summary", response_model=list[ArmSummaryResponse])
async def run_summary(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await _load_run_or_404(run_id, session)
    summaries = await summarize_arms(session, run_id, run.arm_names)
    return [ArmSummaryResponse(**vars(s)) for s in summaries]


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
    if metric != "judge_score":
        raise HTTPException(
            status_code=422,
            detail="equivalence is only supported for metric='judge_score' (direction-sensitive for other metrics)",
        )
    run = await _load_run_or_404(run_id, session)
    _validate_arms(run, arm_local, arm_api)

    repeats_by_cell = await load_metric_by_example(session, run_id, metric, [arm_local, arm_api])
    diffs, n_excluded = paired_diffs(repeats_by_cell, arm_local, arm_api)

    try:
        # equivalence_probability runs PyMC MCMC sampling (~5+ seconds even
        # at reduced draws) -- it must not run on the event loop, same
        # rationale as run_in_threadpool usage in runs.py.
        result = await run_in_threadpool(equivalence_probability, diffs, epsilon)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return EquivalenceResponse(
        arm_local=arm_local,
        arm_api=arm_api,
        metric=metric,
        epsilon=epsilon,
        n_examples=len(diffs),
        n_excluded=n_excluded,
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
    diffs, n_excluded = paired_diffs(repeats_by_cell, arm_a, arm_b)

    try:
        result = estimate_sample_size(diffs, effect_size=effect_size, power=power, alpha=alpha)
    except ValueError as exc:
        # Covers both InsufficientDataError (too few pilot examples) and the
        # plain ValueError estimate_sample_size raises for a zero effect
        # size -- InsufficientDataError is a ValueError subclass, so this
        # single except handles both without redundancy.
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
        n_excluded=n_excluded,
    )
