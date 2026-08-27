from dataclasses import dataclass
import random

from scipy import stats as scipy_stats

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class PairedComparisonResult:
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
    p_value_corrected: float | None = None


def _eligible_examples(
    repeats_by_cell: dict[tuple[int, str], list[float]], arm_a: str, arm_b: str
) -> tuple[list[int], int]:
    example_ids = sorted({example_id for (example_id, arm_name) in repeats_by_cell if arm_name in (arm_a, arm_b)})
    eligible = [
        example_id
        for example_id in example_ids
        if (example_id, arm_a) in repeats_by_cell and (example_id, arm_b) in repeats_by_cell
    ]
    return eligible, len(example_ids) - len(eligible)


def _repeat_mean(repeats_by_cell: dict[tuple[int, str], list[float]], example_id: int, arm: str) -> float:
    values = repeats_by_cell[(example_id, arm)]
    return sum(values) / len(values)


def paired_diffs(
    repeats_by_cell: dict[tuple[int, str], list[float]], arm_a: str, arm_b: str
) -> tuple[list[float], int]:
    eligible, n_excluded = _eligible_examples(repeats_by_cell, arm_a, arm_b)
    diffs = [
        _repeat_mean(repeats_by_cell, example_id, arm_a) - _repeat_mean(repeats_by_cell, example_id, arm_b)
        for example_id in eligible
    ]
    return diffs, n_excluded


def compare_pair(
    repeats_by_cell: dict[tuple[int, str], list[float]],
    arm_a: str,
    arm_b: str,
    metric: str,
    bootstrap_samples: int = 10_000,
    seed: int | None = None,
) -> PairedComparisonResult:
    eligible, n_excluded = _eligible_examples(repeats_by_cell, arm_a, arm_b)
    if len(eligible) < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(
            f"only {len(eligible)} paired examples for {arm_a!r} vs {arm_b!r} on {metric!r}; "
            f"need at least {MIN_PAIRED_EXAMPLES}"
        )

    diffs = [
        _repeat_mean(repeats_by_cell, example_id, arm_a) - _repeat_mean(repeats_by_cell, example_id, arm_b)
        for example_id in eligible
    ]
    n = len(eligible)
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    replicate_means = []
    for _ in range(bootstrap_samples):
        total = 0.0
        for _ in range(n):
            example_id = eligible[rng.randrange(n)]
            a_repeats = repeats_by_cell[(example_id, arm_a)]
            b_repeats = repeats_by_cell[(example_id, arm_b)]
            total += a_repeats[rng.randrange(len(a_repeats))] - b_repeats[rng.randrange(len(b_repeats))]
        replicate_means.append(total / n)
    replicate_means.sort()
    ci_lower = replicate_means[int(0.025 * bootstrap_samples)]
    ci_upper = replicate_means[min(int(0.975 * bootstrap_samples), bootstrap_samples - 1)]

    if all(d == 0.0 for d in diffs):
        # scipy.stats.wilcoxon raises when every paired difference is exactly
        # zero -- e.g. two arms that always agree. Report the (correct) null
        # result directly instead of catching the exception.
        wilcoxon_statistic, p_value = 0.0, 1.0
    else:
        wilcoxon_statistic, p_value = scipy_stats.wilcoxon(diffs)

    return PairedComparisonResult(
        arm_a=arm_a,
        arm_b=arm_b,
        metric=metric,
        n_examples=n,
        n_excluded=n_excluded,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        wilcoxon_statistic=float(wilcoxon_statistic),
        p_value=float(p_value),
    )


def correct_pairwise_pvalues(results: list[PairedComparisonResult]) -> list[PairedComparisonResult]:
    m = len(results)
    if m <= 1:
        for r in results:
            r.p_value_corrected = r.p_value
        return results

    order = sorted(range(m), key=lambda i: results[i].p_value)
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * results[idx].p_value
        running_max = max(running_max, adjusted)
        corrected[idx] = min(running_max, 1.0)
    for i, r in enumerate(results):
        r.p_value_corrected = corrected[i]
    return results
