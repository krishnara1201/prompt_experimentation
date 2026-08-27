from dataclasses import dataclass
import math

from scipy.stats import norm

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class PowerResult:
    pilot_n: int
    pilot_mean_diff: float
    pilot_std_diff: float
    effect_size: float
    alpha: float
    target_power: float
    required_n: int
    achieved_power: float


def estimate_sample_size(
    pilot_diffs: list[float],
    effect_size: float | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> PowerResult:
    n = len(pilot_diffs)
    if n < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(f"only {n} pilot paired examples; need at least {MIN_PAIRED_EXAMPLES}")

    mean_diff = sum(pilot_diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in pilot_diffs) / (n - 1)
    std_diff = math.sqrt(variance) or 1e-6

    delta = effect_size if effect_size is not None else mean_diff
    if delta == 0:
        raise ValueError("effect_size cannot be zero -- no detectable difference to power for")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    required_n = math.ceil(((z_alpha + z_power) ** 2 * std_diff**2) / delta**2)

    z_achieved = abs(delta) * math.sqrt(n) / std_diff - z_alpha
    achieved_power = float(norm.cdf(z_achieved))

    return PowerResult(
        pilot_n=n,
        pilot_mean_diff=mean_diff,
        pilot_std_diff=std_diff,
        effect_size=delta,
        alpha=alpha,
        target_power=power,
        required_n=required_n,
        achieved_power=achieved_power,
    )
