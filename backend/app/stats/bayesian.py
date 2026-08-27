from dataclasses import dataclass

import numpy as np
import pymc as pm

from app.stats.errors import MIN_PAIRED_EXAMPLES, InsufficientDataError


@dataclass
class EquivalenceResult:
    epsilon: float
    posterior_mean: float
    ci_lower: float
    ci_upper: float
    p_equivalent: float


def equivalence_probability(
    diffs: list[float],
    epsilon: float,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 1,
    random_seed: int | None = None,
) -> EquivalenceResult:
    if len(diffs) < MIN_PAIRED_EXAMPLES:
        raise InsufficientDataError(f"only {len(diffs)} paired examples; need at least {MIN_PAIRED_EXAMPLES}")

    diffs_arr = np.asarray(diffs, dtype=float)
    # Weakly-informative priors scaled off the data's own spread, not a
    # hardcoded per-metric range -- this is what keeps the function generic
    # across bounded (judge_score) and unbounded (latency_ms, cost) metrics.
    # The floor guards the degenerate all-identical-diffs case.
    scale = max(float(diffs_arr.std()), 1e-6)

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=10 * scale)
        sigma = pm.HalfNormal("sigma", sigma=10 * scale)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=diffs_arr)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=random_seed, progressbar=False)

    mu_draws = idata.posterior["mu"].values.reshape(-1)
    return EquivalenceResult(
        epsilon=epsilon,
        posterior_mean=float(mu_draws.mean()),
        ci_lower=float(np.percentile(mu_draws, 2.5)),
        ci_upper=float(np.percentile(mu_draws, 97.5)),
        p_equivalent=float(np.mean(mu_draws >= -epsilon)),
    )
