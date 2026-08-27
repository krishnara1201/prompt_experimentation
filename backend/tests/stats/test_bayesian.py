import pytest

from app.stats.bayesian import EquivalenceResult, equivalence_probability
from app.stats.errors import InsufficientDataError

SAMPLE_KWARGS = dict(draws=200, tune=200, chains=2, cores=1, random_seed=0)


def test_equivalence_probability_raises_when_fewer_than_min_examples():
    with pytest.raises(InsufficientDataError):
        equivalence_probability([0.1, 0.2, 0.3], epsilon=0.5, **SAMPLE_KWARGS)


def test_equivalence_probability_near_one_when_arms_effectively_identical():
    diffs = [0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.02, -0.02, 0.0]
    result = equivalence_probability(diffs, epsilon=0.5, **SAMPLE_KWARGS)
    assert isinstance(result, EquivalenceResult)
    assert result.p_equivalent > 0.9


def test_equivalence_probability_near_zero_when_local_much_worse():
    diffs = [-5.0, -4.8, -5.2, -4.9, -5.1, -5.0, -4.7, -5.3, -5.0, -4.9]
    result = equivalence_probability(diffs, epsilon=0.5, **SAMPLE_KWARGS)
    assert result.p_equivalent < 0.1


def test_equivalence_probability_result_fields_are_internally_consistent():
    diffs = [-1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.3, -0.4, 0.6, -0.1]
    result = equivalence_probability(diffs, epsilon=1.0, **SAMPLE_KWARGS)
    assert result.epsilon == 1.0
    assert result.ci_lower <= result.posterior_mean <= result.ci_upper
    assert 0.0 <= result.p_equivalent <= 1.0
