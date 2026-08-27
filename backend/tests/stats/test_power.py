import math
import statistics

import pytest
from scipy.stats import norm

from app.stats.errors import InsufficientDataError
from app.stats.power import PowerResult, estimate_sample_size


def test_estimate_sample_size_raises_when_fewer_than_min_pilot_examples():
    with pytest.raises(InsufficientDataError):
        estimate_sample_size([1.0, 2.0, 3.0])


def test_estimate_sample_size_matches_closed_form_formula():
    pilot_diffs = [-1.0, 0.0, 1.0, 2.0, 3.0]
    result = estimate_sample_size(pilot_diffs, effect_size=1.0, power=0.8, alpha=0.05)

    mean_diff = statistics.mean(pilot_diffs)
    std_diff = statistics.stdev(pilot_diffs)
    z_alpha = norm.ppf(1 - 0.05 / 2)
    z_power = norm.ppf(0.8)
    expected_required_n = math.ceil(((z_alpha + z_power) ** 2 * std_diff**2) / 1.0**2)

    assert isinstance(result, PowerResult)
    assert result.pilot_n == 5
    assert result.pilot_mean_diff == pytest.approx(mean_diff)
    assert result.pilot_std_diff == pytest.approx(std_diff)
    assert result.required_n == expected_required_n
    assert 0.0 <= result.achieved_power <= 1.0


def test_estimate_sample_size_defaults_effect_size_to_pilot_mean_diff():
    pilot_diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = estimate_sample_size(pilot_diffs)
    assert result.effect_size == pytest.approx(statistics.mean(pilot_diffs))


def test_estimate_sample_size_raises_on_zero_effect_size():
    pilot_diffs = [-2.0, -1.0, 0.0, 1.0, 2.0]  # mean 0.0
    with pytest.raises(ValueError):
        estimate_sample_size(pilot_diffs)


def test_estimate_sample_size_smaller_effect_requires_larger_n():
    pilot_diffs = [-1.0, 0.0, 1.0, 2.0, 3.0]
    small_effect = estimate_sample_size(pilot_diffs, effect_size=0.5)
    large_effect = estimate_sample_size(pilot_diffs, effect_size=2.0)
    assert small_effect.required_n > large_effect.required_n
