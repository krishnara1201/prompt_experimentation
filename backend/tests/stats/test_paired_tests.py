import pytest

from app.stats.errors import InsufficientDataError
from app.stats.paired_tests import PairedComparisonResult, compare_pair, correct_pairwise_pvalues, paired_diffs


def _cell(*values: float) -> list[float]:
    return list(values)


def test_paired_diffs_computes_repeat_averaged_diffs_for_shared_examples():
    repeats_by_cell = {
        (1, "a"): _cell(4.0, 6.0),  # mean 5.0
        (1, "b"): _cell(3.0),
        (2, "a"): _cell(2.0),
        (2, "b"): _cell(1.0),
        (3, "a"): _cell(9.0),  # arm b missing for example 3
    }
    diffs, n_excluded = paired_diffs(repeats_by_cell, "a", "b")
    assert diffs == [5.0 - 3.0, 2.0 - 1.0]
    assert n_excluded == 1


def test_compare_pair_raises_when_fewer_than_min_examples():
    repeats_by_cell = {
        (i, arm): _cell(float(i))
        for i in range(3)
        for arm in ("a", "b")
    }
    with pytest.raises(InsufficientDataError):
        compare_pair(repeats_by_cell, "a", "b", "latency_ms")


def test_compare_pair_computes_mean_diff_and_valid_ci():
    # arm "a" is always exactly 2.0 higher than arm "b" for every example.
    repeats_by_cell = {}
    for i in range(10):
        repeats_by_cell[(i, "a")] = _cell(float(i) + 2.0)
        repeats_by_cell[(i, "b")] = _cell(float(i))

    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=500, seed=42)
    assert isinstance(result, PairedComparisonResult)
    assert result.n_examples == 10
    assert result.n_excluded == 0
    assert result.mean_diff == pytest.approx(2.0)
    assert result.ci_lower <= result.mean_diff <= result.ci_upper
    assert result.p_value < 0.05


def test_compare_pair_excludes_examples_missing_from_one_arm():
    repeats_by_cell = {}
    for i in range(6):
        repeats_by_cell[(i, "a")] = _cell(1.0)
        if i != 5:
            repeats_by_cell[(i, "b")] = _cell(1.0)

    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=200, seed=1)
    assert result.n_examples == 5
    assert result.n_excluded == 1


def test_compare_pair_all_identical_diffs_returns_p_value_one_without_raising():
    repeats_by_cell = {(i, arm): _cell(1.0) for i in range(6) for arm in ("a", "b")}
    result = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=200, seed=1)
    assert result.p_value == 1.0


def test_compare_pair_reproducible_with_seed():
    repeats_by_cell = {}
    for i in range(8):
        repeats_by_cell[(i, "a")] = _cell(float(i) * 1.3, float(i) * 1.1)
        repeats_by_cell[(i, "b")] = _cell(float(i))

    first = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=300, seed=7)
    second = compare_pair(repeats_by_cell, "a", "b", "judge_score", bootstrap_samples=300, seed=7)
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper


def test_correct_pairwise_pvalues_applies_holm_bonferroni():
    results = [
        PairedComparisonResult("a", "b", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.01),
        PairedComparisonResult("a", "c", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.02),
        PairedComparisonResult("b", "c", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.03),
    ]
    corrected = correct_pairwise_pvalues(results)
    assert corrected is results
    assert [r.p_value_corrected for r in results] == pytest.approx([0.03, 0.04, 0.04])


def test_correct_pairwise_pvalues_is_noop_for_single_pair():
    results = [PairedComparisonResult("a", "b", "m", 10, 0, 0.1, 0.0, 0.2, 1.0, 0.03)]
    correct_pairwise_pvalues(results)
    assert results[0].p_value_corrected == 0.03
