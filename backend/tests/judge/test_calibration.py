import pytest

from app.judge.calibration import calibration_report, cohens_kappa


def test_cohens_kappa_perfect_agreement():
    pairs = [(5, 5), (1, 1), (5, 5), (1, 1), (3, 2)]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_cohens_kappa_no_agreement():
    pairs = [(5, 1), (1, 5), (5, 1), (1, 5)]
    assert cohens_kappa(pairs) == pytest.approx(-1.0)


def test_cohens_kappa_requires_at_least_one_pair():
    with pytest.raises(ValueError):
        cohens_kappa([])


def test_calibration_report_computes_all_metrics():
    pairs = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
    report = calibration_report(pairs)

    assert report["n"] == 5
    assert report["spearman_r"] == pytest.approx(1.0)
    assert report["cohens_kappa"] == pytest.approx(1.0)
    assert report["mean_abs_diff"] == pytest.approx(0.0)


def test_calibration_report_requires_at_least_one_pair():
    with pytest.raises(ValueError):
        calibration_report([])
