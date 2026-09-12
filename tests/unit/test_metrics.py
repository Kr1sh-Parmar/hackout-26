"""Metric correctness. If the harness is wrong, every reported number is wrong."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    ace,
    crps_from_quantiles,
    interval_width,
    mbe,
    nmae,
    nrmse,
    picp,
    pinball,
    skill_score,
)
from src.evaluation.report import per_lead_hour_report, summarise


def test_perfect_forecast_scores_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse(y, y, 10.0) == 0.0
    assert nmae(y, y, 10.0) == 0.0
    assert mbe(y, y, 10.0) == 0.0


def test_nrmse_is_normalised_by_capacity():
    y = np.zeros(4)
    yhat = np.ones(4)
    assert nrmse(y, yhat, 10.0) == pytest.approx(0.1)
    assert nrmse(y, yhat, 100.0) == pytest.approx(0.01)


def test_mbe_is_signed():
    y = np.zeros(4)
    assert mbe(y, np.ones(4), 10.0) > 0, "over-forecasting must read positive"
    assert mbe(y, -np.ones(4), 10.0) < 0, "under-forecasting must read negative"


def test_skill_score_is_zero_when_matching_the_reference():
    y = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.5, 2.5, 3.5])
    assert skill_score(y, ref, ref, 10.0) == pytest.approx(0.0)


def test_skill_score_is_one_for_a_perfect_model():
    y = np.array([1.0, 2.0, 3.0])
    assert skill_score(y, y, np.array([9.0, 9.0, 9.0]), 10.0) == pytest.approx(1.0)


def test_skill_score_goes_negative_when_worse_than_reference():
    y = np.array([1.0, 2.0, 3.0])
    assert skill_score(y, np.array([9.0, 9.0, 9.0]), y + 0.1, 10.0) < 0


def test_pinball_penalises_asymmetrically():
    """At alpha=0.9 under-forecasting must cost more than over-forecasting."""
    y = np.array([10.0])
    under = pinball(y, np.array([8.0]), 0.9)
    over = pinball(y, np.array([12.0]), 0.9)
    assert under > over


def test_pinball_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20_000)
    true_q90 = np.quantile(y, 0.9)
    at = pinball(y, np.full_like(y, true_q90), 0.9)
    for off in (-0.5, 0.5):
        assert at < pinball(y, np.full_like(y, true_q90 + off), 0.9)


def test_picp_counts_inclusion():
    y = np.array([1.0, 5.0, 9.0])
    assert picp(y, np.zeros(3), np.full(3, 10.0)) == 1.0
    assert picp(y, np.full(3, 2.0), np.full(3, 6.0)) == pytest.approx(1 / 3)


def test_ace_is_coverage_minus_nominal():
    y = np.array([1.0, 5.0, 9.0])
    assert ace(y, np.zeros(3), np.full(3, 10.0), nominal=0.8) == pytest.approx(0.2)


def test_interval_width_is_a_capacity_fraction():
    assert interval_width(np.zeros(3), np.full(3, 5.0), 10.0) == pytest.approx(0.5)


def test_coverage_alone_is_gameable_so_width_is_reported():
    """A 0-to-capacity band covers everything; width is what exposes it."""
    y = np.array([1.0, 5.0, 9.0])
    lo, hi = np.zeros(3), np.full(3, 10.0)
    assert picp(y, lo, hi) == 1.0
    assert interval_width(lo, hi, 10.0) == pytest.approx(1.0)


def test_metrics_ignore_nan_pairs_rather_than_propagating():
    y = np.array([1.0, np.nan, 3.0])
    yhat = np.array([1.0, 2.0, 3.0])
    assert nrmse(y, yhat, 10.0) == 0.0


def test_crps_averages_the_quantile_losses():
    y = np.array([1.0, 2.0])
    qs = {0.1: np.array([0.5, 1.5]), 0.5: y, 0.9: np.array([1.5, 2.5])}
    assert crps_from_quantiles(y, qs) == pytest.approx(
        np.mean([pinball(y, v, a) for a, v in qs.items()])
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _preds(n_per_lead: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for lh in (24, 48, 72):
        y = rng.uniform(0, 1, n_per_lead)
        rows.append(
            pd.DataFrame(
                {
                    "lead_hours": lh,
                    "y_true": y,
                    "p50": y + rng.normal(0, 0.05 * lh / 24, n_per_lead),
                    "p10": y - 0.2,
                    "p90": y + 0.2,
                    "persistence": y + rng.normal(0, 0.25, n_per_lead),
                    "physics": y + rng.normal(0, 0.15, n_per_lead),
                    "is_day": 1,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_report_has_one_row_per_lead_hour():
    r = per_lead_hour_report(_preds(), capacity=1.0)
    assert list(r.lead_hours) == [24, 48, 72]
    assert (r.n_rows == 50).all()


def test_report_requires_is_day_when_daylight_only():
    p = _preds().drop(columns=["is_day"])
    with pytest.raises(KeyError, match="is_day"):
        per_lead_hour_report(p, capacity=1.0, daylight_only=True)


def test_report_error_grows_with_lead_hour():
    """Built that way in the fixture -- if it does not, the grouping is wrong."""
    r = per_lead_hour_report(_preds(), capacity=1.0)
    assert r.nrmse_model.is_monotonic_increasing


def test_report_tolerates_missing_optional_columns():
    p = _preds().drop(columns=["persistence", "physics", "p10", "p90"])
    r = per_lead_hour_report(p, capacity=1.0)
    assert r.nrmse_model.notna().all()
    assert r.nrmse_persistence.isna().all()


def test_summarise_weights_by_row_count():
    """Lead 72 has far fewer rows in reality; it must not dominate the headline."""
    r = pd.DataFrame(
        {
            "lead_hours": [24, 72],
            "n_rows": [1000, 10],
            "nrmse_model": [0.10, 0.90],
            "nrmse_persistence": [0.2, 0.2],
            "nrmse_physics": [np.nan, np.nan],
            "nrmse_tso": [np.nan, np.nan],
            "skill_vs_persistence": [0.5, 0.5],
            "mbe_model": [0.0, 0.0],
            "picp_80": [0.8, 0.8],
        }
    )
    s = summarise(r)
    assert s["nrmse_mean"] < 0.12, "sparse lead hour dominated the weighted mean"
