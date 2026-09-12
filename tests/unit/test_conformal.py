"""Split conformal has to actually deliver the coverage it advertises."""

from __future__ import annotations

import numpy as np

from src.uncertainty.conformal import MIN_CALIBRATION_POINTS, SplitConformal
from src.uncertainty.coverage import coverage_report


def _synthetic(n_per_lead=4000, seed=0):
    rng = np.random.default_rng(seed)
    leads = np.array([24, 48, 72])
    lead = np.repeat(leads, n_per_lead)
    # Error grows with horizon -- which is the whole reason calibration is
    # per lead hour rather than one global width.
    y = rng.normal(0.0, 0.05 * lead / 24.0)
    lo = np.full(lead.size, -0.02)
    hi = np.full(lead.size, 0.02)
    return lo, hi, y, lead


def test_picp_lands_near_nominal_on_held_out_residuals():
    lo, hi, y, lead = _synthetic()
    cal = slice(None, None, 2)
    test = slice(1, None, 2)

    sc = SplitConformal(alpha=0.2).calibrate(lo[cal], hi[cal], y[cal], lead[cal])
    clo, chi = sc.apply(lo[test], hi[test], lead[test])
    rep = coverage_report(clo, chi, y[test], lead[test], nominal=0.8)

    assert (rep["picp"].between(0.77, 0.83)).all(), rep.to_dict("records")
    assert rep["ace"].abs().max() < 0.03


def test_q_hat_grows_with_lead_hour():
    """If the per-lead-hour split were doing nothing, these would be equal."""
    lo, hi, y, lead = _synthetic()
    sc = SplitConformal(alpha=0.2).calibrate(lo, hi, y, lead)
    q = [sc.q_hat[h] for h in (24, 48, 72)]
    assert q[0] < q[1] < q[2]
    assert q[2] > 2 * q[0]


def test_thin_lead_hours_get_no_correction():
    """Fewer than 30 points buys no finite-sample guarantee. Widening by a
    number we cannot justify is worse than not widening."""
    n = MIN_CALIBRATION_POINTS - 1
    lead = np.full(n, 71)
    sc = SplitConformal().calibrate(np.zeros(n), np.zeros(n), np.ones(n) * 5.0, lead)
    assert sc.q_hat[71] == 0.0
    lo, hi = sc.apply(np.zeros(n), np.zeros(n), lead)
    assert np.all(lo == 0.0) and np.all(hi == 0.0)


def test_unseen_lead_hour_is_not_silently_widened():
    sc = SplitConformal()
    lo, hi = sc.apply([1.0], [2.0], [999])
    assert (lo[0], hi[0]) == (1.0, 2.0)
