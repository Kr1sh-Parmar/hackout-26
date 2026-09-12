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
    # assert through the public surface: q_hat is keyed by BAND, not raw lead
    # hour, so indexing it directly couples the test to the banding scheme
    lo, hi = sc.apply(np.zeros(n), np.zeros(n), lead)
    assert np.all(lo == 0.0) and np.all(hi == 0.0)


def test_unseen_lead_hour_is_not_silently_widened():
    sc = SplitConformal()
    lo, hi = sc.apply([1.0], [2.0], [999])
    assert (lo[0], hi[0]) == (1.0, 2.0)


def test_lead_hours_are_pooled_into_bands():
    """Banding is a measured bias/variance choice, not an implementation detail.

    A 45-day window leaves ~30 usable points per individual lead hour, so a
    per-hour q_hat is noisy and some hours fall below the minimum and get no
    correction at all. 12-hour bands quadruple the evidence per estimate and
    measured better on every axis at once (PICP 0.793 vs 0.768, width 11.7% vs
    12.5%) over 22 walk-forward folds.
    """
    sc = SplitConformal(band_hours=12)
    assert list(sc._band([24, 30, 35, 36, 47, 48, 71, 72])) == [24, 24, 24, 36, 36, 48, 60, 72]


def test_adjacent_hours_in_a_band_share_a_correction():
    rng = np.random.default_rng(3)
    n = 400
    lead = rng.choice([24, 28, 33], n)  # all inside the 24-35 band
    err = rng.normal(0, 3.0, n)
    sc = SplitConformal(alpha=0.2, band_hours=12).calibrate(np.zeros(n), np.zeros(n), err, lead)
    lo, hi = sc.apply(np.zeros(3), np.zeros(3), [24, 28, 33])
    assert len(set(np.round(hi, 9))) == 1, "hours in one band must share q_hat"


def test_band_width_of_one_reproduces_per_lead_hour_behaviour():
    """The old behaviour stays reachable, so the choice can be revisited."""
    rng = np.random.default_rng(4)
    n = 600
    lead = rng.choice([24, 48], n)
    err = np.where(lead == 24, rng.normal(0, 1, n), rng.normal(0, 5, n))
    sc = SplitConformal(alpha=0.2, band_hours=1).calibrate(np.zeros(n), np.zeros(n), err, lead)
    _, hi = sc.apply(np.zeros(2), np.zeros(2), [24, 48])
    assert hi[1] > hi[0], "with band_hours=1 each lead hour keeps its own width"


def test_scaled_correction_tracks_condition_dependent_width():
    """The correction must scale WITH the model's own band, not add a constant.

    Aggregate coverage is not conditional coverage. An additive q_hat adds the
    same MW at dawn and at midday; solar error does not work that way.
    """
    rng = np.random.default_rng(11)
    n = 800
    lead = np.full(n, 30)
    width = rng.choice([10.0, 100.0], n)  # two regimes, 10x apart
    lo, hi = -width / 2, width / 2
    y = rng.normal(0, 1, n) * width / 3  # error scales with the regime
    sc = SplitConformal(alpha=0.2, scaled=True).calibrate(lo, hi, y, lead)

    out_lo, out_hi = sc.apply(np.array([-5.0, -50.0]), np.array([5.0, 50.0]), [30, 30])
    narrow = out_hi[0] - out_lo[0]
    wide = out_hi[1] - out_lo[1]
    assert wide > narrow * 5, "a scaled correction must widen the wide band far more"


def test_additive_mode_is_still_available():
    rng = np.random.default_rng(12)
    n = 400
    lead = np.full(n, 30)
    sc = SplitConformal(alpha=0.2, scaled=False).calibrate(
        np.zeros(n), np.zeros(n), rng.normal(0, 4, n), lead
    )
    lo, hi = sc.apply(np.array([0.0, 100.0]), np.array([0.0, 100.0]), [30, 30])
    assert np.isclose((hi[0] - lo[0]), (hi[1] - lo[1])), "additive adds a constant width"


def test_artifact_from_an_older_version_still_applies():
    """A model artifact outlives the code that wrote it.

    Reconstructs a pickle predating band_hours/scaled: it must keep working with
    its original semantics, not die with AttributeError in the serving path.
    """
    import pickle

    sc = SplitConformal()
    sc.calibrate(
        np.zeros(200), np.zeros(200), np.random.default_rng(9).normal(0, 3, 200), np.full(200, 30)
    )
    blob = pickle.loads(pickle.dumps(sc))
    del blob.__dict__["band_hours"], blob.__dict__["scaled"]
    revived = pickle.loads(pickle.dumps(blob))

    lo, hi = revived.apply([0.0], [1.0], [30])  # must not raise
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert revived.band_hours == 1 and revived.scaled is False
