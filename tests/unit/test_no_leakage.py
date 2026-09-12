"""Leakage guards for the evaluation harness.

The skill score divides by the baseline's error, so a baseline that sees the future
makes the model look worse and one that is accidentally degenerate makes it look
better. Neither is visible in the output -- only these tests catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import climatology, leak_free_lag_hours, persistence
from src.evaluation.walk_forward import assert_no_overlap, walk_forward


def _frame(leads=(24, 36, 47, 48, 60, 71, 72)) -> pd.DataFrame:
    run = pd.Timestamp("2025-06-10 00:00", tz="UTC")
    return pd.DataFrame(
        {
            "run_ts_utc": run,
            "lead_hours": list(leads),
            "valid_ts_utc": [run + pd.Timedelta(hours=int(h)) for h in leads],
        }
    )


def _actuals(days: int = 30) -> pd.Series:
    idx = pd.date_range("2025-05-20", periods=days * 24, freq="h", tz="UTC")
    # a diurnal shape so the series is not degenerate
    cf = 0.4 + 0.35 * np.sin((idx.hour - 6) / 24 * 2 * np.pi)
    return pd.Series(np.clip(cf, 0, 1), index=idx, name="actual_cf")


# --------------------------------------------------------------------------- #
# the lag rule
# --------------------------------------------------------------------------- #
def test_lag_is_whole_days_and_covers_the_lead():
    """k = ceil(lead/24) whole days back, so the source always precedes the run."""
    assert list(leak_free_lag_hours(pd.Series([1, 23, 24, 25, 47, 48, 71, 72]))) == [
        24,
        24,
        24,
        48,
        48,
        48,
        72,
        72,
    ]


def test_persistence_never_uses_data_after_the_run():
    f = _frame()
    lag = leak_free_lag_hours(f.lead_hours)
    src = f.valid_ts_utc - pd.to_timedelta(lag, unit="h")
    assert (src <= f.run_ts_utc).all(), "a source observation postdates its own run time"


def test_persistence_raises_on_a_leaking_frame():
    """If someone hand-builds a frame with an impossible lead, fail loudly."""
    f = _frame()
    f.loc[0, "lead_hours"] = -5  # would pull a source after run_ts
    with pytest.raises(ValueError, match="later than their run_ts"):
        persistence(f, _actuals())


def test_persistence_returns_values_and_is_not_all_nan():
    out = persistence(_frame(), _actuals())
    assert out.notna().all(), "baseline produced NaN where history exists"
    assert out.between(0, 1).all()


def test_climatology_only_uses_history_before_the_run():
    """A trailing-window mean must not average in the window it is predicting."""
    f = _frame()
    a = _actuals()
    # poison everything at or after the run: if it leaks, the output moves
    poisoned = a.copy()
    poisoned[poisoned.index >= f.run_ts_utc.iloc[0]] = 99.0
    assert np.allclose(
        climatology(f, a).to_numpy(dtype=float),
        climatology(f, poisoned).to_numpy(dtype=float),
        equal_nan=True,
    ), "climatology changed when only post-run data changed -- it is leaking"


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def _long_frame(months: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-03-05", periods=months * 30 * 24, freq="h", tz="UTC")
    return pd.DataFrame({"valid_ts_utc": idx, "y": np.arange(len(idx), dtype=float)})


def test_walk_forward_folds_never_overlap():
    folds = walk_forward(_long_frame(), train_months=6, test_months=1, gap_days=1)
    assert folds, "no folds produced"
    assert_no_overlap(folds)


def test_walk_forward_gap_is_respected():
    folds = walk_forward(_long_frame(), train_months=6, test_months=1, gap_days=3)
    for f in folds:
        gap = f.test.valid_ts_utc.min() - f.train.valid_ts_utc.max()
        assert gap >= pd.Timedelta(days=2, hours=23), f"gap collapsed to {gap}"


def test_three_way_split_is_chronological():
    """Conformal calibration must sit strictly between training and test."""
    folds = walk_forward(_long_frame(), train_months=6, test_months=1, calibrate_days=30)
    assert folds
    for f in folds:
        assert not f.calibrate.empty, "three-way split produced an empty calibration window"
        assert f.train.valid_ts_utc.max() < f.calibrate.valid_ts_utc.min()
        assert f.calibrate.valid_ts_utc.max() < f.test.valid_ts_utc.min()
    assert_no_overlap(folds)


def test_training_window_expands():
    folds = walk_forward(_long_frame(), train_months=6, test_months=1)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0], "training window is not expanding"


def test_no_random_split_anywhere_in_src():
    """A random split must never be CALLED or IMPORTED in src/.

    Matches usage, not the bare name -- the docs deliberately mention
    `train_test_split` in prose to explain why it is banned, and a substring
    search would flag its own warning.
    """
    import pathlib
    import re

    pattern = re.compile(r"train_test_split\s*\(|import\s+train_test_split")
    hits = [p for p in pathlib.Path("src").rglob("*.py") if pattern.search(p.read_text("utf-8"))]
    assert not hits, f"random split used in {hits} -- adjacent hours leak"
