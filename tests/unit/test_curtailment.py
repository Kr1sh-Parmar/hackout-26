"""Pins the PV curtailment heuristic: it only fires when output is flat,
sunny, AND well below the physics estimate -- and the sample-weight helper
zeroes exactly the rows it should (curtailed, bad qc, low availability).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.quality.curtailment import detect_curtailment, sample_weights


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2026-06-01", periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=idx)


def test_flags_flat_underperforming_daytime_output():
    n = 12
    capacity = 100.0
    physics = _series([80.0] * n)  # physics says plenty of sun
    actual = _series([10.0] * n)  # but output is flat and low -- classic curtailment
    kt = _series([0.9] * n)  # clear sky
    is_day = _series([1] * n)

    flagged = detect_curtailment(actual, physics, capacity, kt, is_day)

    assert flagged.dtype == np.int8
    # first ROLLING_WINDOW-1 rows can't have a rolling std yet -- everything after can
    assert flagged.iloc[3:].eq(1).all()


def test_does_not_flag_output_tracking_physics():
    n = 12
    capacity = 100.0
    physics = _series([80.0] * n)
    actual = _series([75.0] * n)  # tracking physics closely -- not curtailed
    kt = _series([0.9] * n)
    is_day = _series([1] * n)

    flagged = detect_curtailment(actual, physics, capacity, kt, is_day)

    assert flagged.eq(0).all()


def test_does_not_flag_cloudy_or_nighttime_low_output():
    n = 12
    capacity = 100.0
    physics = _series([5.0] * n)  # physics itself says low output expected
    actual = _series([1.0] * n)
    kt = _series([0.2] * n)  # overcast -- kt below the "good" threshold
    is_day = _series([1] * n)

    flagged = detect_curtailment(actual, physics, capacity, kt, is_day)

    assert flagged.eq(0).all()


def test_sample_weights_zero_curtailed_bad_qc_and_low_availability():
    df = pd.DataFrame(
        {
            "curtailed": [0, 1, 0, 0, 0],
            "qc_flag": ["OK", "OK", "MISSING", "OK", "OK"],
            "availability_pct": [100.0, 100.0, 100.0, 50.0, 100.0],
        }
    )

    w = sample_weights(df)

    assert list(w) == [1.0, 0.0, 0.0, 0.0, 1.0]
