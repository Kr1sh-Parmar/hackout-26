"""Pins the drift-monitoring contract: PSI actually detects a shifted
distribution and ignores an unshifted one, error_drift's threshold direction
is right, and should_retrain gives a reason alongside its verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.drift import error_drift, feature_drift, should_retrain

rng = np.random.default_rng(0)


def test_feature_drift_flags_a_shifted_distribution():
    reference = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    shifted = pd.DataFrame({"x": rng.normal(3, 1, 2000)})  # 3-sigma shift

    out = feature_drift(reference, shifted, ["x"])

    assert out.loc[0, "drifted"]
    assert out.loc[0, "psi"] > 0.2


def test_feature_drift_does_not_flag_the_same_distribution():
    reference = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"x": rng.normal(0, 1, 2000)})  # same distribution, new draw

    out = feature_drift(reference, current, ["x"])

    assert not out.loc[0, "drifted"]
    assert out.loc[0, "psi"] < 0.2


def test_error_drift_flags_materially_worse_recent_error():
    baseline_rmse = 0.10
    recent_errors = pd.Series(rng.normal(0, 0.20, 500))  # ~2x the baseline scale

    out = error_drift(recent_errors, baseline_rmse)

    assert out["degraded"]
    assert out["relative_increase"] > 0.20


def test_error_drift_does_not_flag_comparable_error():
    baseline_rmse = 0.10
    recent_errors = pd.Series(rng.normal(0, 0.10, 500))

    out = error_drift(recent_errors, baseline_rmse)

    assert not out["degraded"]


def test_should_retrain_gives_a_reason_when_true():
    drift_table = pd.DataFrame({"feature": ["ghi_wm2"], "psi": [0.5], "drifted": [True]})
    error_result = {"degraded": True, "relative_increase": 0.5}

    retrain, reason = should_retrain(drift_table, error_result)

    assert retrain is True
    assert isinstance(reason, str) and len(reason) > 0


def test_should_retrain_false_when_nothing_drifted():
    drift_table = pd.DataFrame({"feature": ["ghi_wm2"], "psi": [0.01], "drifted": [False]})
    error_result = {"degraded": False, "relative_increase": 0.02}

    retrain, reason = should_retrain(drift_table, error_result)

    assert retrain is False
    assert isinstance(reason, str) and len(reason) > 0


def test_capacity_growth_is_not_a_retrain_signal():
    """Belgium's fleet grew all window (cap_solar_mw PSI 10.7 same-season).

    The model trains on CAPACITY FACTOR precisely so growth is absorbed. Firing
    "retrain" on it is a false alarm, and a monitor that cries wolf gets ignored.
    """
    from src.evaluation.drift import should_retrain

    table = pd.DataFrame(
        {"feature": ["cap_solar_mw", "cap_wind_mw"], "psi": [10.7, 3.3], "drifted": [True, True]}
    )
    retrain, reason = should_retrain(table, {"degraded": False})
    assert retrain is False, reason
    assert "expected drift" in reason


def test_real_drift_alongside_expected_drift_still_fires():
    from src.evaluation.drift import should_retrain

    table = pd.DataFrame(
        {"feature": ["cap_solar_mw", "ghi_wm2"], "psi": [10.7, 0.9], "drifted": [True, True]}
    )
    retrain, reason = should_retrain(table, {"degraded": False})
    assert retrain is True
    assert "ghi_wm2" in reason and "cap_solar_mw" not in reason
