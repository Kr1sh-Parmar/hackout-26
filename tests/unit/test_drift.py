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


def test_comparable_features_ignores_targets_and_identifiers():
    """PSI on a target measures the weather, not input drift; PSI on an id is
    noise with a number attached. The list is derived, so a new input column is
    monitored automatically and a new target is still ignored."""
    from src.evaluation.drift import comparable_features

    frame = pd.DataFrame(
        {
            "ghi_wm2": [1.0],
            "wind_speed_100m_ms": [1.0],
            "y_solar_cf": [1.0],
            "tso_solar_p50_mw": [1.0],
            "sample_weight_solar": [1.0],
            "demand_mw": [1.0],
            "lead_hours": [24],
            "region_id": ["BE"],
            "valid_ts_utc": pd.to_datetime(["2026-09-09"], utc=True),
        }
    )

    assert comparable_features(frame, frame) == ["ghi_wm2", "wind_speed_100m_ms"]


def test_health_reports_the_retrain_verdict(tmp_path, monkeypatch):
    """The monitor was fully built, fully tested and called by nothing. This is
    the wire: a cycle writes its verdict to `gold/drift`, /health repeats the
    reason the monitor already gave rather than re-deriving one."""
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import app
    from src.core.config import get_settings
    from src.evaluation.drift import DRIFT_COLUMNS

    run_ts = pd.Timestamp("2026-09-12T00:00:00Z")
    row = dict.fromkeys(DRIFT_COLUMNS, 0.0)
    row |= {"tech": "wind", "retrain": True, "reason": "error degraded 41% above baseline"}
    for table in ("forecast", "drift"):
        dest = tmp_path / "gold" / table / "region_id=BE" / f"run_date={run_ts:%Y-%m-%d}"
        dest.mkdir(parents=True)
        pd.DataFrame([{**row, "region_id": "BE", "run_ts_utc": run_ts}]).to_parquet(
            dest / "part-0.parquet", index=False
        )

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("REPLAY_MODE", "false")
    get_settings.cache_clear()
    deps.get_store.cache_clear()
    try:
        body = TestClient(app).get("/health").json()
        assert any("error degraded 41%" in w for w in body["warnings"]), body["warnings"]
        assert body["status"] == "degraded"
    finally:
        get_settings.cache_clear()
        deps.get_store.cache_clear()


def test_seasonal_window_compares_like_with_like():
    """A one-month current window against an all-season reference flags every
    seasonal variable. Restricting the reference to the same calendar period
    turns "it is September" back into "this September is unusual"."""
    from src.evaluation.drift import feature_drift, seasonal_window

    def temp(idx):
        """A seasonal swing the size of the weather noise around it -- nothing
        about this series has drifted, ever."""
        return 10 * np.sin(2 * np.pi * idx.dayofyear / 365.0) + rng.normal(0, 5, len(idx))

    ts = pd.date_range("2024-01-01", "2026-04-28", freq="h", tz="UTC")
    reference = pd.DataFrame({"valid_ts_utc": ts, "temp": temp(ts)})
    sep = pd.date_range("2026-09-01", "2026-09-30", freq="h", tz="UTC")
    current = pd.DataFrame({"valid_ts_utc": sep, "temp": temp(sep)})

    naive = feature_drift(reference, current, ["temp"])
    seasonal_ref = feature_drift(
        seasonal_window(reference, current["valid_ts_utc"]), current, ["temp"]
    )

    assert naive.loc[0, "drifted"], "sanity: the naive comparison is the false alarm"
    assert not seasonal_ref.loc[0, "drifted"]


def test_seasonal_window_wraps_around_new_year():
    """A January window must match late December, not exclude it."""
    from src.evaluation.drift import seasonal_window

    ts = pd.date_range("2024-01-01", "2025-12-31", freq="D", tz="UTC")
    reference = pd.DataFrame({"valid_ts_utc": ts})
    jan = pd.Series(pd.date_range("2026-01-01", "2026-01-10", freq="D", tz="UTC"))

    out = seasonal_window(reference, jan, half_width_days=21)
    months = set(pd.to_datetime(out["valid_ts_utc"]).dt.month)

    assert months == {12, 1}


def test_a_thin_sample_never_escalates_to_a_retrain():
    """One cycle contributes ~22 scored hours. That sample put recent RMSE 27%
    above a baseline measured over 31,091 rows -- sampling noise wearing a
    retrain recommendation. Report the number, refuse to act on it."""
    from src.evaluation.drift import MIN_SCORED_HOURS, error_drift, should_retrain

    thin = pd.Series(rng.normal(0, 0.20, 22))
    thick = pd.Series(rng.normal(0, 0.20, MIN_SCORED_HOURS))
    quiet = pd.DataFrame(columns=["feature", "psi", "drifted"])

    small = error_drift(thin, 0.10)
    assert small["relative_increase"] > 0.2, "the raw number is still reported"
    assert not small["degraded"]
    assert small["underpowered"]
    retrain, reason = should_retrain(quiet, small)
    assert not retrain
    assert "22 scored hours" in reason and "too few" in reason

    big = error_drift(thick, 0.10)
    assert big["degraded"] and not big["underpowered"]
    assert should_retrain(quiet, big)[0]
