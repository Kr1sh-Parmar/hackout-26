"""Training and serving must compute features identically.

The single most common production failure in a forecasting system is not a bad
model -- it is a feature that is computed one way during training and a slightly
different way at serving time. A different fill rule, a different timezone, a lag
from a different origin. The model is then asked a question it was never trained
on, and it answers confidently.

`build_features` is one pure function used by both paths, and this test is what
holds that in place. If it fails, do not relax it: find the impurity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_region, site_master
from src.features.build import FEATURE_COLUMNS, build_features
from src.features.lags import LAG_COLUMNS, lag_origin
from tests.conftest import read_built_parquet

GOLD = "data/gold/training_base_24_72h/part-0.parquet"
NON_LAG = [c for c in FEATURE_COLUMNS if c not in LAG_COLUMNS and c != "tech"]


@pytest.fixture(scope="module")
def slice_():
    df = read_built_parquet(GOLD)
    return df[df["forecast_vintage"] == "prev_day1"].head(480).reset_index(drop=True)


@pytest.fixture(scope="module")
def actuals():
    return read_built_parquet("data/silver/generation_actuals_solar/part-0.parquet")


@pytest.fixture(scope="module")
def wind_actuals():
    return read_built_parquet("data/silver/generation_actuals_wind/part-0.parquet")


@pytest.mark.parametrize("tech", ["solar", "wind"])
def test_training_and_serving_paths_agree_on_every_non_lag_column(
    slice_, actuals, wind_actuals, tech
):
    """The serving path has no actuals. Everything that is not a lag must come
    out bit-for-bit the same anyway."""
    site = site_master(load_region("BE"), tech)[0]
    hist = actuals if tech == "solar" else wind_actuals

    trained = build_features(slice_, site, actuals=hist)
    served = build_features(slice_, site, actuals=None)

    assert list(trained.columns) == FEATURE_COLUMNS
    assert trained.index.equals(served.index)
    pd.testing.assert_frame_equal(
        trained[NON_LAG], served[NON_LAG], rtol=1e-9, atol=0.0, check_dtype=True
    )


@pytest.mark.parametrize("tech", ["solar", "wind"])
def test_only_the_lag_family_differs(slice_, actuals, wind_actuals, tech):
    site = site_master(load_region("BE"), tech)[0]
    hist = actuals if tech == "solar" else wind_actuals
    trained = build_features(slice_, site, actuals=hist)
    served = build_features(slice_, site, actuals=None)

    assert served[LAG_COLUMNS].isna().all().all(), "serving lags must be NaN, never invented"
    assert trained[LAG_COLUMNS].notna().any().any(), "fixture has no usable history"


def test_build_features_does_not_mutate_its_input(slice_, actuals):
    site = site_master(load_region("BE"), "solar")[0]
    before = slice_.copy(deep=True)
    build_features(slice_, site, actuals=actuals)
    pd.testing.assert_frame_equal(slice_, before)


def test_build_features_is_deterministic(slice_, actuals):
    site = site_master(load_region("BE"), "solar")[0]
    a = build_features(slice_, site, actuals=actuals)
    b = build_features(slice_, site, actuals=actuals)
    pd.testing.assert_frame_equal(a, b)


def test_row_order_does_not_change_the_answer(slice_, actuals):
    """A feature that depends on row order is a feature that changes when the
    serving query comes back sorted differently."""
    site = site_master(load_region("BE"), "solar")[0]
    shuffled = slice_.sample(frac=1.0, random_state=7)
    a = build_features(slice_, site, actuals=actuals)
    b = build_features(shuffled, site, actuals=actuals).loc[a.index]
    order_free = [c for c in NON_LAG if c not in ("nwp_ghi_ramp", "nwp_ws_ramp")]
    pd.testing.assert_frame_equal(a[order_free], b[order_free], rtol=1e-9)


def test_no_lag_originates_inside_the_forecast_horizon(slice_):
    """The real leak test: every lag origin must be at or before the run time."""
    origin = lag_origin(pd.DatetimeIndex(slice_["valid_ts_utc"]), slice_["lead_hours"])
    run_ts = pd.DatetimeIndex(slice_["run_ts_utc"])
    assert (origin <= run_ts).all(), "a lag reaches past the run it belongs to"
    gap = (pd.DatetimeIndex(slice_["valid_ts_utc"]) - origin) / np.timedelta64(1, "h")
    assert gap.min() >= 24.0
