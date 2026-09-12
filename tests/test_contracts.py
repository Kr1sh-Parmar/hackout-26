"""The three interface contracts, pinned.

These fail if anyone changes a signature or a column set without meaning to --
which is the whole point of fixing the contracts on day one. Four people build
against these shapes in parallel; a silent rename is an integration failure
discovered at 2 a.m.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.core.config import GridState, RegionConfig, load_region, site_master
from src.decisions.recommend import RECOMMEND_COLUMNS, Action, Flag, empty_actions, recommend
from src.features.build import (
    FEATURE_COLUMNS,
    FEATURE_INDEX,
    MIN_LAG_HOURS,
    build_features,
    empty_features,
)
from src.models.predict import PREDICT_COLUMNS, QUANTILES, empty_forecast, predict


# --------------------------------------------------------------------------- #
# signatures
# --------------------------------------------------------------------------- #
def test_build_features_signature():
    p = inspect.signature(build_features).parameters
    assert list(p) == ["weather", "site", "actuals"]
    assert p["actuals"].default is None, "serving path must work without actuals"


def test_predict_signature():
    p = inspect.signature(predict).parameters
    assert list(p) == ["region_id", "run_ts", "horizons", "quantiles"]
    assert p["horizons"].default == range(1, 73)
    assert p["quantiles"].default == QUANTILES


def test_recommend_signature():
    """The three positional inputs are the contract. Extra params are allowed
    only if optional, so a caller written against the contract keeps working."""
    p = inspect.signature(recommend).parameters
    names = list(p)
    assert names[:3] == ["outlook", "grid_state", "config"]
    for extra in names[3:]:
        assert p[extra].default is not inspect.Parameter.empty, (
            f"{extra} was added to recommend() without a default, breaking the contract"
        )


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def test_empty_frames_match_declared_columns():
    assert list(empty_features().columns) == FEATURE_COLUMNS
    assert list(empty_features().index.names) == FEATURE_INDEX
    assert list(empty_forecast().columns) == PREDICT_COLUMNS
    assert list(empty_actions().columns) == RECOMMEND_COLUMNS


def test_timestamps_are_utc_aware():
    """Every timestamp in src/ is UTC and tz-aware. Convert at the edge only."""
    f = empty_forecast()
    for c in ("run_ts_utc", "valid_ts_utc"):
        assert str(f[c].dtype) == "datetime64[ns, UTC]", c
    a = empty_actions()
    for c in ("valid_from", "valid_to"):
        assert str(a[c].dtype) == "datetime64[ns, UTC]", c
    assert all(str(lv.dtype) == "datetime64[ns, UTC]" for lv in empty_features().index.levels)


def test_quantiles_sorted_and_symmetric():
    assert list(QUANTILES) == sorted(QUANTILES)
    assert QUANTILES == (0.1, 0.5, 0.9)


def test_forecast_carries_provenance():
    """Without these, "which model produced this number?" is unanswerable --
    which is exactly the question asked after a bad forecast."""
    for c in ("model_version", "calibration_date", "calibrated"):
        assert c in PREDICT_COLUMNS


def test_actions_carry_a_number_and_a_confidence():
    for c in ("mwh", "value_inr", "confidence", "decisive", "rationale"):
        assert c in RECOMMEND_COLUMNS


def test_no_autoregressive_lag_under_24h():
    """At lead hour 48 `power_lag_1h` does not exist. A shorter lag scores
    brilliantly in a backtest and fails completely in deployment."""
    assert MIN_LAG_HOURS == 24
    for col in FEATURE_COLUMNS:
        if col.startswith("power_lag_"):
            hours = int(col.rsplit("_", 1)[1].removesuffix("h"))
            assert hours >= MIN_LAG_HOURS, f"{col} leaks information the live system lacks"


def test_enums_cover_the_documented_actions():
    assert {a.value for a in Action} == {
        "HOLD",
        "CURTAIL",
        "CHARGE_BESS",
        "DISCHARGE_BESS",
        "COMMIT_BACKUP",
    }
    assert "OVER_GENERATION" in {f.value for f in Flag}
    assert "LOW_CONFIDENCE" in {f.value for f in Flag}


# --------------------------------------------------------------------------- #
# region config
# --------------------------------------------------------------------------- #
def test_belgium_config_loads():
    cfg = load_region("BE")
    assert isinstance(cfg, RegionConfig)
    assert cfg.region_id == "BE"
    assert set(cfg.archetypes) == {"solar", "wind"}
    assert cfg.nwp_models == ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]


def test_archetype_shares_sum_to_one():
    """Validated at load, not debugged later as a 6% output deficit."""
    cfg = load_region("BE")
    for tech, arcs in cfg.archetypes.items():
        assert abs(sum(a.share for a in arcs) - 1.0) < 1e-6, tech


def test_grid_weights_sum_to_one():
    assert abs(sum(p.weight for p in load_region("BE").weather_grid) - 1.0) < 1e-6


def test_site_master_capacity_splits_the_fleet():
    cfg = load_region("BE")
    for tech in ("solar", "wind"):
        sites = site_master(cfg, tech)
        assert sites, tech
        assert abs(sum(s.capacity_mw for s in sites) - cfg.capacity_mw[tech]) < 1e-6


def test_unknown_region_raises_with_a_hint():
    with pytest.raises(FileNotFoundError, match="Configured regions"):
        load_region("ZZ")


# --------------------------------------------------------------------------- #
# A contract is honoured in exactly one of two ways: it is still a stub and says
# so loudly, or it is implemented and returns the declared shape. Silently
# returning something of the wrong shape is the failure this guards against.
#
# `predict` may also raise for a missing model artifact or an unknown region --
# those are documented behaviours of an implemented function, not stub-ness.
# --------------------------------------------------------------------------- #
def _grid_state() -> GridState:
    return GridState(
        region_id="BE",
        must_run_mw=2800,
        storage_soc_mwh=200,
        storage=load_region("BE").storage,
        ramp_limit_mw_per_h=400,
    )


@pytest.mark.parametrize(
    ("call", "columns", "allowed"),
    [
        (
            lambda: build_features(pd.DataFrame(), site_master(load_region("BE"), "solar")[0]),
            FEATURE_COLUMNS,
            (NotImplementedError,),
        ),
        (
            lambda: predict("BE", pd.Timestamp("2026-09-10", tz="UTC")),
            PREDICT_COLUMNS,
            (NotImplementedError, FileNotFoundError, RuntimeError),
        ),
        (
            lambda: recommend(pd.DataFrame(), _grid_state(), load_region("BE")),
            RECOMMEND_COLUMNS,
            (NotImplementedError,),
        ),
    ],
    ids=["build_features", "predict", "recommend"],
)
def test_contract_returns_declared_shape_or_says_it_is_a_stub(call, columns, allowed):
    try:
        out = call()
    except allowed:
        return  # still a stub, or a documented runtime precondition
    assert isinstance(out, pd.DataFrame), f"contract returned {type(out).__name__}, not a DataFrame"
    assert list(out.columns) == columns, "implemented contract drifted from its declared columns"
