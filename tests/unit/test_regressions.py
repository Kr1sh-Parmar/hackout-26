"""Bugs that shipped once. Each test fails if its fix is reverted.

All of these produced plausible-looking output, which is why they survived until
something downstream contradicted them. A test is the only thing that keeps a
silent bug fixed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.core.config import GridState, load_region
from src.decisions.recommend import Action, recommend


# --------------------------------------------------------------------------- #
# NaN is not JSON. `ramp_mw_per_h` is a first difference and is legitimately
# absent on the first row of a horizon; the API used to 500 on it with
# "Out of range float values are not JSON compliant".
# --------------------------------------------------------------------------- #
def test_records_encodes_absent_values_as_null_not_nan():
    from src.api.routes._common import records

    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
    out = records(df, ["a", "b"])
    assert out[1]["a"] is None, "NaN must become None, not stay NaN"
    json.dumps(out)  # would raise ValueError on a NaN


def test_every_api_route_uses_the_nan_safe_encoder():
    """A new route that calls .to_dict("records") directly reintroduces the 500."""
    import pathlib

    offenders = [
        p.name
        for p in pathlib.Path("src/api/routes").glob("*.py")
        if p.name != "_common.py" and 'to_dict("records")' in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{offenders} bypass records() and will 500 on NaN"


# --------------------------------------------------------------------------- #
# DuckDB renders TIMESTAMPTZ in the SESSION timezone. On a machine set to IST
# every served timestamp came back as +05:30 -- the same instant, but the API is
# not the presentation layer and must speak UTC regardless of where it runs.
# --------------------------------------------------------------------------- #
def test_store_session_timezone_is_utc(tmp_path):
    from src.core.store import ParquetStore

    store = ParquetStore(tmp_path)
    tz = store.con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
    assert tz == "UTC", f"DuckDB session tz is {tz!r}; timestamps will not serialise as UTC"


def test_store_reads_timestamps_back_as_utc(tmp_path):
    from src.core.store import ParquetStore

    dest = tmp_path / "gold" / "forecast" / "region_id=BE" / "run_date=2026-09-10"
    dest.mkdir(parents=True)
    ts = pd.Timestamp("2026-09-10 00:00", tz="UTC")
    pd.DataFrame(
        {
            "region_id": ["BE"],
            "run_ts_utc": [ts],
            "valid_ts_utc": [ts + pd.Timedelta(hours=24)],
            "lead_hours": [24],
            "tech": ["solar"],
            "p10_mw": [1.0],
            "p50_mw": [2.0],
            "p90_mw": [3.0],
            "capacity_mw": [100.0],
        }
    ).to_parquet(dest / "part-0.parquet", index=False)

    out = ParquetStore(tmp_path).read_forecast("BE")
    assert not out.empty
    got = pd.Timestamp(out["valid_ts_utc"].iloc[0])
    assert got.utcoffset() == pd.Timedelta(0), f"expected UTC, got offset {got.utcoffset()}"


# --------------------------------------------------------------------------- #
# An action the asset cannot execute is worse than no action: it puts a number
# on screen that an operator would act on. Observed: DISCHARGE_BESS 10,296 MWh
# at 1,450 MW from a 400 MWh / 100 MW battery -- 25x its entire capacity.
# --------------------------------------------------------------------------- #
def _outlook_with_a_huge_ramp() -> pd.DataFrame:
    """A ramp far larger than any battery could follow."""
    idx = pd.date_range("2026-09-10", periods=12, freq="h", tz="UTC")
    net = np.linspace(3000, 20000, 12)  # ~1,500 MW/h ramp
    return pd.DataFrame(
        {
            "valid_ts_utc": idx,
            "lead_hours": range(24, 36),
            "demand_mw": net + 2000,
            "solar_p10_mw": 0.0,
            "solar_p50_mw": 0.0,
            "solar_p90_mw": 0.0,
            "wind_p10_mw": 0.0,
            "wind_p50_mw": 0.0,
            "wind_p90_mw": 0.0,
            "net_load_mw": net,
            "net_load_p10_mw": net - 500,
            "net_load_p90_mw": net + 500,
            "must_run_mw": 2800.0,
            "headroom_mw": net - 2800.0,
            "ramp_mw_per_h": np.r_[np.nan, np.diff(net)],
        }
    )


@pytest.fixture
def grid_state() -> GridState:
    cfg = load_region("BE")
    return GridState(
        region_id="BE",
        must_run_mw=cfg.decision_thresholds.must_run_mw,
        storage_soc_mwh=cfg.storage.energy_capacity_mwh * 0.5,
        storage=cfg.storage,
        ramp_limit_mw_per_h=cfg.decision_thresholds.ramp_limit_mw_per_h,
    )


def test_storage_actions_never_exceed_the_asset(grid_state):
    cfg = load_region("BE")
    st = cfg.storage
    actions = recommend(_outlook_with_a_huge_ramp(), grid_state, cfg)
    storage_actions = actions[
        actions.action.isin([Action.CHARGE_BESS.value, Action.DISCHARGE_BESS.value])
    ]
    assert len(storage_actions), "fixture should trigger at least one storage action"

    for _, a in storage_actions.iterrows():
        assert a.power_mw <= st.power_rating_mw + 1e-6, (
            f"{a.action} asks {a.power_mw:.0f} MW of a {st.power_rating_mw:.0f} MW asset"
        )
        assert a.mwh <= st.energy_capacity_mwh + 1e-6, (
            f"{a.action} asks {a.mwh:.0f} MWh of a {st.energy_capacity_mwh:.0f} MWh asset"
        )


def test_shortfall_is_stated_rather_than_hidden(grid_state):
    """Clipping silently would under-report the problem. Say what is uncovered."""
    cfg = load_region("BE")
    actions = recommend(_outlook_with_a_huge_ramp(), grid_state, cfg)
    clipped = actions[actions.rationale.str.contains("needs another instrument", na=False)]
    assert len(clipped), "a ramp beyond the battery must name the uncovered remainder"


def test_empty_outlook_returns_the_contract_shape_not_an_error(grid_state):
    """No forecast yet is a legitimate state, not a malformed call."""
    from src.decisions.recommend import RECOMMEND_COLUMNS

    out = recommend(pd.DataFrame(), grid_state, load_region("BE"))
    assert list(out.columns) == RECOMMEND_COLUMNS
    assert out.empty


def test_malformed_outlook_raises_a_specific_error(grid_state):
    """Wrong shape IS an error -- and not NotImplementedError, which would claim
    the function was never built."""
    bad = pd.DataFrame({"valid_ts_utc": pd.date_range("2026-09-10", periods=3, tz="UTC")})
    with pytest.raises(ValueError, match="missing"):
        recommend(bad, grid_state, load_region("BE"))
