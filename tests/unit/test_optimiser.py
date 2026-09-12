"""Pins the LP dispatch optimiser's physical invariants: SoC stays inside its
configured band, round-trip efficiency lands on the charge leg only, and the
LP absorbs over-generation into storage before it resorts to curtailment
when the battery has room.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import Prices, StorageConfig
from src.decisions.optimiser import optimise_dispatch


def _outlook(headrooms: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=len(headrooms), freq="h", tz="UTC")
    return pd.DataFrame({"valid_ts_utc": ts, "headroom_mw": headrooms})


def test_soc_never_leaves_configured_band():
    st = StorageConfig(
        energy_capacity_mwh=400,
        power_rating_mw=100,
        round_trip_efficiency=0.88,
        soc_min_frac=0.10,
        soc_max_frac=0.95,
    )
    headrooms = [-500, -500, -500, 500, 500, 500] * 3
    result = optimise_dispatch(_outlook(headrooms), st, Prices())

    lo, hi = st.soc_min_frac * st.energy_capacity_mwh, st.soc_max_frac * st.energy_capacity_mwh
    assert (result["soc_mwh"] >= lo - 1e-6).all()
    assert (result["soc_mwh"] <= hi + 1e-6).all()


def test_prefers_charging_over_curtailing_when_storage_has_room():
    """One hour of modest over-generation, well within power/SoC headroom --
    the LP should absorb essentially all of it rather than curtail, since
    curtailment costs far more than a battery cycle."""
    st = StorageConfig(
        energy_capacity_mwh=400,
        power_rating_mw=100,
        round_trip_efficiency=0.88,
        soc_min_frac=0.10,
        soc_max_frac=0.95,
    )
    result = optimise_dispatch(_outlook([-50.0]), st, Prices())

    assert result.iloc[0]["charge_mw"] > 49.0
    assert result.iloc[0]["curtail_mw"] < 1.0


def test_efficiency_applied_once_on_charge_leg():
    st = StorageConfig(
        energy_capacity_mwh=1000,
        power_rating_mw=100,
        round_trip_efficiency=0.5,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
    )
    result = optimise_dispatch(_outlook([-100.0]), st, Prices())

    start_soc = 0.5 * st.energy_capacity_mwh
    charged_mwh = result.iloc[0]["charge_mw"] * 1.0
    assert charged_mwh == 100.0
    expected_soc = start_soc + charged_mwh * st.round_trip_efficiency
    assert abs(result.iloc[0]["soc_mwh"] - expected_soc) < 1e-6


def test_horizon_hours_truncates_the_outlook():
    st = StorageConfig()
    result = optimise_dispatch(_outlook([0.0] * 100), st, Prices(), horizon_hours=10)

    assert len(result) == 10
