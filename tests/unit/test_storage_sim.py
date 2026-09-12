"""Pins the BESS simulation's physical invariants: SoC stays inside its
configured band, round-trip efficiency is applied exactly once (on charge),
and the battery only charges when there is over-generation (headroom < 0).
"""

from __future__ import annotations

import pandas as pd

from src.core.config import StorageConfig
from src.decisions.storage_sim import simulate


def _outlook(headrooms: list[float], ramps: list[float] | None = None) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=len(headrooms), freq="h", tz="UTC")
    ramps = ramps or [0.0] * len(headrooms)
    return pd.DataFrame(
        {
            "valid_ts_utc": ts,
            "net_load_mw": [0.0] * len(headrooms),  # unused directly by simulate
            "headroom_mw": headrooms,
            "ramp_mw_per_h": ramps,
        }
    )


def test_soc_never_leaves_configured_band():
    st = StorageConfig(
        energy_capacity_mwh=400,
        power_rating_mw=100,
        round_trip_efficiency=0.88,
        soc_min_frac=0.10,
        soc_max_frac=0.95,
    )
    # alternate deep over-generation and deep deficit to stress both ends
    headrooms = [-500, -500, -500, -500, -500, 500, 500, 500, 500, 500] * 3
    outlook = _outlook(headrooms)

    sim = simulate(outlook, st)

    lo, hi = st.soc_min_frac * st.energy_capacity_mwh, st.soc_max_frac * st.energy_capacity_mwh
    assert (sim["soc_mwh"] >= lo - 1e-9).all()
    assert (sim["soc_mwh"] <= hi + 1e-9).all()


def test_charging_only_when_headroom_negative():
    st = StorageConfig(energy_capacity_mwh=400, power_rating_mw=100)
    outlook = _outlook([-200, 0, 200, -50, 50])

    sim = simulate(outlook, st)

    over_gen = pd.Series([-200, 0, 200, -50, 50]) < 0
    assert (sim.loc[over_gen, "charge_mw"] > 0).all()
    assert (sim.loc[~over_gen, "charge_mw"] == 0).all()


def test_round_trip_efficiency_applied_once_on_charge():
    """Charging 100 MW for 1h at efficiency 0.5 must add exactly 50 MWh to
    SoC -- not 25 MWh (efficiency applied twice) and not 100 MWh (efficiency
    ignored)."""
    st = StorageConfig(
        energy_capacity_mwh=1000,
        power_rating_mw=100,
        round_trip_efficiency=0.5,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
    )
    outlook = _outlook([-100.0])  # one hour, 100 MW of over-generation

    sim = simulate(outlook, st)

    start_soc = 0.5 * st.energy_capacity_mwh  # simulate()'s documented starting point
    charged_mwh = sim.iloc[0]["charge_mw"] * 1.0  # 1h step
    assert charged_mwh == 100.0  # full over-generation absorbed (within power rating)
    expected_soc = start_soc + charged_mwh * st.round_trip_efficiency
    assert sim.iloc[0]["soc_mwh"] == expected_soc
