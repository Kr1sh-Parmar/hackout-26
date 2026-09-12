"""Archetype aggregation is where a plausible-looking number goes silently wrong."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_region, site_master
from src.features.archetype import aggregate_archetypes
from src.features.build import build_all_features


@pytest.fixture(scope="module")
def wx() -> pd.DataFrame:
    df = pd.read_parquet("data/gold/training_base_24_72h/part-0.parquet")
    return df.head(240).reset_index(drop=True)


@pytest.mark.parametrize("tech", ["solar", "wind"])
def test_regional_capacity_is_the_regional_capacity(wx, tech):
    """A capacity-weighted MEAN of per-archetype capacity returns
    total * sum(share^2) -- 0.355 * total for Belgian solar. It looks plausible
    and it deflates every MW the platform reports."""
    cfg = load_region("BE")
    out = build_all_features("BE", wx, None, tech)
    assert out["capacity_mw"].iloc[0] == cfg.capacity_mw[tech]
    assert (out["capacity_mw"] == cfg.capacity_mw[tech]).all()


@pytest.mark.parametrize("tech", ["solar", "wind"])
def test_physics_mw_sums_to_at_most_the_fleet(wx, tech):
    out = build_all_features("BE", wx, None, tech)
    col = "physics_pac_mw" if tech == "solar" else "physics_power_mw"
    assert out[col].min() >= 0.0
    assert out[col].max() <= out["capacity_mw"].iloc[0] + 1e-6


def test_weighted_average_matches_a_hand_computed_value():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    a = pd.DataFrame({"power_curve_cf": [0.2, 0.8], "capacity_mw": [600.0, 600.0]}, index=idx)
    b = pd.DataFrame({"power_curve_cf": [0.6, 0.4], "capacity_mw": [400.0, 400.0]}, index=idx)

    out = aggregate_archetypes([a, b], [0.6, 0.4])
    # intensive -> weighted;  MW -> summed
    assert out["power_curve_cf"].to_list() == pytest.approx([0.36, 0.64])
    assert out["capacity_mw"].to_list() == pytest.approx([1000.0, 1000.0])


def test_wind_cutout_flag_is_a_fleet_fraction_not_archetype_zero():
    """Onshore cuts out at 25 m/s, offshore at 27 m/s from a taller hub. Copying
    archetype 0's flag claims the whole fleet trips when 62% of it does."""
    cfg = load_region("BE")
    sites = site_master(cfg, "wind")
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    wx = pd.DataFrame(
        {
            "wind_speed_100m_ms": [26.0],
            "wind_speed_10m_ms": [20.0],
            "wind_gusts_10m_ms": [32.0],
            "temperature_2m_c": [8.0],
            "surface_pressure_hpa": [990.0],
            "wind_direction_100m_deg": [250.0],
        },
        index=idx,
    )
    from src.features.wind import wind_features

    parts = [wind_features(wx, s) for s in sites]
    flags = [p["above_cutout_flag"].iloc[0] for p in parts]
    assert flags == [1.0, 0.0], "the two archetypes must disagree for this test to mean anything"

    out = aggregate_archetypes(parts, [s.capacity_share for s in sites])
    frac = out["above_cutout_flag"].iloc[0]
    assert 0.0 < frac < 1.0
    assert frac == pytest.approx(sites[0].capacity_share)


def test_solar_geometry_is_taken_not_averaged():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    a = pd.DataFrame({"solar_zenith_deg": [40.0, 50.0], "is_day": [1.0, 1.0]}, index=idx)
    b = pd.DataFrame({"solar_zenith_deg": [40.0, 50.0], "is_day": [1.0, 1.0]}, index=idx)
    out = aggregate_archetypes([a, b], [0.9, 0.1])
    assert np.array_equal(out["solar_zenith_deg"].to_numpy(), a["solar_zenith_deg"].to_numpy())
