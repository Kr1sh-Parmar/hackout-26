"""The physics has to be right before anything learned on top of it can be."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from src.core.config import load_region, site_master
from src.features.wind import power_curve, wind_features


def test_clearsky_ghi_at_equatorial_solar_noon_is_physical():
    """~1000 W/m2 is the number the whole industry normalises to. If this drifts,
    every capacity factor downstream drifts with it."""
    idx = pd.date_range("2024-06-21 12:00", periods=1, freq="h", tz="UTC")
    ghi = Location(0.0, 0.0, tz="UTC").get_clearsky(idx, model="ineichen")["ghi"].iloc[0]
    assert 900 <= ghi <= 1100, ghi


def test_clearsky_ghi_is_zero_at_night_in_belgium():
    idx = pd.date_range("2024-12-21 00:00", periods=4, freq="h", tz="UTC")
    ghi = Location(50.85, 4.35, tz="UTC").get_clearsky(idx, model="ineichen")["ghi"]
    assert (ghi == 0).all(), ghi.to_dict()


@pytest.mark.parametrize(("v", "expected"), [(2.0, 0.0), (12.0, 1.0), (20.0, 1.0), (26.0, 0.0)])
def test_power_curve_anchor_points(v, expected):
    assert power_curve(v) == pytest.approx(expected)


def test_power_curve_cut_out_is_a_cliff_not_a_slope():
    assert power_curve(25.0) == 1.0
    assert power_curve(25.0001) == 0.0


def test_power_curve_is_monotone_on_the_ramp():
    v = np.linspace(0.0, 25.0, 500)
    assert np.all(np.diff(power_curve(v)) >= -1e-12)


def test_hot_air_gives_a_lower_density_corrected_wind_speed():
    """Air density, not just wind speed, sets the power. A 45 C afternoon is a
    genuinely weaker wind than the same 10 m/s at 15 C."""
    site = site_master(load_region("BE"), "wind")[0]
    idx = pd.date_range("2024-07-01", periods=2, freq="h", tz="UTC")
    wx = pd.DataFrame(
        {
            "wind_speed_100m_ms": [10.0, 10.0],
            "wind_speed_10m_ms": [7.0, 7.0],
            "wind_gusts_10m_ms": [11.0, 11.0],
            "temperature_2m_c": [15.0, 45.0],
            "surface_pressure_hpa": [1013.0, 1013.0],
            "wind_direction_100m_deg": [270.0, 270.0],
        },
        index=idx,
    )
    out = wind_features(wx, site)
    cool, hot = out["ws_density_corrected_ms"]
    assert hot < cool
    assert out["air_density_kgm3"].iloc[1] < out["air_density_kgm3"].iloc[0]


def test_a_tracker_is_not_modelled_as_a_flat_panel():
    """`tracking` was a declared archetype field nothing read, which was
    invisible while every Belgian archetype was `fixed`. India's fleet is half
    single-axis trackers whose nameplate tilt is 0 degrees, so reading it as a
    fixed array models a horizontal panel and loses exactly the morning and
    evening output a tracker exists to capture."""
    import numpy as np
    import pandas as pd

    from src.core.config import Archetype, SiteMaster
    from src.features.solar import solar_features

    idx = pd.date_range("2026-06-21", periods=24, freq="h", tz="UTC")
    ramp = np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None)
    wx = pd.DataFrame(
        {
            "ghi_wm2": 900 * ramp,
            "dni_wm2": 700 * ramp,
            "dhi_wm2": 200 * ramp,
            "temperature_2m_c": 30.0,
            "wind_speed_10m_ms": 3.0,
        },
        index=idx,
    )

    def day_mwh(tracking, tilt):
        a = Archetype(
            id="x",
            share=1.0,
            tilt=tilt,
            azimuth=180,
            tracking=tracking,
            dc_ac_ratio=1.3,
            gamma_pdc=-0.004,
            albedo=0.25,
        )
        site = SiteMaster(
            site_id="s",
            region_id="IN",
            tech="solar",
            lat=24.0,
            lon=78.0,
            capacity_mw=100,
            capacity_share=1.0,
            archetype=a,
        )
        return float(solar_features(wx, site).physics_pac_mw.sum())

    flat = day_mwh("fixed", 0)
    tracked = day_mwh("single_axis", 0)
    assert tracked > flat * 1.1, "a tracker must beat the flat panel it would otherwise be"

    with pytest.raises(ValueError, match="unknown tracking mode"):
        day_mwh("dual_axis", 0)
