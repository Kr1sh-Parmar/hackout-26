"""Solar physics chain (dev-03 SS2).

Pure NumPy/pvlib on an already-regional weather frame. Spatial and NWP-model
aggregation happened in ETL; this runs ONE archetype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from ..core.config import SiteMaster

_SAPM = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

SOLAR_COLUMNS = [
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "solar_elevation_deg",
    "airmass",
    "is_day",
    "clearsky_ghi_wm2",
    "clearsky_index_kt",
    "poa_global_wm2",
    "cell_temperature_c",
    "thermal_derate",
    "physics_pac_mw",
    "clipping_headroom",
    "mins_since_sunrise",
    "mins_to_sunset",
]

# Cloud-edge enhancement genuinely pushes measured GHI above the clear-sky
# value. Clipping at 1.0 would erase a real high-output signal.
KT_MAX = 1.3

TRACKING_MODES = ("fixed", "single_axis")
# Typical utility-scale horizontal single-axis tracker: +/-60 degrees of travel,
# backtracking on, ground-cover ratio 0.35.
TRACKER_MAX_ANGLE = 60.0
TRACKER_GCR = 0.35


def _orientation(a, sp: pd.DataFrame):
    """Panel tilt and azimuth -- a constant for fixed tilt, a series for a tracker.

    `tracking` was a declared-but-never-read archetype field: every Belgian
    archetype is `fixed`, so nothing noticed. India's fleet is half single-axis
    trackers, and reading a tracker's nameplate tilt of 0 degrees as a fixed
    array models it as a flat panel -- which under-predicts precisely the
    morning and evening hours a tracker exists to capture. An unknown mode
    raises rather than falling back to fixed, because silently modelling the
    wrong array is how a config typo becomes a forecast nobody questions.
    """
    mode = a.tracking or "fixed"
    if mode not in TRACKING_MODES:
        raise ValueError(f"unknown tracking mode {mode!r}; expected one of {TRACKING_MODES}")
    if mode == "fixed":
        return float(a.tilt), float(a.azimuth)
    tr = pvlib.tracking.singleaxis(
        apparent_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        axis_tilt=0.0,
        axis_azimuth=float(a.azimuth),
        max_angle=TRACKER_MAX_ANGLE,
        backtrack=True,
        gcr=TRACKER_GCR,
    )
    # NaN where the sun is down; the array is parked flat and POA is zero anyway.
    return tr["surface_tilt"].fillna(0.0), tr["surface_azimuth"].fillna(float(a.azimuth))


def _sun_minutes(loc: Location, idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    days = idx.normalize().unique()
    rs = loc.get_sun_rise_set_transit(days)
    day_key = idx.normalize()
    sunrise = rs["sunrise"].reindex(day_key).dt.tz_localize(None).to_numpy()
    sunset = rs["sunset"].reindex(day_key).dt.tz_localize(None).to_numpy()
    now = idx.tz_convert("UTC").tz_localize(None).to_numpy()
    minute = np.timedelta64(1, "m")
    return (now - sunrise) / minute, (sunset - now) / minute


def solar_features(wx: pd.DataFrame, site: SiteMaster) -> pd.DataFrame:
    """Irradiance -> plane-of-array -> cell temperature -> AC power, for one archetype."""
    idx = pd.DatetimeIndex(wx.index)
    a = site.archetype
    loc = Location(site.lat, site.lon, tz="UTC", altitude=site.elevation_m or 0.0)

    sp = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx, model="ineichen", solar_position=sp)
    am = loc.get_airmass(idx, solar_position=sp)

    ghi = wx["ghi_wm2"].to_numpy(dtype=float)
    dni = wx["dni_wm2"].to_numpy(dtype=float)
    dhi = wx["dhi_wm2"].to_numpy(dtype=float)
    cs_ghi = cs["ghi"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(cs_ghi > 1.0, ghi / cs_ghi, 0.0)
    kt = np.clip(np.nan_to_num(kt), 0.0, KT_MAX)

    surface_tilt, surface_azimuth = _orientation(a, sp)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=pd.Series(dni, index=idx),
        ghi=pd.Series(ghi, index=idx),
        dhi=pd.Series(dhi, index=idx),
        dni_extra=pvlib.irradiance.get_extra_radiation(idx),
        albedo=a.albedo if a.albedo is not None else 0.20,
        model="haydavies",
    )["poa_global"].to_numpy(dtype=float)
    poa = np.nan_to_num(np.clip(poa, 0.0, None))

    t_cell = pvlib.temperature.sapm_cell(
        poa_global=poa,
        temp_air=wx["temperature_2m_c"].to_numpy(dtype=float),
        wind_speed=wx["wind_speed_10m_ms"].to_numpy(dtype=float),
        **_SAPM,
    )
    gamma = a.gamma_pdc if a.gamma_pdc is not None else -0.0035
    derate = 1.0 + gamma * (t_cell - 25.0)

    cap = site.capacity_mw
    dc_cap = cap * (a.dc_ac_ratio if a.dc_ac_ratio is not None else 1.0)
    # 0.985 = inverter + wiring; the residual model absorbs the rest of the
    # fleet loss stack (soiling, snow, shading, availability).
    pac = np.minimum(dc_cap * poa / 1000.0 * derate * 0.985, cap)
    pac = np.clip(pac, 0.0, cap)

    since, to = _sun_minutes(loc, idx)
    elev = sp["apparent_elevation"].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "solar_zenith_deg": sp["apparent_zenith"].to_numpy(dtype=float),
            "solar_azimuth_deg": sp["azimuth"].to_numpy(dtype=float),
            "solar_elevation_deg": elev,
            "airmass": am["airmass_relative"].to_numpy(dtype=float),
            "is_day": (elev > 0.0).astype(float),
            "clearsky_ghi_wm2": cs_ghi,
            "clearsky_index_kt": kt,
            "poa_global_wm2": poa,
            "cell_temperature_c": np.asarray(t_cell, dtype=float),
            "thermal_derate": np.asarray(derate, dtype=float),
            "physics_pac_mw": pac,
            "clipping_headroom": 1.0 - pac / cap if cap else np.zeros(len(idx)),
            "mins_since_sunrise": since,
            "mins_to_sunset": to,
        },
        index=idx,
    )
