"""Wind physics chain (dev-03 SS3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import SiteMaster

WIND_COLUMNS = [
    "ws_hub_ms",
    "air_density_kgm3",
    "ws_density_corrected_ms",
    "power_curve_cf",
    "physics_power_mw",
    "dP_dv",
    "turbulence_proxy",
    "wind_dir_sin",
    "wind_dir_cos",
    "below_cutin_flag",
    "above_cutout_flag",
]

R_SPECIFIC = 287.05  # dry air, J/(kg K)
RHO_REF = 1.225  # IEC 61400-12 reference density


def power_curve(
    v: float | np.ndarray,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> float | np.ndarray:
    """Generic cubic power curve, normalised to [0, 1].

    Above cut-out the turbine shuts down: a cliff to zero, not a slope. A
    smoothed cut-out looks nicer on a plot and mis-sizes reserve during a storm.
    """
    arr = np.asarray(v, dtype=float)
    ramp = (arr**3 - cut_in**3) / (rated**3 - cut_in**3)
    cf = np.where(arr < cut_in, 0.0, np.where(arr < rated, ramp, 1.0))
    cf = np.where(arr > cut_out, 0.0, cf)
    cf = np.clip(cf, 0.0, 1.0)
    return float(cf) if np.isscalar(v) or np.ndim(v) == 0 else cf


def wind_features(wx: pd.DataFrame, site: SiteMaster) -> pd.DataFrame:
    idx = pd.DatetimeIndex(wx.index)
    a = site.archetype
    cut_in, rated, cut_out = float(a.cut_in_ms), float(a.rated_ms), float(a.cut_out_ms)

    ws100 = wx["wind_speed_100m_ms"].to_numpy(dtype=float)
    ws10 = wx["wind_speed_10m_ms"].to_numpy(dtype=float)
    gusts = wx["wind_gusts_10m_ms"].to_numpy(dtype=float)
    temp_k = wx["temperature_2m_c"].to_numpy(dtype=float) + 273.15
    pressure_pa = wx["surface_pressure_hpa"].to_numpy(dtype=float) * 100.0
    direction = np.deg2rad(wx["wind_direction_100m_deg"].to_numpy(dtype=float))

    ws_hub = ws100 * (float(a.hub_height_m) / 100.0) ** float(a.shear_alpha)
    rho = pressure_pa / (R_SPECIFIC * temp_k)
    ws_corr = ws_hub * (rho / RHO_REF) ** (1.0 / 3.0)

    cf = power_curve(ws_corr, cut_in, rated, cut_out)
    wake = float(a.wake_loss_frac or 0.0)

    dp_dv = (
        power_curve(ws_corr + 0.5, cut_in, rated, cut_out)
        - power_curve(ws_corr - 0.5, cut_in, rated, cut_out)
    ) / 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        turb = np.where(ws10 > 0.5, (gusts - ws10) / ws10, 0.0)

    return pd.DataFrame(
        {
            "ws_hub_ms": ws_hub,
            "air_density_kgm3": rho,
            "ws_density_corrected_ms": ws_corr,
            "power_curve_cf": cf,
            "physics_power_mw": cf * site.capacity_mw * (1.0 - wake),
            "dP_dv": dp_dv,
            "turbulence_proxy": np.nan_to_num(turb),
            "wind_dir_sin": np.sin(direction),
            "wind_dir_cos": np.cos(direction),
            "below_cutin_flag": (ws_corr < cut_in).astype(float),
            "above_cutout_flag": (ws_corr > cut_out).astype(float),
        },
        index=idx,
    )
