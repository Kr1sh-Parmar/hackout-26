"""Rung 1: the physics-only forecast (dev-03 SS6).

Works with zero training history, which is what makes a new region deployable on
day one. Everything above it is a correction to this.
"""

from __future__ import annotations

import pandas as pd

from ..core.config import RegionConfig, site_master
from ..features.solar import solar_features
from ..features.wind import wind_features


def physics_forecast(cfg: RegionConfig, wx: pd.DataFrame, tech: str) -> pd.Series:
    """Regional capacity factor from the physics chain alone.

    Capacity factor, not MW: fleet capacity changes month to month, and a model
    trained in MW has to relearn the fleet every time someone commissions a farm.
    """
    sites = site_master(cfg, tech)
    kernel = solar_features if tech == "solar" else wind_features
    col = "physics_pac_mw" if tech == "solar" else "physics_power_mw"

    wxi = wx.set_index(pd.DatetimeIndex(wx["valid_ts_utc"])) if "valid_ts_utc" in wx else wx
    # Sum, not weighted-mean: each archetype's MW already carries its own share.
    total_mw = sum(kernel(wxi, s)[col] for s in sites)
    return (total_mw / cfg.capacity_mw[tech]).clip(0.0, 1.0).rename(f"physics_{tech}_cf")
