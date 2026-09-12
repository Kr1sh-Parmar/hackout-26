"""Spatial and NWP-model aggregation: per-grid-point weather -> one regional row.

THIS MODULE EXISTS TO BE SHARED. The historical path (`scripts/build_gold.py`)
and the live path (`src/ingest/live.py`) must aggregate identically, or the model
is served inputs computed differently from the ones it was trained on. That is
train/serve skew, it does not show up in a backtest, and it is the failure this
project is most careful about elsewhere. One function, two callers.

A region has no single coordinate. Weather is sampled at 5-10 points weighted by
installed capacity nearby, because a single centroid systematically misses
frontal systems crossing the region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The variables carried through to the model matrix, in a fixed order.
WX_VARS: list[str] = [
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
    "direct_radiation_wm2",
    "temperature_2m_c",
    "dew_point_2m_c",
    "relative_humidity_2m_pct",
    "surface_pressure_hpa",
    "cloud_cover_pct",
    "cloud_cover_low_pct",
    "cloud_cover_mid_pct",
    "cloud_cover_high_pct",
    "wind_speed_10m_ms",
    "wind_speed_100m_ms",
    "wind_direction_100m_deg",
    "wind_gusts_10m_ms",
    "precipitation_mm",
]

KEY = ["run_ts_utc", "valid_ts_utc", "forecast_vintage"]


def capacity_weighted(wx: pd.DataFrame, key: list[str] | None = None) -> pd.DataFrame:
    """Weighted mean across grid points, renormalised over non-null weights.

    Renormalising matters: if one grid point is missing a variable, a plain
    weighted sum silently returns a smaller regional value rather than the mean
    of the points that did report. The result looks like calm weather.
    """
    key = list(key or KEY)
    group = key + ["nwp_model"]
    vars_ = [v for v in WX_VARS if v in wx.columns]

    num = wx[group].copy()
    den = wx[group].copy()
    for v in vars_:
        num[v] = wx[v] * wx["weight"]
        den[v] = wx["weight"].where(wx[v].notna())

    agg_n = num.groupby(group, observed=True)[vars_].sum(min_count=1)
    agg_w = den.groupby(group, observed=True)[vars_].sum(min_count=1)
    reg = (agg_n / agg_w.replace(0, np.nan)).reset_index()

    if "is_day" in wx.columns:
        # geometric, identical across points -- take it rather than averaging
        day = wx.groupby(group, observed=True)["is_day"].max().reset_index()
        reg = reg.merge(day, on=group, how="left")
    return reg


def blend_models(reg: pd.DataFrame, key: list[str] | None = None) -> pd.DataFrame:
    """One column block per NWP model, plus the blend and the DISAGREEMENT.

    Model disagreement is not decoration: it is among the strongest available
    predictors of our own error, and it is what lets a point forecast become an
    honest interval. Measured on this dataset it grows with lead time, which is
    the behaviour that proves the vintages really are different runs.
    """
    key = list(key or KEY)
    vars_ = [v for v in WX_VARS if v in reg.columns]

    wide = reg.pivot_table(index=key, columns="nwp_model", values=vars_, observed=True)
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    wide = wide.reset_index()

    for v in vars_:
        cols = [c for c in wide.columns if c.startswith(f"{v}__")]
        if not cols:
            continue
        wide[v] = wide[cols].mean(axis=1)
        wide[f"{v}_model_std"] = wide[cols].std(axis=1)
        wide[f"{v}_model_range"] = wide[cols].max(axis=1) - wide[cols].min(axis=1)

    if "is_day" in reg.columns:
        day = reg.groupby(key, observed=True)["is_day"].max().reset_index()
        wide = wide.merge(day, on=key, how="left")
    return wide


def regionalise(wx: pd.DataFrame, key: list[str] | None = None) -> pd.DataFrame:
    """Full path: per-point, per-model rows -> one regional row per key.

    This is the function both the historical ETL and the live fetch must call.
    """
    return blend_models(capacity_weighted(wx, key), key)
