"""Autoregressive features (dev-03 SS4), lead-aware and leak-free.

NO LAG UNDER 24 HOURS, EVER. At lead hour 48 `power_lag_1h` does not exist: the
live system has not observed it. The origin of every lag is
`valid_ts - 24*ceil(lead/24)`, which is always at or before the run time, so a
backtest sees exactly what deployment sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_COLUMNS = [
    "power_lag_24h",
    "power_lag_168h",
    "roll_mean_24h",
    "roll_std_24h",
    "smart_persistence_cf",
]

MIN_LAG_HOURS = 24


def lag_origin(index: pd.DatetimeIndex, lead_hours) -> pd.DatetimeIndex:
    """The most recent timestamp a run at `valid_ts - lead` could have observed."""
    lead = np.asarray(lead_hours, dtype=float)
    back = MIN_LAG_HOURS * np.ceil(np.maximum(lead, 1.0) / MIN_LAG_HOURS)
    if (back < MIN_LAG_HOURS).any():
        raise ValueError(f"lag shorter than {MIN_LAG_HOURS}h requested; that leaks")
    return pd.DatetimeIndex(index) - pd.to_timedelta(back, unit="h")


def _hourly_cf(actuals: pd.DataFrame) -> pd.Series:
    cap = actuals["monitored_capacity_mw"].replace(0.0, np.nan)
    cf = (actuals["power_mw"] / cap).astype(float)
    cf.index = pd.DatetimeIndex(actuals["ts_utc"])
    return cf.groupby(level=0).mean().sort_index().resample("1h").mean()


def lag_features(
    actuals: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    lead_hours,
) -> pd.DataFrame:
    """Lag family for `index`, honouring the horizon each row was forecast at.

    `actuals is None` is the serving path: all-NaN columns, which LightGBM
    handles natively. Identical code path, so training and serving cannot drift.
    """
    index = pd.DatetimeIndex(index)
    origin = lag_origin(index, lead_hours)

    if actuals is None or len(actuals) == 0:
        return pd.DataFrame({c: np.full(len(index), np.nan) for c in LAG_COLUMNS}, index=index)

    cf = _hourly_cf(actuals)
    roll = cf.rolling(24, min_periods=6)
    # Seven same-hour-of-day observations ending at the origin -- a persistence
    # that knows about the diurnal cycle instead of fighting it.
    smart = sum(cf.shift(24 * k) for k in range(7)) / 7.0

    look = {
        "power_lag_24h": cf,
        "power_lag_168h": cf.shift(168 - MIN_LAG_HOURS),
        "roll_mean_24h": roll.mean(),
        "roll_std_24h": roll.std(),
        "smart_persistence_cf": smart,
    }
    return pd.DataFrame(
        {name: s.reindex(origin).to_numpy(dtype=float) for name, s in look.items()},
        index=index,
    )
