"""Rung-0 baselines. Every reported accuracy number is a skill score against these.

WHY THIS FILE IS CAREFUL ABOUT LEAKAGE
--------------------------------------
The skill score divides by the baseline's error, so a baseline that is accidentally
too weak inflates every claim, and one that accidentally sees the future makes the
model look worse than it is. Both failures are invisible in the output.

Measured on this dataset: every run is issued at 00Z, so the textbook
"persistence = the value at issue time" is *predicting midnight* for solar -- i.e.
predicting zero. It scores 20.3% nRMSE, and a model at 9% would appear to have
skill 0.56 against it. The honest, leak-free baseline scores 11.6-12.5% and the
same model has skill ~0.28. Only the second number is real.

THE RULE: a forecast issued at `run_ts` may only use actuals observed at or before
`run_ts`. For a valid time at lead h, the most recent same-hour-of-day observation
available is `valid_ts - 24 * ceil(h / 24)` hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24


def leak_free_lag_hours(lead_hours: pd.Series | np.ndarray) -> np.ndarray:
    """Whole days to step back so the source observation precedes the run time.

    valid - k*24 <= run  and  valid = run + lead   =>   k = ceil(lead / 24)
    """
    lead = np.asarray(lead_hours, dtype=float)
    return (np.ceil(lead / HOURS_PER_DAY) * HOURS_PER_DAY).astype("int64")


def hourly_actual_cf(actuals: pd.DataFrame) -> pd.Series:
    """Collapse a 15-minute silver generation table to an hourly capacity-factor series.

    Rows that failed quality control are dropped rather than averaged in -- a
    curtailed interval is a censored label, not a weather outcome.
    """
    df = actuals.copy()
    if "qc_flag" in df.columns:
        df = df[df["qc_flag"] == "OK"]
    df["_bucket"] = pd.to_datetime(df["ts_utc"], utc=True).dt.floor("h")
    g = df.groupby("_bucket").agg(p=("power_mw", "mean"), c=("monitored_capacity_mw", "mean"))
    out = (g["p"] / g["c"]).rename("actual_cf")
    out.index.name = "ts_utc"  # never "hour" -- it would collide downstream
    return out


def persistence(frame: pd.DataFrame, actual_cf: pd.Series) -> pd.Series:
    """Lead-aware, leak-free persistence in capacity-factor space.

    Args:
        frame: must carry `valid_ts_utc`, `lead_hours` and `run_ts_utc`.
        actual_cf: hourly capacity factor indexed by timestamp (see `hourly_actual_cf`).

    Returns:
        Series aligned to `frame.index`.

    Raises:
        ValueError: if any source observation would postdate its run time. This is
            a hard failure, not a warning -- a leaking baseline silently corrupts
            every skill score downstream.
    """
    lag_h = leak_free_lag_hours(frame["lead_hours"])
    src = pd.to_datetime(frame["valid_ts_utc"], utc=True) - pd.to_timedelta(lag_h, unit="h")

    run = pd.to_datetime(frame["run_ts_utc"], utc=True)
    bad = int((src > run).sum())
    if bad:
        raise ValueError(
            f"persistence baseline would use {bad} observations later than their run_ts -- "
            "this leaks the future into the reference and inflates every skill score"
        )
    return pd.Series(src.map(actual_cf).to_numpy(), index=frame.index, name="persistence_cf")


def smart_persistence(
    frame: pd.DataFrame, actual_cf: pd.Series, clearsky_cf: pd.Series | None = None
) -> pd.Series:
    """Persistence scaled by the clear-sky ratio: kt(past) * clearsky(valid).

    The standard solar improvement on plain persistence: it carries yesterday's
    *clearness* forward rather than yesterday's power, so it is not fooled by the
    seasonal and diurnal cycle. Needs a clear-sky series from the physics layer;
    without one this degrades to plain persistence and says so.
    """
    base = persistence(frame, actual_cf)
    if clearsky_cf is None:
        return base.rename("smart_persistence_cf")

    lag_h = leak_free_lag_hours(frame["lead_hours"])
    src = pd.to_datetime(frame["valid_ts_utc"], utc=True) - pd.to_timedelta(lag_h, unit="h")
    cs_past = pd.Series(src.map(clearsky_cf).to_numpy(), index=frame.index)
    cs_now = pd.Series(
        pd.to_datetime(frame["valid_ts_utc"], utc=True).map(clearsky_cf).to_numpy(),
        index=frame.index,
    )
    kt = (base / cs_past.where(cs_past > 1e-6)).clip(0, 1.3)
    return (kt * cs_now).clip(lower=0).rename("smart_persistence_cf")


def climatology(frame: pd.DataFrame, actual_cf: pd.Series, window_days: int = 30) -> pd.Series:
    """Mean capacity factor for this hour-of-day over the trailing window.

    The weakest defensible reference, and the one that exposes a model with no
    real skill: anything that cannot beat the seasonal average is not forecasting.
    Only observations strictly before each run time contribute.
    """
    hist = actual_cf.to_frame("cf")
    hist.index.name = "ts_utc"
    hist["hour_of_day"] = hist.index.hour
    out = np.full(len(frame), np.nan)

    run_ts = pd.to_datetime(frame["run_ts_utc"], utc=True)
    valid_hour = pd.to_datetime(frame["valid_ts_utc"], utc=True).dt.hour

    for run, grp in frame.groupby(run_ts):
        lo = run - pd.Timedelta(days=window_days)
        win = hist[(hist.index >= lo) & (hist.index < run)]
        if win.empty:
            continue
        means = win.groupby("hour_of_day")["cf"].mean()
        idx = frame.index.get_indexer(grp.index)
        out[idx] = valid_hour.loc[grp.index].map(means).to_numpy()

    return pd.Series(out, index=frame.index, name="climatology_cf")
