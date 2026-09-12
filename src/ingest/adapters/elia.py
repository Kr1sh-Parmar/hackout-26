"""Elia live load forecast: the demand side of the decision layer.

Without a forward demand forecast the decision layer has nothing to compute net
load against, and the docs are explicit that running it on invented demand is a
project-killer. Elia publishes ~4 days of forward load on `ods001` at 15-minute
resolution with day-ahead, week-ahead and most-recent vintages plus P10/P90.

VINTAGE PRECEDENCE MATTERS. `dayaheadforecast` is the operator's committed
day-ahead number and is what the market clears against; `weekaheadforecast`
covers the tail of a 72 h horizon where day-ahead does not yet exist;
`mostrecentforecast` is the freshest but is only published close to delivery.
We take the best available per hour, in that order, and record which was used.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

RECORDS_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{ds}/records"
LOAD_DATASET = "ods001"
PAGE = 100  # the /records endpoint caps at 100 rows per call

# preference order, best first
VINTAGES = ["dayaheadforecast", "weekaheadforecast", "mostrecentforecast"]


class LoadUnavailable(RuntimeError):
    """Forward demand could not be fetched. Fall back, do not invent it."""


def _page(ds: str, where: str, offset: int, timeout: int) -> list[dict]:
    r = requests.get(
        RECORDS_URL.format(ds=ds),
        params={"where": where, "limit": PAGE, "offset": offset, "order_by": "datetime"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_load_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int = 90,
    tries: int = 3,
) -> pd.DataFrame:
    """Forward load forecast between `start` and `end`, hourly, UTC.

    Returns columns: `valid_ts_utc, demand_da_mw, demand_da_p10_mw,
    demand_da_p90_mw, demand_vintage`.
    """
    where = f"datetime >= '{start:%Y-%m-%dT%H:%M:%S}' AND datetime <= '{end:%Y-%m-%dT%H:%M:%S}'"
    rows: list[dict] = []
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            rows, offset = [], 0
            while True:
                got = _page(LOAD_DATASET, where, offset, timeout)
                rows.extend(got)
                if len(got) < PAGE or offset > 5_000:
                    break
                offset += PAGE
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * attempt)
    else:
        raise LoadUnavailable(f"elia {LOAD_DATASET}: {last}") from last

    if not rows:
        raise LoadUnavailable(f"elia {LOAD_DATASET} returned no rows for {start}..{end}")

    df = pd.DataFrame(rows)
    df["valid_ts_utc"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")

    # best available vintage per interval, recorded rather than silently blended
    df["demand_da_mw"] = pd.NA
    df["demand_vintage"] = pd.NA
    for v in VINTAGES:
        if v not in df.columns:
            continue
        fill = df["demand_da_mw"].isna() & df[v].notna()
        df.loc[fill, "demand_da_mw"] = df.loc[fill, v]
        df.loc[fill, "demand_vintage"] = v

    for q, col in (("10", "demand_da_p10_mw"), ("90", "demand_da_p90_mw")):
        src = f"mostrecentconfidence{q}"
        df[col] = df[src] if src in df.columns else pd.NA

    df = df.dropna(subset=["demand_da_mw"])
    if df.empty:
        raise LoadUnavailable("elia published no usable forward load vintage")

    df["hour"] = df["valid_ts_utc"].dt.floor("h")
    out = (
        df.groupby("hour")
        .agg(
            demand_da_mw=("demand_da_mw", "mean"),
            demand_da_p10_mw=("demand_da_p10_mw", "mean"),
            demand_da_p90_mw=("demand_da_p90_mw", "mean"),
            demand_vintage=("demand_vintage", "first"),
        )
        .reset_index()
        .rename(columns={"hour": "valid_ts_utc"})
    )
    out["demand_mw"] = out["demand_da_mw"]  # the decision layer's alias
    return out


def climatology_load(
    history: pd.DataFrame, index: pd.DatetimeIndex, weeks: int = 8
) -> pd.DataFrame:
    """Hour-of-week demand profile from recent history. The offline fallback.

    Demand is strongly hour-of-week periodic, so this is a genuinely serviceable
    substitute -- but it is a FALLBACK and every row says so via
    `demand_vintage='climatology'`, because an operator must never mistake it for
    the TSO's own number.
    """
    h = history.copy()
    h["ts"] = pd.to_datetime(h["ts_utc"], utc=True)
    recent = h[h["ts"] >= h["ts"].max() - pd.Timedelta(weeks=weeks)]
    recent = recent.assign(how=recent["ts"].dt.dayofweek * 24 + recent["ts"].dt.hour)
    profile = recent.groupby("how")["demand_mw"].mean()

    how = pd.Index(index).dayofweek * 24 + pd.Index(index).hour
    vals = pd.Series(how, index=index).map(profile)
    return pd.DataFrame(
        {
            "valid_ts_utc": index,
            "demand_mw": vals.to_numpy(),
            "demand_da_mw": vals.to_numpy(),
            "demand_da_p10_mw": vals.to_numpy() * 0.93,
            "demand_da_p90_mw": vals.to_numpy() * 1.07,
            "demand_vintage": "climatology",
        }
    )
