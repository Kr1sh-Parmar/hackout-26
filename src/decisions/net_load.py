"""CONTRACT M6 -- forecast to net-load band.

Net load = demand MINUS generation. So the UPPER net-load bound (the
deficit-risk case) comes from the renewable P10 -- the pessimistic generation
scenario -- and the LOWER net-load bound comes from the renewable P90. Getting
this backwards inverts every deficit warning while still looking plausible,
because both numbers are "a p10 and a p90 of something". See
tests/unit/test_net_load_band.py, which pins this.
"""

from __future__ import annotations

import pandas as pd

from ..core.config import RegionConfig

OUTLOOK_COLUMNS: list[str] = [
    "region_id",
    "valid_ts_utc",
    "lead_hours",
    "demand_mw",
    "solar_p10_mw",
    "solar_p50_mw",
    "solar_p90_mw",
    "wind_p10_mw",
    "wind_p50_mw",
    "wind_p90_mw",
    "net_load_mw",
    "net_load_p10_mw",
    "net_load_p90_mw",
    "must_run_mw",
    "headroom_mw",
    "ramp_mw_per_h",
]


def _pivot_tech(fc: pd.DataFrame, tech: str) -> pd.DataFrame:
    cols = {"p10_mw": f"{tech}_p10_mw", "p50_mw": f"{tech}_p50_mw", "p90_mw": f"{tech}_p90_mw"}
    sub = fc.loc[fc["tech"] == tech, ["valid_ts_utc", "lead_hours", *cols]].rename(columns=cols)
    return sub


def build_outlook(fc: pd.DataFrame, demand: pd.DataFrame, cfg: RegionConfig) -> pd.DataFrame:
    """Build the net-load outlook one forecast run implies.

    Args:
        fc: long-format forecast, PREDICT_COLUMNS shape (one row per
            valid_ts_utc x tech), already filtered to a single run_ts_utc.
        demand: a day-ahead demand forecast frame. Must carry a timestamp
            column (`valid_ts_utc` or `ts_utc`) and a day-ahead point column.
            **Never `demand_mw`** -- that is the measured actual, unknown at
            forecast time; using it leaks the future into a decision that has
            to be made before the fact. The silver load table names the
            day-ahead column `demand_da_6pm_mw`; the gold feature table
            renames it to `demand_da_mw`. Both are accepted so callers don't
            have to rename first.
        cfg: region thresholds (`must_run_mw`) and resolution.

    Returns:
        One row per valid_ts_utc, OUTLOOK_COLUMNS.
    """
    if fc.empty or demand.empty:
        return pd.DataFrame(columns=OUTLOOK_COLUMNS)

    demand = demand.rename(columns={"ts_utc": "valid_ts_utc"})
    if "demand_da_mw" not in demand.columns and "demand_da_6pm_mw" in demand.columns:
        demand = demand.rename(columns={"demand_da_6pm_mw": "demand_da_mw"})
    if "demand_da_mw" not in demand.columns:
        raise KeyError(
            "demand frame has no day-ahead demand column "
            "(expected demand_da_mw or demand_da_6pm_mw); demand_mw (the "
            "actual) must never be used to build a forward-looking outlook"
        )

    solar = _pivot_tech(fc, "solar")
    wind = _pivot_tech(fc, "wind").drop(columns=["lead_hours"])
    merged = solar.merge(wind, on="valid_ts_utc", how="outer")
    merged = merged.merge(demand[["valid_ts_utc", "demand_da_mw"]], on="valid_ts_utc", how="inner")
    merged = merged.sort_values("valid_ts_utc").reset_index(drop=True)
    merged = merged.rename(columns={"demand_da_mw": "demand_mw"})

    region_id = fc["region_id"].iloc[0] if "region_id" in fc.columns and len(fc) else cfg.region_id
    merged["region_id"] = region_id

    # inversion is deliberate: net load = demand - generation, so the upper
    # (deficit-risk) bound uses the renewable P10, the lower bound uses P90.
    merged["net_load_mw"] = merged["demand_mw"] - merged["solar_p50_mw"] - merged["wind_p50_mw"]
    merged["net_load_p90_mw"] = merged["demand_mw"] - merged["solar_p10_mw"] - merged["wind_p10_mw"]
    merged["net_load_p10_mw"] = merged["demand_mw"] - merged["solar_p90_mw"] - merged["wind_p90_mw"]

    must_run = cfg.decision_thresholds.must_run_mw
    merged["must_run_mw"] = must_run
    merged["headroom_mw"] = merged["net_load_mw"] - must_run

    hours_per_step = (
        merged["valid_ts_utc"].diff().dt.total_seconds().median() / 3600.0
        if len(merged) > 1
        else 1.0
    )
    hours_per_step = hours_per_step or 1.0
    merged["ramp_mw_per_h"] = merged["net_load_mw"].diff() / hours_per_step

    return merged[OUTLOOK_COLUMNS]


def empty_outlook() -> pd.DataFrame:
    """The contract's shape, for building against before a real run lands."""
    return pd.DataFrame(columns=OUTLOOK_COLUMNS)
