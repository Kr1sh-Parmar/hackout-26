"""Pins fact A: net load = demand - generation, so the upper net-load bound
must come from the renewable P10 (least generation), and the lower bound from
the renewable P90 (most generation). Getting this backwards inverts every
deficit warning while the numbers still look plausible.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_region
from src.decisions.net_load import build_outlook


def _fc(region_id="BE"):
    ts = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    rows = []
    for t in ts:
        rows.append(
            dict(
                region_id=region_id,
                valid_ts_utc=t,
                lead_hours=24,
                tech="solar",
                p10_mw=100.0,
                p50_mw=300.0,
                p90_mw=600.0,
            )
        )
        rows.append(
            dict(
                region_id=region_id,
                valid_ts_utc=t,
                lead_hours=24,
                tech="wind",
                p10_mw=50.0,
                p50_mw=200.0,
                p90_mw=500.0,
            )
        )
    return pd.DataFrame(rows), ts


def test_upper_net_load_bound_uses_renewable_p10():
    fc, ts = _fc()
    demand = pd.DataFrame({"valid_ts_utc": ts, "demand_da_mw": [8000.0, 8000.0, 8000.0]})
    cfg = load_region("BE")

    outlook = build_outlook(fc, demand, cfg)

    assert len(outlook) == 3
    # renewable P10 < P50 < P90 => net_load_p90 (uses renewable P10) is the
    # LARGEST net load, net_load_p10 (uses renewable P90) is the SMALLEST.
    assert (outlook["net_load_p90_mw"] > outlook["net_load_mw"]).all()
    assert (outlook["net_load_mw"] > outlook["net_load_p10_mw"]).all()

    row = outlook.iloc[0]
    assert row["net_load_mw"] == 8000.0 - 300.0 - 200.0
    assert row["net_load_p90_mw"] == 8000.0 - 100.0 - 50.0
    assert row["net_load_p10_mw"] == 8000.0 - 600.0 - 500.0


def test_build_outlook_uses_day_ahead_demand_not_actual():
    """demand_mw is the measured actual -- unknown at forecast time. Using it
    would leak the future into a decision that has to be made before the fact.
    build_outlook must read the day-ahead column instead."""
    fc, ts = _fc()
    cfg = load_region("BE")
    demand = pd.DataFrame(
        {
            "valid_ts_utc": ts,
            "demand_da_mw": [8000.0] * 3,
            # a wildly different "actual" -- if build_outlook used this, the
            # net load numbers below would be very different.
            "demand_mw": [1.0] * 3,
        }
    )

    outlook = build_outlook(fc, demand, cfg)

    assert outlook.iloc[0]["demand_mw"] == 8000.0  # the day-ahead figure, not the actual (1.0)
    assert outlook.iloc[0]["net_load_mw"] == 8000.0 - 300.0 - 200.0


def test_accepts_silver_table_day_ahead_column_name():
    """The silver load table names the point day-ahead column
    demand_da_6pm_mw, not demand_da_mw -- accept either."""
    fc, ts = _fc()
    cfg = load_region("BE")
    demand = pd.DataFrame({"ts_utc": ts, "demand_da_6pm_mw": [8000.0] * 3})

    outlook = build_outlook(fc, demand, cfg)

    assert len(outlook) == 3
    assert outlook.iloc[0]["net_load_mw"] == 8000.0 - 300.0 - 200.0
