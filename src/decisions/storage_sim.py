"""A virtual BESS forward-simulated against one outlook.

Round-trip efficiency is applied on the CHARGE leg only: energy that flows in
is de-rated once by `round_trip_efficiency` before it lands on the SoC ledger.
Applying it again on discharge would double-count the loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Prices, StorageConfig

SIM_COLUMNS: list[str] = ["valid_ts_utc", "charge_mw", "discharge_mw", "soc_mwh"]


def _hours_per_step(ts: pd.Series) -> float:
    if len(ts) < 2:
        return 1.0
    h = ts.diff().dt.total_seconds().median() / 3600.0
    return float(h) if h and not np.isnan(h) else 1.0


def simulate(outlook: pd.DataFrame, st: StorageConfig) -> pd.DataFrame:
    """Forward energy-balance simulation of a BESS against one outlook.

    Charges into over-generation (`headroom_mw < 0`); discharges into deficit
    conditions (`headroom_mw > 0`) or a ramp too steep for anything but
    storage to follow. Power is capped at `power_rating_mw`; SoC is clamped to
    `[soc_min_frac, soc_max_frac] * energy_capacity_mwh`.

    ponytail: no initial-SoC parameter -- this sizes storage's steady-state
    contribution, so it starts at the midpoint of the allowed SoC band. Wire
    in `GridState.storage_soc_mwh` here if a caller needs the real starting
    charge (e.g. live dispatch rather than planning).
    """
    if outlook.empty:
        return pd.DataFrame(columns=SIM_COLUMNS)

    ordered = outlook.sort_values("valid_ts_utc").reset_index(drop=True)
    dt_h = _hours_per_step(ordered["valid_ts_utc"])

    cap = st.energy_capacity_mwh
    soc_min = st.soc_min_frac * cap
    soc_max = st.soc_max_frac * cap
    power = st.power_rating_mw
    soc = min(max(0.5 * cap, soc_min), soc_max)

    rows = []
    for _, r in ordered.iterrows():
        headroom = float(r["headroom_mw"])
        ramp = float(r["ramp_mw_per_h"]) if pd.notna(r["ramp_mw_per_h"]) else 0.0
        charge_mw = 0.0
        discharge_mw = 0.0

        if headroom < 0:
            room_mwh = max(soc_max - soc, 0.0)
            charge_mw = max(0.0, min(-headroom, power, room_mwh / dt_h))
            soc += charge_mw * dt_h * st.round_trip_efficiency
        elif headroom > 0 or abs(ramp) > power:
            avail_mwh = max(soc - soc_min, 0.0)
            target = headroom if headroom > 0 else power
            discharge_mw = max(0.0, min(target, power, avail_mwh / dt_h))
            soc -= discharge_mw * dt_h

        soc = min(max(soc, soc_min), soc_max)
        rows.append(
            {
                "valid_ts_utc": r["valid_ts_utc"],
                "charge_mw": charge_mw,
                "discharge_mw": discharge_mw,
                "soc_mwh": soc,
            }
        )
    return pd.DataFrame(rows, columns=SIM_COLUMNS)


def sweep(
    outlook: pd.DataFrame, st: StorageConfig, max_mwh: float = 1000, steps: int = 25
) -> pd.DataFrame:
    """Sweep candidate storage sizes and value what each one buys.

    ponytail: no `prices` argument in the pinned signature -- uses the
    RegionConfig default (`Prices()`). Pass a region-specific storage config
    with different economics upstream if that default materially understates
    the region's curtailment price.
    """
    cols = ["energy_capacity_mwh", "curtailment_avoided_gwh_yr", "value_inr_yr", "cycles_per_year"]
    if outlook.empty:
        return pd.DataFrame(columns=cols)

    ordered = outlook.sort_values("valid_ts_utc").reset_index(drop=True)
    dt_h = _hours_per_step(ordered["valid_ts_utc"])
    hours_covered = len(ordered) * dt_h
    annualise = 8760.0 / hours_covered if hours_covered else 0.0
    price = Prices().curtailment_opportunity_cost_per_mwh

    rows = []
    for cap in np.linspace(0, max_mwh, steps):
        variant = st.model_copy(update={"energy_capacity_mwh": float(cap)})
        sim = simulate(ordered, variant)
        charge_mwh = float((sim["charge_mw"] * dt_h).sum())
        avoided_gwh_yr = charge_mwh * annualise / 1000.0
        value_inr_yr = charge_mwh * annualise * price
        cycles_per_year = (charge_mwh / cap * annualise) if cap > 0 else 0.0
        rows.append(
            {
                "energy_capacity_mwh": float(cap),
                "curtailment_avoided_gwh_yr": avoided_gwh_yr,
                "value_inr_yr": value_inr_yr,
                "cycles_per_year": cycles_per_year,
            }
        )
    return pd.DataFrame(rows, columns=cols)
