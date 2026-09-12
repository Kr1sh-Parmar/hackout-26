"""LP dispatch optimiser -- the `[STRETCH]` upgrade from rule-based storage
dispatch (`storage_sim.simulate`) to true optimisation.

Every hour's surplus (over-generation) or deficit (net-load shortfall) is a
fixed physical quantity implied by the outlook; the only decision is how much
of it storage absorbs versus how much is curtailed or backed up by fuel. That
makes this a small linear program: minimise `curtail_mw * curtailment_price +
backup_mw * backup_price` over the horizon, subject to the battery's own
physics.

Round-trip efficiency is applied on the CHARGE leg only, matching
`storage_sim.simulate` -- applying it on both legs double-counts the loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from ..core.config import Prices, StorageConfig

RESULT_COLUMNS: list[str] = [
    "valid_ts_utc",
    "charge_mw",
    "discharge_mw",
    "soc_mwh",
    "curtail_mw",
    "backup_mw",
]


def _hours_per_step(ts: pd.Series) -> float:
    if len(ts) < 2:
        return 1.0
    h = ts.diff().dt.total_seconds().median() / 3600.0
    return float(h) if h and not np.isnan(h) else 1.0


def optimise_dispatch(
    outlook: pd.DataFrame,
    storage: StorageConfig,
    prices: Prices,
    horizon_hours: int = 72,
) -> pd.DataFrame:
    """Minimise curtailed MWh + backup fuel cost over the horizon.

    Args:
        outlook: must carry `valid_ts_utc` and `headroom_mw` (net load minus
            must-run; negative = over-generation, positive = deficit) -- see
            `decisions.net_load.build_outlook`.
        storage: battery physics (power/energy limits, efficiency, SoC band).
        prices: `curtailment_opportunity_cost_per_mwh`, `backup_fuel_cost_per_mwh`.
        horizon_hours: rows beyond this many hours from the start are dropped.

    Returns:
        One row per timestep, RESULT_COLUMNS. Empty (but shaped) frame if
        `outlook` is empty.
    """
    if outlook.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    ordered = outlook.sort_values("valid_ts_utc").reset_index(drop=True)
    dt_h = _hours_per_step(ordered["valid_ts_utc"])
    if dt_h > 0:
        max_rows = max(1, int(round(horizon_hours / dt_h)))
        ordered = ordered.iloc[:max_rows].reset_index(drop=True)
    n = len(ordered)

    cap = storage.energy_capacity_mwh
    soc_min = storage.soc_min_frac * cap
    soc_max = storage.soc_max_frac * cap
    power = storage.power_rating_mw
    soc_init = min(max(0.5 * cap, soc_min), soc_max)

    headroom = ordered["headroom_mw"].astype(float).to_numpy()
    surplus = np.clip(-headroom, 0.0, None)  # over-generation MW, per hour
    deficit = np.clip(headroom, 0.0, None)  # net-load shortfall MW, per hour

    prob = pulp.LpProblem("dispatch", pulp.LpMinimize)
    charge = [pulp.LpVariable(f"charge_{t}", 0, power) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", 0, power) for t in range(n)]
    curtail = [pulp.LpVariable(f"curtail_{t}", 0) for t in range(n)]
    backup = [pulp.LpVariable(f"backup_{t}", 0) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", soc_min, soc_max) for t in range(n)]

    prob += pulp.lpSum(
        (
            curtail[t] * prices.curtailment_opportunity_cost_per_mwh
            + backup[t] * prices.backup_fuel_cost_per_mwh
        )
        * dt_h
        for t in range(n)
    )

    for t in range(n):
        prev_soc = soc_init if t == 0 else soc[t - 1]
        prob += (
            soc[t]
            == prev_soc + charge[t] * dt_h * storage.round_trip_efficiency - discharge[t] * dt_h
        )
        prob += charge[t] + curtail[t] == float(surplus[t])
        prob += discharge[t] + backup[t] == float(deficit[t])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"dispatch LP did not solve to optimality: {pulp.LpStatus[status]}")

    return pd.DataFrame(
        {
            "valid_ts_utc": ordered["valid_ts_utc"].to_numpy(),
            "charge_mw": [charge[t].value() for t in range(n)],
            "discharge_mw": [discharge[t].value() for t in range(n)],
            "soc_mwh": [soc[t].value() for t in range(n)],
            "curtail_mw": [curtail[t].value() for t in range(n)],
            "backup_mw": [backup[t].value() for t in range(n)],
        },
        columns=RESULT_COLUMNS,
    )
