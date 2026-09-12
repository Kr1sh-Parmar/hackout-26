"""CONTRACT M7 -- forecast to grid action.

The brief's users do not need a curve. They need to know that on Tuesday between
11:00 and 14:00 the region will be over-generating, how much storage absorbs,
what the remainder costs to curtail, and whether the recommendation is safe to
act on from the forecast alone.

Two properties separate a recommendation from an `if` statement, and both are in
the returned columns:

  * every action carries a NUMBER  -- `mwh` and `value_inr`
  * every action carries its CONFIDENCE -- `decisive` is True only when the
    ENTIRE p10-p90 band clears the threshold. A P50 that crosses while the band
    straddles it is a suggestion, not a decision. Operators do not need a
    probability; they need to know which of the two they are looking at.

Owner: Backend.  Consumers: api/, ui/.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from ..core.config import GridState, RegionConfig
from .events import scan_events
from .storage_sim import simulate


class Flag(str, Enum):
    NORMAL = "NORMAL"
    OVER_GENERATION = "OVER_GENERATION"
    STEEP_RAMP = "STEEP_RAMP"
    DEFICIT_RISK = "DEFICIT_RISK"
    STORM_SHUTDOWN = "STORM_SHUTDOWN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class Action(str, Enum):
    HOLD = "HOLD"
    CURTAIL = "CURTAIL"
    CHARGE_BESS = "CHARGE_BESS"
    DISCHARGE_BESS = "DISCHARGE_BESS"
    COMMIT_BACKUP = "COMMIT_BACKUP"


RECOMMEND_COLUMNS: list[str] = [
    "region_id",
    "action",
    "flag",
    "valid_from",
    "valid_to",
    "power_mw",
    "mwh",
    "value_inr",
    "confidence",
    "decisive",
    "rationale",
    "linked_event_id",
]

_DTYPES: dict[str, str] = {
    "region_id": "object",
    "action": "object",
    "flag": "object",
    "power_mw": "float64",
    "mwh": "float64",
    "value_inr": "float64",
    "confidence": "float64",
    "decisive": "bool",
    "rationale": "object",
    "linked_event_id": "object",
}


def recommend(
    outlook: pd.DataFrame,
    grid_state: GridState,
    config: RegionConfig,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank and size the grid actions implied by one forecast run.

    Args:
        outlook: one row per valid_ts with demand and the propagated renewable
            band. Must carry `net_load_mw`, `net_load_p10_mw`, `net_load_p90_mw`,
            `headroom_mw` and `ramp_mw_per_h`.
        grid_state: must-run floor, storage state of charge, ramp limit.
        config: region thresholds and prices. Thresholds are configuration, never
            code -- if you are about to type a number into this function body, it
            belongs in RegionConfig.

    Returns:
        Ranked actions with RECOMMEND_COLUMNS, most valuable first.

    Note:
        Net load is demand MINUS generation, so the UPPER net-load bound comes
        from the renewable P10, not P90. Getting this backwards inverts every
        deficit warning and is very easy to miss, because the numbers still look
        plausible.
    """
    required = {"net_load_mw", "net_load_p10_mw", "net_load_p90_mw", "headroom_mw", "valid_ts_utc"}
    if outlook is None or len(outlook) == 0:
        # Nothing to act on is a legitimate state (no forecast run yet), not an
        # error: return the contract's empty shape.
        return empty_actions()
    missing = required - set(outlook.columns)
    if missing:
        # Wrong shape IS an error, and a specific one. NotImplementedError would
        # claim this function was never built; it was.
        raise ValueError(
            f"outlook is missing {sorted(missing)} -- it must come from "
            "decisions.net_load.build_outlook()"
        )

    # Reuse the caller's events when given. Rescanning here would both duplicate
    # work and silently drop STORM_SHUTDOWN, which can only be detected from
    # per-grid-point weather the caller holds and this signature never sees.
    if events is None:
        events = scan_events(outlook, wx_points=None, cfg=config)
    if events.empty:
        return empty_actions()

    ordered = outlook.sort_values("valid_ts_utc").reset_index(drop=True)
    sim = simulate(ordered, grid_state.storage)
    total_capacity = config.capacity_mw.get("solar", 0) + config.capacity_mw.get("wind", 0)
    prices = config.prices

    rows: list[dict] = []
    for _, ev in events.iterrows():
        window = ordered[
            (ordered["valid_ts_utc"] >= ev["valid_from"])
            & (ordered["valid_ts_utc"] <= ev["valid_to"])
        ]
        if window.empty:
            continue
        band = window["net_load_p90_mw"] - window["net_load_p10_mw"]
        confidence = (
            float(np.clip(1 - band.mean() / total_capacity, 0, 1)) if total_capacity else 0.5
        )
        mwh = float(ev["energy_mwh"])
        power_mw = float(ev["magnitude_mw"])
        flag = ev["flag"]

        if flag == Flag.OVER_GENERATION.value:
            # decisive only if the ENTIRE band (worst case for a deficit,
            # i.e. net_load_p90_mw) still sits below the must-run floor --
            # otherwise the band straddles it and this is a suggestion, not
            # a decision.
            decisive = bool((window["net_load_p90_mw"] < window["must_run_mw"]).all())
            sim_window = sim[
                (sim["valid_ts_utc"] >= ev["valid_from"]) & (sim["valid_ts_utc"] <= ev["valid_to"])
            ]
            charged_mwh = float(sim_window["charge_mw"].sum())
            action = Action.CHARGE_BESS if charged_mwh > 0 else Action.CURTAIL
            value_inr = mwh * prices.curtailment_opportunity_cost_per_mwh
            rationale = (
                f"Net load falls up to {power_mw:.0f} MW below the "
                f"{window['must_run_mw'].iloc[0]:.0f} MW must-run floor -- "
                f"{'charge storage' if action is Action.CHARGE_BESS else 'curtail'} "
                f"{mwh:.0f} MWh rather than spill it."
            )
        elif flag == Flag.STEEP_RAMP.value:
            # a ramp is a point-signal diagnostic (derivative of the p50
            # curve, no quantile band of its own) -- decisive reduces to
            # "detected", which is the honest signal it can give.
            decisive = True
            action = Action.DISCHARGE_BESS
            value_inr = mwh * prices.backup_fuel_cost_per_mwh
            rationale = (
                f"Net load is moving {power_mw:.0f} MW/h -- discharge storage to "
                f"flatten the ramp rather than lean on backup fuel."
            )
        elif flag == Flag.DEFICIT_RISK.value:
            # by construction (events.py) this flag only fires when the
            # central forecast does NOT cross must_run but the pessimistic
            # band does -- the straddle itself is the definition, so this is
            # never decisive.
            decisive = False
            action = Action.COMMIT_BACKUP
            value_inr = mwh * prices.backup_fuel_cost_per_mwh
            rationale = (
                f"Worst-case net load could reach {power_mw:.0f} MW above the must-run "
                f"floor, though the central forecast does not cross it -- "
                f"pre-commit {mwh:.0f} MWh of backup as insurance, not a certainty."
            )
        elif flag == Flag.LOW_CONFIDENCE.value:
            decisive = False
            action = Action.HOLD
            mwh = 0.0
            power_mw = 0.0
            value_inr = 0.0
            rationale = (
                f"Forecast band is {ev['magnitude_mw']:.0f} MW wide here -- hold rather "
                f"than commit to an action the next run may reverse."
            )
        elif flag == Flag.STORM_SHUTDOWN.value:
            # observed/forecast wind speed vs. a physical cut-out, not a
            # quantile band -- same reasoning as STEEP_RAMP.
            decisive = True
            action = Action.COMMIT_BACKUP
            value_inr = mwh * prices.backup_fuel_cost_per_mwh
            rationale = (
                f"{power_mw:.0f} MW of wind capacity is at risk of storm cut-out -- "
                f"commit backup to cover the shortfall."
            )
        else:
            continue

        # A recommendation the asset cannot physically execute is worse than no
        # recommendation: it puts a number on screen that an operator would act
        # on. Observed before this clamp: DISCHARGE_BESS 10,296 MWh at 1,450 MW
        # from a 400 MWh / 100 MW battery -- 25x its entire capacity.
        if action in (Action.CHARGE_BESS, Action.DISCHARGE_BESS):
            st = grid_state.storage
            duration_h = max((ev["valid_to"] - ev["valid_from"]) / pd.Timedelta(hours=1) + 1.0, 1.0)
            if action is Action.DISCHARGE_BESS:
                usable_mwh = max(
                    grid_state.storage_soc_mwh - st.energy_capacity_mwh * st.soc_min_frac, 0.0
                )
            else:
                usable_mwh = max(
                    st.energy_capacity_mwh * st.soc_max_frac - grid_state.storage_soc_mwh, 0.0
                )

            capped_power = min(power_mw, st.power_rating_mw)
            capped_mwh = min(mwh, usable_mwh, capped_power * duration_h)
            if capped_mwh < mwh - 1e-6:
                rationale += (
                    f" Storage covers {capped_mwh:,.0f} of {mwh:,.0f} MWh "
                    f"({st.power_rating_mw:.0f} MW / {st.energy_capacity_mwh:.0f} MWh asset); "
                    f"the remaining {mwh - capped_mwh:,.0f} MWh needs another instrument."
                )
            power_mw, mwh = capped_power, capped_mwh
            value_inr = mwh * (
                prices.curtailment_opportunity_cost_per_mwh
                if action is Action.CHARGE_BESS
                else prices.backup_fuel_cost_per_mwh
            )
            if mwh <= 0:
                continue  # nothing the asset can do here

        rows.append(
            {
                "region_id": config.region_id,
                "action": action.value,
                "flag": flag,
                "valid_from": ev["valid_from"],
                "valid_to": ev["valid_to"],
                "power_mw": power_mw,
                "mwh": mwh,
                "value_inr": value_inr,
                "confidence": confidence,
                "decisive": decisive,
                "rationale": rationale,
                "linked_event_id": ev["event_id"],
            }
        )

    if not rows:
        return empty_actions()
    df = pd.DataFrame(rows)[RECOMMEND_COLUMNS]
    return df.sort_values("value_inr", ascending=False).reset_index(drop=True)


def empty_actions() -> pd.DataFrame:
    """The contract's shape, for building against before M7 lands."""
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES.get(c, "float64")) for c in RECOMMEND_COLUMNS})
    for c in ("valid_from", "valid_to"):
        df[c] = pd.Series(dtype="datetime64[ns, UTC]")
    return df[RECOMMEND_COLUMNS]
