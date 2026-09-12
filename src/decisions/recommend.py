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

import pandas as pd

from ..core.config import GridState, RegionConfig


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
    raise NotImplementedError("M7 -- see dev-01 §10 and 01-technical-approach §5")


def empty_actions() -> pd.DataFrame:
    """The contract's shape, for building against before M7 lands."""
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES.get(c, "float64")) for c in RECOMMEND_COLUMNS})
    for c in ("valid_from", "valid_to"):
        df[c] = pd.Series(dtype="datetime64[ns, UTC]")
    return df[RECOMMEND_COLUMNS]
