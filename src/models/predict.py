"""CONTRACT M4 + M5 -- forecast generation.

Returns calibrated quantiles, never a point forecast alone. Point forecasts are
operationally useless for reserve sizing: an operator needs to know how wrong
this might be, at each hour.

`calibrated` is part of the contract, not decoration. An uncalibrated band is a
decoration; a conformally calibrated one is a reserve requirement, and the
consumer is entitled to know which it just received.

Owner: ML / Physics.  Consumers: decisions/, api/, evaluation/.
"""

from __future__ import annotations

import pandas as pd

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

PREDICT_COLUMNS: list[str] = [
    "region_id",
    "run_ts_utc",
    "valid_ts_utc",
    "lead_hours",
    "tech",
    "p10_mw",
    "p50_mw",
    "p90_mw",
    "capacity_mw",
    "model_version",
    "calibration_date",
    "calibrated",
]

_DTYPES: dict[str, str] = {
    "region_id": "object",
    "lead_hours": "int32",
    "tech": "object",
    "p10_mw": "float64",
    "p50_mw": "float64",
    "p90_mw": "float64",
    "capacity_mw": "float64",
    "model_version": "object",
    "calibration_date": "object",
    "calibrated": "bool",
}


def predict(
    region_id: str,
    run_ts: pd.Timestamp,
    horizons: range = range(1, 73),
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    """Produce calibrated quantile forecasts for one region and one NWP run.

    Args:
        region_id: configured region, e.g. "BE".
        run_ts: model initialisation time, tz-aware UTC.
        horizons: lead hours to emit. The platform's operative band is 24-72 h;
            shorter leads are served but carry no autoregressive skill.
        quantiles: must be sorted ascending.

    Returns:
        One row per (valid_ts, tech) with PREDICT_COLUMNS. Quantiles are sorted
        per row -- independently fitted quantile heads can cross (p10 > p50),
        which is a real and embarrassing failure mode.

    Raises:
        FileNotFoundError: region not configured.
        RuntimeError: no model artifact registered for this region and tech.
    """
    raise NotImplementedError("M4 + M5 -- see dev-03 §6-§7")


def empty_forecast() -> pd.DataFrame:
    """The contract's shape, for building against before M4 lands."""
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES.get(c, "float64")) for c in PREDICT_COLUMNS})
    for c in ("run_ts_utc", "valid_ts_utc"):
        df[c] = pd.Series(dtype="datetime64[ns, UTC]")
    return df[PREDICT_COLUMNS]
