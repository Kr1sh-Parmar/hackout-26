"""Drift monitoring.

Two independent questions, kept separate on purpose:

1. Have the INPUTS shifted (`feature_drift`) -- e.g. the fleet grew, a sensor
   was recalibrated, a season nobody trained on arrived.
2. Has the OUTPUT (error) gotten worse (`error_drift`) -- the thing that
   actually costs money, and the only one a retrain is justified by on its own.

Feature drift without error drift is often benign (the model generalises);
error drift without feature drift usually means a regime change the features
don't capture. `should_retrain` combines both into one decision with a reason,
because a monitor that says "retrain" without saying why just moves the
judgment call to whoever reads the alert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# PSI > 0.2 is the standard credit-risk-modelling line for "material" drift;
# 0.1-0.2 is "moderate", < 0.1 is noise. We alert only on the material line.
PSI_THRESHOLD = 0.2
PSI_BINS = 10

# The per-tech verdict a forecast cycle writes to `gold/drift` and /health reads
# back. Owned here rather than by the script, so the producer and the two
# consumers (the cycle's log line, the health endpoint) agree by construction.
DRIFT_COLUMNS: list[str] = [
    "tech",
    "n_features_drifted",
    "drifted_features",
    "max_psi",
    "n_scored_hours",
    "recent_rmse_cf",
    "baseline_rmse_cf",
    "relative_increase",
    "retrain",
    "reason",
]

# error is "materially worse" than the backtest baseline once recent RMSE
# exceeds it by this relative margin.
ERROR_DEGRADATION_FRAC = 0.20

# ...and only once there is enough of it to mean anything. One week of hours.
# Measured here: a single cycle contributes ~22 scored wind hours, and that
# sample put recent RMSE 27% above a baseline computed over 31,091 rows -- which
# is sampling noise wearing a retrain recommendation. A monitor that fires on
# one bad afternoon is a monitor somebody switches off.
MIN_SCORED_HOURS = 168


def _psi(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two 1-D distributions.

    Bin edges come from the reference distribution's quantiles so each
    reference bin starts with ~equal mass; `current` is scored against those
    same edges. Empty bins on either side are floored rather than allowed to
    blow up as log(0) or divide-by-zero.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size < bins or cur.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0  # reference is (near-)constant -- no meaningful bins to compare
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_frac = np.maximum(ref_counts / ref_counts.sum(), 1e-6)
    cur_frac = np.maximum(cur_counts / cur_counts.sum(), 1e-6)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


# Columns that are not model INPUTS: the targets, the TSO's own forecast, demand,
# identifiers and bookkeeping. Drift in a target is not input drift, and a PSI on
# an id column is noise with a number attached.
_NON_INPUT_PREFIXES = ("y_", "tso_", "sample_weight", "qc_ok", "demand_")
_NON_INPUT = frozenset(
    {"region_id", "run_ts_utc", "valid_ts_utc", "lead_hours", "forecast_vintage", "is_day"}
)


def comparable_features(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """Numeric input columns present in BOTH frames.

    Derived rather than hardcoded: the training matrix grows columns as the
    feature set does, and a hand-maintained list would monitor last month's
    inputs while silently ignoring the new ones -- the exact failure a drift
    monitor exists to prevent.
    """
    return sorted(
        c
        for c in reference.columns
        if c in current.columns
        and c not in _NON_INPUT
        and not c.startswith(_NON_INPUT_PREFIXES)
        and is_numeric_dtype(reference[c])
        and is_numeric_dtype(current[c])
    )


def seasonal_window(
    frame: pd.DataFrame, around: pd.Series, half_width_days: int = 21, ts_col: str = "valid_ts_utc"
) -> pd.DataFrame:
    """Restrict `frame` to the same calendar period as `around`, in any year.

    Measured on this dataset: a September current window against the full
    training history flags 12 features -- temperature, pressure, humidity and
    wind direction, at all three NWP models. Every one of them is the calendar,
    not the fleet. The reference spans two and a half years of every season, so
    a one-month window is guaranteed to sit off-centre in it, and a monitor that
    fires every autumn is a monitor somebody switches off.

    Comparing September against previous Septembers asks the question actually
    worth asking: is THIS September unlike the ones the model was fitted on.
    """
    if frame.empty or around.empty:
        return frame
    ref_doy = pd.to_datetime(frame[ts_col], utc=True).dt.dayofyear
    target = float(pd.to_datetime(around, utc=True).dt.dayofyear.median())
    # circular distance in days, so a window spanning New Year still matches
    delta = (ref_doy - target + 182.5) % 365.0 - 182.5
    return frame[delta.abs() <= half_width_days]


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    threshold: float = PSI_THRESHOLD,
) -> pd.DataFrame:
    """Per-feature Population Stability Index, reference vs. current window.

    Returns one row per feature: `feature`, `psi`, `drifted` (psi > threshold,
    NaN psi counts as not-drifted -- an untestable feature isn't evidence of
    anything).
    """
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            rows.append({"feature": f, "psi": float("nan"), "drifted": False})
            continue
        psi = _psi(reference[f], current[f])
        rows.append(
            {"feature": f, "psi": psi, "drifted": bool(np.isfinite(psi) and psi > threshold)}
        )
    return pd.DataFrame(rows, columns=["feature", "psi", "drifted"])


def error_drift(
    recent_errors: pd.Series,
    baseline_rmse: float,
    degradation_frac: float = ERROR_DEGRADATION_FRAC,
    min_scored_hours: int = MIN_SCORED_HOURS,
) -> dict:
    """Is recent error materially worse than the backtest baseline?

    `recent_errors` is a series of signed or absolute residuals (same units as
    `baseline_rmse`, e.g. capacity factor); RMSE is computed here so the
    caller doesn't have to pre-aggregate.

    `relative_increase` is always reported -- it is useful to watch a trend --
    but `degraded` stays False below `min_scored_hours`, so a thin sample never
    escalates into a retrain recommendation on its own.
    """
    errs = pd.to_numeric(recent_errors, errors="coerce").dropna().to_numpy(dtype=float)
    recent_rmse = float(np.sqrt(np.mean(errs**2))) if errs.size else float("nan")
    out = {
        "n": int(errs.size),
        "recent_rmse": recent_rmse,
        "baseline_rmse": float(baseline_rmse),
        "relative_increase": float("nan"),
        "degraded": False,
        "underpowered": bool(errs.size < min_scored_hours),
    }
    if not np.isfinite(recent_rmse) or baseline_rmse <= 0:
        return out
    out["relative_increase"] = (recent_rmse - baseline_rmse) / baseline_rmse
    out["degraded"] = bool(
        out["relative_increase"] > degradation_frac and errs.size >= min_scored_hours
    )
    return out


# Columns whose drift is DESIGNED FOR, not a warning. Belgium's installed
# capacity grew steadily across the window (`cap_solar_mw` PSI 10.7 on a
# same-season comparison), and the model trains on CAPACITY FACTOR precisely so
# that fleet growth is absorbed rather than learned. Treating it as a retrain
# signal is a false alarm, and a monitor that cries wolf gets switched off.
EXPECTED_DRIFT_PREFIXES = ("cap_", "monitored_capacity")


def is_expected_drift(feature: str) -> bool:
    return feature.startswith(EXPECTED_DRIFT_PREFIXES)


def should_retrain(
    drift_table: pd.DataFrame,
    error_result: dict,
    min_drifted_features: int = 1,
) -> tuple[bool, str]:
    """Combine feature drift and error drift into one decision + reason.

    Error degradation alone is sufficient (it is the thing that costs money).
    Feature drift alone only triggers a recommendation once at least
    `min_drifted_features` UNEXPECTED features have crossed the line -- one noisy
    feature is not a fleet change, and capacity growth is not drift at all.
    """
    if len(drift_table):
        flagged = drift_table.loc[drift_table["drifted"], "feature"].tolist()
    else:
        flagged = []
    expected = [f for f in flagged if is_expected_drift(f)]
    drifted_names = [f for f in flagged if not is_expected_drift(f)]
    n_drifted = len(drifted_names)
    degraded = bool(error_result.get("degraded", False))

    if degraded and n_drifted:
        pct = error_result["relative_increase"] * 100
        return True, (
            f"error degraded {pct:.0f}% above baseline AND {n_drifted} feature(s) drifted "
            f"({', '.join(drifted_names)}) -- retrain"
        )
    if degraded:
        pct = error_result["relative_increase"] * 100
        return True, f"error degraded {pct:.0f}% above baseline RMSE -- retrain"
    if error_result.get("underpowered") and np.isfinite(
        error_result.get("relative_increase", float("nan"))
    ):
        pct = error_result["relative_increase"] * 100
        watching = (
            f"recent error is {pct:+.0f}% vs baseline on only {error_result.get('n', 0)} "
            f"scored hours -- too few to act on"
        )
    else:
        watching = ""

    if n_drifted >= min_drifted_features:
        return True, (
            f"{n_drifted} feature(s) drifted ({', '.join(drifted_names)}) with no error "
            "degradation yet -- retrain pre-emptively or keep monitoring"
            + (f"; {watching}" if watching else "")
        )
    if expected:
        return False, (
            f"only expected drift ({', '.join(expected)}) -- installed capacity grew, which "
            "training on capacity factor absorbs by design; no retrain needed"
        )
    if watching:
        return False, f"no material feature drift; {watching}"
    return False, "no material feature or error drift detected"
