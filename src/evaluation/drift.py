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

# PSI > 0.2 is the standard credit-risk-modelling line for "material" drift;
# 0.1-0.2 is "moderate", < 0.1 is noise. We alert only on the material line.
PSI_THRESHOLD = 0.2
PSI_BINS = 10

# error is "materially worse" than the backtest baseline once recent RMSE
# exceeds it by this relative margin.
ERROR_DEGRADATION_FRAC = 0.20


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
) -> dict:
    """Is recent error materially worse than the backtest baseline?

    `recent_errors` is a series of signed or absolute residuals (same units as
    `baseline_rmse`, e.g. capacity factor); RMSE is computed here so the
    caller doesn't have to pre-aggregate.
    """
    errs = pd.to_numeric(recent_errors, errors="coerce").dropna().to_numpy(dtype=float)
    recent_rmse = float(np.sqrt(np.mean(errs**2))) if errs.size else float("nan")
    if not np.isfinite(recent_rmse) or baseline_rmse <= 0:
        return {
            "recent_rmse": recent_rmse,
            "baseline_rmse": float(baseline_rmse),
            "relative_increase": float("nan"),
            "degraded": False,
        }
    relative_increase = (recent_rmse - baseline_rmse) / baseline_rmse
    return {
        "recent_rmse": recent_rmse,
        "baseline_rmse": float(baseline_rmse),
        "relative_increase": float(relative_increase),
        "degraded": bool(relative_increase > degradation_frac),
    }


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
    if n_drifted >= min_drifted_features:
        return True, (
            f"{n_drifted} feature(s) drifted ({', '.join(drifted_names)}) with no error "
            "degradation yet -- retrain pre-emptively or keep monitoring"
        )
    if expected:
        return False, (
            f"only expected drift ({', '.join(expected)}) -- installed capacity grew, which "
            "training on capacity factor absorbs by design; no retrain needed"
        )
    return False, "no material feature or error drift detected"
