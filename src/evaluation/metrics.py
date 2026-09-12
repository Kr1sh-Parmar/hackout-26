"""Scoring primitives.

Every metric is normalised by INSTALLED CAPACITY, not by the mean. Mean-normalising
lets low-output periods dominate the score: a 1 MW error at 03:00 when output is
2 MW reads as 50% error, and a whole night of those drowns out a real afternoon
miss. Capacity-normalised numbers are also comparable across regions and across
a fleet that grows during the evaluation window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def _clean(y: ArrayLike, yhat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is missing. Never silently fill."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def nrmse(y: ArrayLike, yhat: ArrayLike, capacity: float) -> float:
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yhat - y) ** 2)) / capacity)


def nmae(y: ArrayLike, yhat: ArrayLike, capacity: float) -> float:
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yhat - y)) / capacity)


def mbe(y: ArrayLike, yhat: ArrayLike, capacity: float) -> float:
    """Mean bias, SIGNED. Positive means over-forecasting.

    This is the metric traders care about most: a symmetric RMSE can hide a
    systematic lean that costs money on every settlement interval.
    """
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return float("nan")
    return float(np.mean(yhat - y) / capacity)


def skill_score(y: ArrayLike, yhat: ArrayLike, y_ref: ArrayLike, capacity: float) -> float:
    """1 - RMSE_model / RMSE_reference. The headline number.

    Scale-free and comparable, but only as meaningful as the reference. See
    `baselines.persistence` -- a degenerate reference inflates this without
    the model improving at all.
    """
    a = nrmse(y, yhat, capacity)
    b = nrmse(y, y_ref, capacity)
    if not np.isfinite(a) or not np.isfinite(b) or b < 1e-12:
        return float("nan")
    return float(1.0 - a / b)


def pinball(y: ArrayLike, q: ArrayLike, alpha: float) -> float:
    """Quantile (pinball) loss at level alpha -- the GEFCom2014 metric."""
    y, q = _clean(y, q)
    if y.size == 0:
        return float("nan")
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def crps_from_quantiles(y: ArrayLike, qs: dict[float, ArrayLike]) -> float:
    """Approximate CRPS by averaging pinball loss across the available quantiles."""
    losses = [pinball(y, v, a) for a, v in sorted(qs.items())]
    losses = [x for x in losses if np.isfinite(x)]
    return float(np.mean(losses)) if losses else float("nan")


def picp(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Prediction interval coverage probability -- the honesty check.

    An 80% band that covers 55% of outcomes is not a conservative forecast; it is
    a false statement, and it will size reserves wrong.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return float("nan")
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


def ace(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, nominal: float = 0.80) -> float:
    """Average coverage error: observed PICP minus nominal. Target ~0."""
    return picp(y, lo, hi) - nominal


def interval_width(lo: ArrayLike, hi: ArrayLike, capacity: float) -> float:
    """Mean band width as a fraction of capacity. Sharpness, paired with PICP.

    Coverage alone is trivially gamed -- a band from 0 to capacity covers 100%.
    Always report width beside it.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return float("nan")
    return float(np.mean(hi[m] - lo[m]) / capacity)
