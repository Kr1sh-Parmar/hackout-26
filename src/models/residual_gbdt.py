"""Rung 2: gradient-boosted quantile correction on the PHYSICS RESIDUAL.

The model learns `y_cf - physics_cf`, never `y_cf`. Learning the raw target
makes the model re-derive the solar geometry from scratch, badly, and lose it
the moment the weather leaves the training distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

PARAMS = dict(
    objective="quantile",
    metric="quantile",
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
)

# `tech` is a label, not a feature; the model is per-tech already.
DROP_COLUMNS = ("tech",)


def design_matrix(x: pd.DataFrame) -> pd.DataFrame:
    """One place that decides what the model sees, so train and serve agree."""
    return x.drop(columns=[c for c in DROP_COLUMNS if c in x.columns]).astype(float)


def train_residual(
    x: pd.DataFrame,
    y_cf: pd.Series,
    physics_cf: pd.Series,
    sample_weight: pd.Series,
    quantiles: tuple[float, ...] = QUANTILES,
) -> dict[float, LGBMRegressor]:
    xm = design_matrix(x)
    residual = np.asarray(y_cf, dtype=float) - np.asarray(physics_cf, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    # Censored labels carry zero weight, not a dropped row: keeping the row keeps
    # the time axis intact for anything that indexes by position downstream.
    keep = np.isfinite(residual) & np.isfinite(w) & (w > 0)
    if keep.sum() == 0:
        raise ValueError("no usable training rows: every sample weight is zero or the target null")

    return {
        q: LGBMRegressor(alpha=q, **PARAMS).fit(xm.loc[keep], residual[keep], sample_weight=w[keep])
        for q in quantiles
    }


def predict_residual(
    models: dict[float, LGBMRegressor],
    x: pd.DataFrame,
    physics_cf: pd.Series,
    capacity_mw: float | pd.Series,
) -> pd.DataFrame:
    """Physics + residual quantiles -> MW, sorted.

    Independently fitted quantile heads cross. Sorting is not cosmetic: an
    interval with p10 above p50 sizes reserve backwards.
    """
    xm = design_matrix(x)
    base = np.asarray(physics_cf, dtype=float)
    qs = sorted(models)
    cf = np.column_stack([base + models[q].predict(xm) for q in qs])
    cf = np.clip(np.sort(cf, axis=1), 0.0, 1.0)

    cap = np.asarray(capacity_mw, dtype=float)
    out = pd.DataFrame(
        {f"p{int(round(q * 100))}_mw": cf[:, i] * cap for i, q in enumerate(qs)},
        index=x.index,
    )
    for i, q in enumerate(qs):
        out[f"p{int(round(q * 100))}_cf"] = cf[:, i]
    return out
