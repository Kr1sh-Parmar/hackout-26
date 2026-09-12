"""Why is this forecast low? -- SHAP attribution for a single forecast hour.

Attribution on a RESIDUAL model is readable in a way attribution on a raw model
is not. A raw model spends its explanation budget re-deriving solar geometry
("zenith 62 deg", "it is 14:00"), which nobody needed explained. Here the
physics is already subtracted, so what is left is the correction itself: low
cloud, a disagreeing NWP, a persistently biased week.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from ..core.config import load_region
from ..features.build import build_all_features
from .registry import load_model
from .residual_gbdt import _align, design_matrix

TOP_N = 5
MEDIAN = 0.5

EXPLAIN_COLUMNS: list[str] = [
    "valid_ts_utc",
    "lead_hours",
    "tech",
    "rank",
    "feature",
    "value",
    "contribution",
    "base_cf",
    "prediction_cf",
]


@lru_cache(maxsize=8)
def _explainer(model: LGBMRegressor) -> shap.TreeExplainer:
    """TreeExplainer setup costs more than the per-row call it enables."""
    return shap.TreeExplainer(model)


def explain_run(
    region_id: str,
    run_ts: pd.Timestamp,
    weather: pd.DataFrame | None = None,
    horizons: range = range(1, 73),
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Attribution for a whole forecast cycle, one frame, every technology.

    Walks the same path `predict._forecast_tech` walks, so an explanation is
    always of the features that actually produced the served number.

    Restricted to leads at or above `MIN_TRAINED_LEAD_H`: below it the platform
    serves physics only, and attributing a residual correction that was never
    applied would explain a number nobody was shown.
    """
    from .predict import MIN_TRAINED_LEAD_H, load_actuals, load_run_weather

    cfg = load_region(region_id)
    run_ts = pd.Timestamp(run_ts)
    wx = load_run_weather(region_id, run_ts, horizons) if weather is None else weather

    frames = []
    for tech in sorted(cfg.capacity_mw):
        try:
            artifact = load_model(region_id, tech)
        except FileNotFoundError:
            continue
        x = build_all_features(region_id, wx, actuals=load_actuals(region_id, tech), tech=tech)
        x = x[x["lead_hours"] >= MIN_TRAINED_LEAD_H]
        frames.append(explain_frame(artifact["models"], x, tech, top_n=top_n))

    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in EXPLAIN_COLUMNS})
    return pd.concat(frames, ignore_index=True)


def explain(
    models: dict[float, LGBMRegressor],
    x_row: pd.DataFrame,
    top_n: int = TOP_N,
) -> dict:
    """Top drivers of the p50 residual correction for ONE forecast hour.

    Args:
        models: the fitted quantile heads from `train_residual`.
        x_row: exactly one row of the same feature frame `predict_residual` takes.
        top_n: how many drivers to return, ranked by absolute contribution.

    Returns:
        `base_cf` (the model's average residual), `prediction_cf` (this row's
        residual correction) and `drivers`: feature, its value, and its SIGNED
        contribution in capacity-factor units. Positive means this feature pushed
        the forecast ABOVE physics.
    """
    if len(x_row) != 1:
        raise ValueError(f"explain() attributes one forecast hour; got {len(x_row)} rows")
    head = models[MEDIAN] if MEDIAN in models else models[sorted(models)[len(models) // 2]]

    xm = _align(models, design_matrix(x_row))
    contrib = np.asarray(_explainer(head).shap_values(xm), dtype=float).reshape(-1)
    values = xm.to_numpy(dtype=float).reshape(-1)

    order = np.argsort(-np.abs(contrib))[:top_n]
    return {
        "base_cf": float(np.ravel(_explainer(head).expected_value)[0]),
        "prediction_cf": float(head.predict(xm)[0]),
        "drivers": [
            {
                "feature": str(xm.columns[i]),
                "value": float(values[i]),
                "contribution": float(contrib[i]),
            }
            for i in order
        ],
    }


def explain_frame(
    models: dict[float, LGBMRegressor],
    x: pd.DataFrame,
    tech: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Top drivers for EVERY row of a forecast horizon, as a tidy frame.

    One `shap_values` call over the whole horizon rather than `explain()` in a
    loop: TreeExplainer is vectorised, and the per-call setup is what costs.

    Returns `top_n` rows per forecast hour, `rank` 0 = largest absolute
    contribution. Empty in, empty out -- a horizon with no rows is a valid
    outcome, not an error.
    """
    if x.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in EXPLAIN_COLUMNS})

    head = models[MEDIAN] if MEDIAN in models else models[sorted(models)[len(models) // 2]]
    xm = _align(models, design_matrix(x))
    explainer = _explainer(head)

    contrib = np.asarray(explainer.shap_values(xm), dtype=float).reshape(len(xm), -1)
    values = xm.to_numpy(dtype=float)
    # argsort on the negated absolute value puts the biggest driver first; only
    # the leading top_n columns are kept, so a 44-feature attribution stays a
    # readable answer rather than a full table nobody reads.
    order = np.argsort(-np.abs(contrib), axis=1)[:, :top_n]

    idx = x.index
    rows = np.repeat(np.arange(len(xm)), order.shape[1])
    cols = order.reshape(-1)
    return pd.DataFrame(
        {
            "valid_ts_utc": idx.get_level_values("valid_ts_utc")[rows],
            "lead_hours": x["lead_hours"].to_numpy()[rows].astype("int32"),
            "tech": tech,
            "rank": np.tile(np.arange(order.shape[1]), len(xm)).astype("int32"),
            "feature": xm.columns.to_numpy()[cols],
            "value": values[rows, cols],
            "contribution": contrib[rows, cols],
            "base_cf": float(np.ravel(explainer.expected_value)[0]),
            "prediction_cf": head.predict(xm)[rows],
        }
    )[EXPLAIN_COLUMNS]
