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

from .residual_gbdt import _align, design_matrix

TOP_N = 5
MEDIAN = 0.5


@lru_cache(maxsize=8)
def _explainer(model: LGBMRegressor) -> shap.TreeExplainer:
    """TreeExplainer setup costs more than the per-row call it enables."""
    return shap.TreeExplainer(model)


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
