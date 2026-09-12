"""Rung 2: gradient-boosted quantile correction on the PHYSICS RESIDUAL.

The model learns `y_cf - physics_cf`, never `y_cf`. Learning the raw target
makes the model re-derive the solar geometry from scratch, badly, and lose it
the moment the weather leaves the training distribution.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import lightgbm as lgb
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

TUNING_DIR = pathlib.Path("artifacts/tuning")

# `tech` is a label, not a feature; the model is per-tech already.
DROP_COLUMNS = ("tech",)


def load_params(region: str, tech: str) -> tuple[dict, str]:
    """Tuned hyperparameters if `scripts/tune.py` has written any, else defaults.

    Returns `(params, source)`; the source string goes into the artifact metadata
    so "which hyperparameters produced this model?" is answerable later.
    """
    path = TUNING_DIR / f"{region}_{tech}.json"
    if not path.exists():
        return dict(PARAMS), "defaults"
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {**PARAMS, **blob.get("params", {})}, str(path)


def design_matrix(x: pd.DataFrame) -> pd.DataFrame:
    """One place that decides what the model sees, so train and serve agree.

    NaN stays NaN. LightGBM treats missing as its own branch; a fillna(0) here
    would tell the model "the plant produced nothing" every time a lag was simply
    not observed, which is a different and much worse statement.
    """
    return x.drop(columns=[c for c in DROP_COLUMNS if c in x.columns]).astype(float)


def fitted_features(models: dict[float, LGBMRegressor]) -> list[str] | None:
    """The exact feature list and ORDER each head was fitted on.

    None for a duck-typed head that was never fitted through LightGBM -- it
    carries no schema, so there is nothing to check it against.
    """
    names = getattr(next(iter(models.values())), "feature_name_", None)
    return list(names) if names is not None else None


def _align(models: dict[float, LGBMRegressor], xm: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the fitted feature order, or fail loudly.

    Column drift is silent: LightGBM will happily score a frame whose columns
    moved, and the forecast is then wrong in a way no test catches. Positional
    feature meaning is not something to leave to whoever last edited
    FEATURE_COLUMNS.
    """
    want = fitted_features(models)
    if want is None or list(xm.columns) == want:
        return xm
    missing = [c for c in want if c not in xm.columns]
    extra = [c for c in xm.columns if c not in want]
    if missing or extra:
        raise ValueError(
            f"feature set drifted since fit: missing={missing} unexpected={extra}. "
            "Refit the model or fix the feature builder; do not score through this."
        )
    return xm[want]


def _chronological_tail(x: pd.DataFrame, n_keep: int) -> np.ndarray:
    """Positions of the LAST `n_keep` rows in time, for the validation slice.

    A random slice would put the same hour on both sides of the fence and make
    early stopping decide on rows it has effectively already seen.
    """
    if "valid_ts_utc" in (x.index.names or []):
        order = np.argsort(x.index.get_level_values("valid_ts_utc").to_numpy(), kind="stable")
    else:
        order = np.arange(len(x))
    return order[-n_keep:]


def train_residual(
    x: pd.DataFrame,
    y_cf: pd.Series,
    physics_cf: pd.Series,
    sample_weight: pd.Series,
    quantiles: tuple[float, ...] = QUANTILES,
    *,
    params: dict | None = None,
    valid_frac: float = 0.0,
    early_stopping_rounds: int = 50,
    log_period: int = 0,
    on_head: Callable[[float], None] | None = None,
) -> dict[float, LGBMRegressor]:
    """Fit one quantile head per `quantiles` on the physics residual.

    `valid_frac > 0` carves that fraction off the CHRONOLOGICAL END of the
    training window as an early-stopping / logging validation slice.
    `log_period > 0` prints the validation loss every N iterations.
    """
    xm = design_matrix(x)
    residual = np.asarray(y_cf, dtype=float) - np.asarray(physics_cf, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    # Zero-weight rows are DROPPED, not passed with weight 0: identical fit,
    # less data through the histogram builder.
    keep = np.isfinite(residual) & np.isfinite(w) & (w > 0)
    if keep.sum() == 0:
        raise ValueError("no usable training rows: every sample weight is zero or the target null")

    xk, rk, wk = xm.loc[keep], residual[keep], w[keep]

    n_val = int(len(xk) * valid_frac) if valid_frac > 0 else 0
    fit_kw: dict = {}
    if n_val >= 50:
        val = _chronological_tail(xk, n_val)
        tr = np.setdiff1d(np.arange(len(xk)), val, assume_unique=False)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        if log_period > 0:
            callbacks.append(lgb.log_evaluation(period=log_period))
        fit_kw = dict(
            eval_set=[(xk.iloc[val], rk[val])],
            eval_sample_weight=[wk[val]],
            eval_names=["holdout"],
            callbacks=callbacks,
        )
        xk, rk, wk = xk.iloc[tr], rk[tr], wk[tr]

    models = {}
    for q in quantiles:
        models[q] = LGBMRegressor(alpha=q, **(params or PARAMS)).fit(
            xk, rk, sample_weight=wk, **fit_kw
        )
        if on_head is not None:
            on_head(q)
    return models


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
    xm = _align(models, design_matrix(x))
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
