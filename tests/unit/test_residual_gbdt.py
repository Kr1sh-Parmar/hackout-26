"""The residual model's non-negotiables: sorted bands, honest NaNs, no drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.residual_gbdt import (
    design_matrix,
    fitted_features,
    predict_residual,
    train_residual,
)

FAST = dict(n_estimators=30, num_leaves=7, min_child_samples=5, verbose=-1)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_arrays(
        [
            pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            pd.date_range("2024-01-02", periods=n, freq="h", tz="UTC"),
        ],
        names=["run_ts_utc", "valid_ts_utc"],
    )
    x = pd.DataFrame(
        {
            "ghi": rng.uniform(0, 900, n),
            "wind": rng.uniform(0, 20, n),
            "power_lag_24h": rng.uniform(0, 1, n),
            "tech": "solar",
        },
        index=idx,
    )
    physics = pd.Series(rng.uniform(0.1, 0.6, n))
    y = physics.to_numpy() + rng.normal(0, 0.05, n)
    return x, pd.Series(y), physics


def _fit(x, y, physics, w=None, **kw):
    w = pd.Series(np.ones(len(x))) if w is None else w
    return train_residual(x, y, physics, w, params=FAST, **kw)


def test_quantiles_never_cross():
    """Independently fitted heads DO cross. An interval with p10 above p50 sizes
    reserve backwards, which is the expensive direction to be wrong in."""
    x, y, physics = _frame()
    out = predict_residual(_fit(x, y, physics), x, physics, capacity_mw=1000.0)
    assert (out["p10_mw"] <= out["p50_mw"] + 1e-9).all()
    assert (out["p50_mw"] <= out["p90_mw"] + 1e-9).all()
    assert (out[["p10_cf", "p50_cf", "p90_cf"]].to_numpy() >= 0).all()


def test_nan_features_reach_the_model_as_nan():
    """A NaN lag means "not observed"; 0 means "the plant produced nothing".
    Conflating them teaches the model that every cold start was a blackout."""
    x, y, physics = _frame()
    x = x.copy()
    x.loc[x.index[:100], "power_lag_24h"] = np.nan

    xm = design_matrix(x)
    assert xm["power_lag_24h"].isna().sum() == 100, "design_matrix silently filled NaN"

    models = _fit(x, y, physics)
    # the booster split on missing rather than on a fabricated zero
    assert models[0.5].predict(xm).shape == (len(x),)
    assert np.isfinite(models[0.5].predict(xm)).all()


def test_an_all_nan_feature_does_not_become_a_zero_column():
    x, y, physics = _frame()
    x = x.copy()
    x["never_observed"] = np.nan
    assert design_matrix(x)["never_observed"].isna().all()
    assert "never_observed" in fitted_features(_fit(x, y, physics))


def test_feature_drift_at_predict_time_raises():
    """Column drift is silent otherwise: LightGBM scores by position and the
    forecast is wrong in a way no downstream assertion catches."""
    x, y, physics = _frame()
    models = _fit(x, y, physics)

    renamed = x.rename(columns={"ghi": "ghi_wm2"})
    with pytest.raises(ValueError, match="drifted"):
        predict_residual(models, renamed, physics, 1000.0)

    with pytest.raises(ValueError, match="drifted"):
        predict_residual(models, x.drop(columns=["wind"]), physics, 1000.0)


def test_reordered_columns_are_realigned_not_scored_by_position():
    x, y, physics = _frame()
    models = _fit(x, y, physics)
    straight = predict_residual(models, x, physics, 1000.0)
    shuffled = x[["power_lag_24h", "tech", "wind", "ghi"]]
    assert np.allclose(
        straight["p50_mw"].to_numpy(),
        predict_residual(models, shuffled, physics, 1000.0)["p50_mw"].to_numpy(),
    )


def test_zero_weight_rows_are_excluded_from_the_fit():
    """Dropped, not passed with weight 0 -- same fit, less work. Proven by
    making the zero-weight half wildly inconsistent: it must not move anything."""
    x, y, physics = _frame()
    w = pd.Series(np.where(np.arange(len(x)) < len(x) // 2, 1.0, 0.0))

    poisoned = y.copy()
    poisoned[len(x) // 2 :] = 50.0  # nonsense, but zero-weighted
    a = _fit(x, y, physics, w=w)
    b = _fit(x, poisoned, physics, w=w)
    assert np.allclose(a[0.5].predict(design_matrix(x)), b[0.5].predict(design_matrix(x)))


def test_every_weight_zero_is_an_error_not_an_empty_model():
    x, y, physics = _frame()
    with pytest.raises(ValueError, match="no usable training rows"):
        _fit(x, y, physics, w=pd.Series(np.zeros(len(x))))


def test_early_stopping_uses_a_chronological_tail_and_shortens_the_fit():
    x, y, physics = _frame(n=2000)
    models = _fit(x, y, physics, valid_frac=0.2, early_stopping_rounds=5)
    # it stopped somewhere, and the validation slice never entered the fit
    assert models[0.5].best_iteration_ is not None
    assert models[0.5]._Booster.num_trees() <= FAST["n_estimators"] * 3
