"""SHAP attribution for one forecast hour: "why is this forecast low?"."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.explain import explain
from src.models.residual_gbdt import train_residual

FAST = dict(n_estimators=40, num_leaves=15, min_child_samples=5, verbose=-1)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 600
    x = pd.DataFrame(
        {
            "clearsky_index_kt": rng.uniform(0, 1, n),
            "ghi_model_disagreement": rng.uniform(0, 200, n),
            "hour_sin": rng.uniform(-1, 1, n),
            "power_lag_24h": rng.uniform(0, 1, n),
            "roll_std_24h": rng.uniform(0, 0.3, n),
            "nwp_ghi_ramp": rng.normal(0, 50, n),
            "tech": "solar",
        }
    )
    physics = pd.Series(rng.uniform(0.1, 0.6, n))
    # the residual genuinely depends on cloudiness and on NWP disagreement, so
    # the attribution has something real to find
    y = physics + 0.3 * (x["clearsky_index_kt"] - 0.5) - 0.001 * x["ghi_model_disagreement"]
    models = train_residual(x, y, physics, pd.Series(np.ones(n)), params=FAST)
    return models, x


def test_returns_five_signed_finite_drivers(fitted):
    models, x = fitted
    out = explain(models, x.iloc[[0]])
    drivers = out["drivers"]

    assert x.shape[1] - 1 > 5, "fixture must have more features than the top-5 cut"
    assert len(drivers) == 5
    assert all(np.isfinite(d["contribution"]) for d in drivers)
    assert all(np.isfinite(d["value"]) or np.isnan(d["value"]) for d in drivers)
    assert np.isfinite(out["base_cf"]) and np.isfinite(out["prediction_cf"])


def test_drivers_are_ranked_by_absolute_contribution(fitted):
    models, x = fitted
    mags = [abs(d["contribution"]) for d in explain(models, x.iloc[[3]])["drivers"]]
    assert mags == sorted(mags, reverse=True)


def test_contributions_are_signed_both_ways_across_rows(fitted):
    """An attribution that only ever pushes one way is not an attribution."""
    models, x = fitted
    signs = {
        np.sign(d["contribution"])
        for i in range(20)
        for d in explain(models, x.iloc[[i]])["drivers"]
    }
    assert {-1.0, 1.0} <= signs


def test_contributions_reconstruct_the_prediction(fitted):
    """SHAP values are additive by construction; if they do not sum back to the
    prediction the explanation is decoration, not attribution."""
    models, x = fitted
    out = explain(models, x.iloc[[7]], top_n=x.shape[1])  # every feature
    total = out["base_cf"] + sum(d["contribution"] for d in out["drivers"])
    assert total == pytest.approx(out["prediction_cf"], abs=1e-6)


def test_explaining_more_than_one_hour_is_refused(fitted):
    models, x = fitted
    with pytest.raises(ValueError, match="one forecast hour"):
        explain(models, x.iloc[:2])
