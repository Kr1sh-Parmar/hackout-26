"""Guards for rungs 6/7 (`src/models/deep_seq.py`).

The deep rungs are a benchmark, not a served model, but a benchmark that leaks
is worse than no benchmark: it publishes a number the platform cannot deliver.
These five tests hold the reshape, the leakage rule, the quantile ordering, the
seed contract and the train-only standardiser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import MIN_LAG_HOURS
from src.models import deep_seq

LEADS = deep_seq.LEADS


def _frame(n_runs: int = 8, ragged: bool = True) -> pd.DataFrame:
    """`n_runs` daily 00:00 UTC runs over 24..72 h, one deliberately ragged."""
    rows = []
    for i in range(n_runs):
        run = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)
        leads = list(LEADS)
        if ragged and i == 2:
            leads = leads[3:-5]  # a run that is missing both ends
        for lh in leads:
            rows.append(
                {
                    "region_id": "BE",
                    "run_ts_utc": run,
                    "lead_hours": lh,
                    "valid_ts_utc": run + pd.Timedelta(hours=int(lh)),
                }
            )
    return pd.DataFrame(rows)


def _features(frame: pd.DataFrame, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(frame), 6)).astype(np.float32)


def _target(frame: pd.DataFrame, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        0.1 * np.sin(frame["lead_hours"].to_numpy() / 7) + 0.02 * rng.normal(len(frame))
    ).astype(float)


def _fit_once(frame, x, y, seed: int = 0):
    xs, mask, pos = deep_seq.to_sequences(frame, x)
    ys, _, _ = deep_seq.to_sequences(frame, y)
    model = deep_seq.fit(
        xs,
        ys[:, :, 0],
        mask,
        seed=seed,
        run_ts=deep_seq.run_timestamps(frame),
        epochs=6,
        patience=6,
        threads=1,
    )
    return model, xs, pos


# --------------------------------------------------------------------------- #
def test_reshape_round_trip():
    """Flat -> sequences -> flat is the identity, and pads never come back."""
    f = _frame()
    # carry valid_ts_utc through as a value column so misordering is visible
    vals = np.column_stack(
        [
            # hours since the first valid hour: float32 carries this exactly,
            # epoch seconds would not and the test would be measuring dtype
            (f["valid_ts_utc"] - f["valid_ts_utc"].min()).dt.total_seconds().to_numpy() / 3600.0,
            f["lead_hours"].to_numpy(dtype=float),
            np.arange(len(f), dtype=float),
        ]
    )
    x, mask, pos = deep_seq.to_sequences(f, vals)

    assert x.shape == (f["run_ts_utc"].nunique(), len(LEADS), 3)
    n_pad = int((pos < 0).sum())
    assert n_pad == x.shape[0] * len(LEADS) - len(f) > 0, "the ragged run produced no pads"
    assert mask[pos < 0].sum() == 0, "a padded slot carries weight"

    back = deep_seq.from_sequences(x, pos, len(f))
    assert len(back) == len(f), "a padded slot was emitted as a row"
    assert np.isfinite(back).all()
    np.testing.assert_allclose(back[:, 2], np.arange(len(f)))  # exact input row order
    np.testing.assert_allclose(back[:, 0], vals[:, 0], rtol=0, atol=1e-3)  # valid_ts preserved
    np.testing.assert_allclose(back[:, 1], vals[:, 1])


def test_no_lag_shorter_than_24h_can_reach_the_deep_model():
    """The leakage rule, at the only place it can be enforced: the feature list.

    Every feature this model sees is for a valid hour AFTER the run was issued --
    that is what a 24-72 h forecast IS -- so "does a future value change the
    prediction" cannot be the question. The question is whether any feature
    encodes an observation the live system will not have at issue time, and the
    lag family is the only place that can happen.
    """
    from src.features.build import FEATURE_COLUMNS

    lags = [c for c in FEATURE_COLUMNS if "lag" in c and c.endswith("h")]
    assert set(lags) == {"power_lag_24h", "power_lag_168h"}, lags
    assert all(int(c.rsplit("_", 1)[1][:-1]) >= MIN_LAG_HOURS for c in lags)


def test_one_runs_prediction_cannot_depend_on_another_run():
    """Poison run 5's features; every OTHER run's prediction must not move.

    The runs share a forward pass, so anything that pooled across the batch
    (a batch norm, a cross-run statistic) would quietly let a later run inform
    an earlier one -- a leak that no per-row feature audit would catch.
    """
    f = _frame()
    x, y = _features(f), _target(f)
    model, xs, _ = _fit_once(f, x, y)

    clean = deep_seq.predict_sequences(model, xs)
    poisoned = xs.copy()
    poisoned[5] = 99.0
    dirty = deep_seq.predict_sequences(model, poisoned)

    assert not np.allclose(clean[5], dirty[5]), "the poison did not land; test is vacuous"
    others = [i for i in range(len(xs)) if i != 5]
    np.testing.assert_allclose(clean[others], dirty[others], rtol=0, atol=0)


def test_quantiles_are_sorted():
    f = _frame()
    x = _features(f)
    xs, mask, pos = deep_seq.to_sequences(f, x)
    import torch

    torch.manual_seed(0)
    model = deep_seq.DeepSeq(xs.shape[-1])  # random init: crossing is likely
    band = deep_seq.predict_deep(
        [model], xs, None, pos, np.full(len(f), 0.3), 1000.0, index=f.index
    )
    assert (band["p10_mw"] <= band["p50_mw"]).all()
    assert (band["p50_mw"] <= band["p90_mw"]).all()
    assert ((band["p10_cf"] >= 0) & (band["p90_cf"] <= 1)).all()


def test_same_seed_same_numbers():
    f = _frame()
    x, y = _features(f), _target(f)
    m1, xs, pos = _fit_once(f, x, y, seed=3)
    m2, _, _ = _fit_once(f, x, y, seed=3)
    np.testing.assert_allclose(
        deep_seq.predict_sequences(m1, xs), deep_seq.predict_sequences(m2, xs), rtol=0, atol=0
    )


def test_standardiser_is_fitted_on_train_only():
    """Mutating calibrate/test rows must not touch the stored statistics."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 4))
    train[:50, 1] = np.nan
    train[:, 3] = np.nan  # 100% NaN -> dropped
    cols = ["a", "b", "c", "d"]

    std = deep_seq.Standardiser().fit(train, cols)
    snapshot = (std.mean.copy(), std.std.copy(), std.median.copy(), list(std.columns))

    later = rng.normal(size=(400, 4)) * 1000 + 5000
    std.transform(later)
    assert np.array_equal(std.mean, snapshot[0])
    assert np.array_equal(std.std, snapshot[1])
    assert np.array_equal(std.median, snapshot[2])
    assert std.columns == snapshot[3]
    assert "b_isna" in std.columns and "d" not in std.columns

    out = std.transform(train)
    assert np.isfinite(out).all(), "a NaN survived the standardiser"
    assert abs(out[:, :3].mean()) < 0.5

    with pytest.raises(ValueError, match="NaN on train"):
        mostly_missing = train.copy()
        mostly_missing[:380, 0] = np.nan
        deep_seq.Standardiser(max_nan=0.5).fit(mostly_missing, cols)
