"""The shared spatial/model aggregation.

This function is the seam where train/serve skew would enter: the historical ETL
and the live fetch both call it, and if they ever stopped doing so the model
would be served weather computed differently from the weather it learned on --
a failure no backtest can show.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.regional import KEY, blend_models, capacity_weighted, regionalise


def _points(values: dict[str, list[float]], weights: list[float], model: str = "m1"):
    n = len(weights)
    ts = pd.Timestamp("2026-09-12 12:00", tz="UTC")
    return pd.DataFrame(
        {
            "run_ts_utc": ts,
            "valid_ts_utc": ts + pd.Timedelta(hours=24),
            "forecast_vintage": "live",
            "nwp_model": model,
            "grid_point_id": [f"p{i}" for i in range(n)],
            "weight": weights,
            **values,
        }
    )


def test_weighted_mean_is_capacity_weighted():
    wx = _points({"ghi_wm2": [100.0, 200.0]}, [0.25, 0.75])
    out = capacity_weighted(wx)
    assert out["ghi_wm2"].iloc[0] == pytest.approx(0.25 * 100 + 0.75 * 200)


def test_weights_renormalise_when_a_point_is_missing_a_variable():
    """A plain weighted sum would return a smaller regional value and look calm."""
    wx = _points({"ghi_wm2": [100.0, float("nan")]}, [0.25, 0.75])
    out = capacity_weighted(wx)
    # only the 0.25 point reported, so the regional value IS that point's value
    assert out["ghi_wm2"].iloc[0] == pytest.approx(100.0)


def test_blend_and_disagreement_across_models():
    a = _points({"ghi_wm2": [100.0]}, [1.0], model="ecmwf_ifs025")
    b = _points({"ghi_wm2": [200.0]}, [1.0], model="icon_seamless")
    reg = capacity_weighted(pd.concat([a, b], ignore_index=True))
    wide = blend_models(reg)

    assert wide["ghi_wm2"].iloc[0] == pytest.approx(150.0)
    assert wide["ghi_wm2_model_range"].iloc[0] == pytest.approx(100.0)
    assert wide["ghi_wm2_model_std"].iloc[0] > 0
    assert {"ghi_wm2__ecmwf_ifs025", "ghi_wm2__icon_seamless"} <= set(wide.columns)


def test_is_day_is_taken_not_averaged():
    """Geometric: the sun is up or it is not. Averaging would produce 0.5."""
    wx = _points({"ghi_wm2": [10.0, 20.0], "is_day": [1, 1]}, [0.5, 0.5])
    assert capacity_weighted(wx)["is_day"].iloc[0] == 1


def test_one_row_per_key():
    a = _points({"ghi_wm2": [1.0, 2.0]}, [0.5, 0.5], model="m1")
    b = _points({"ghi_wm2": [3.0, 4.0]}, [0.5, 0.5], model="m2")
    out = regionalise(pd.concat([a, b], ignore_index=True))
    assert len(out) == 1
    assert set(KEY) <= set(out.columns)


def test_historical_etl_and_live_call_the_same_function():
    """Guards the seam itself: if either stops importing `regionalise`, the two
    aggregations can drift silently."""
    import pathlib

    etl = pathlib.Path("scripts/build_gold.py").read_text(encoding="utf-8")
    live = pathlib.Path("src/ingest/live.py").read_text(encoding="utf-8")
    assert "regionalise" in etl, "build_gold.py must delegate to src.ingest.regional"
    assert "regionalise" in live, "live path must delegate to src.ingest.regional"
