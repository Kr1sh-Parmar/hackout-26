"""Properties that must hold for every input, not just the ones we thought of."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.config import load_region, site_master
from src.features.wind import power_curve, wind_features
from src.models.residual_gbdt import predict_residual

FINITE = dict(allow_nan=False, allow_infinity=False)


class _ConstantHead:
    """A quantile head that always predicts the same residual."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


@given(
    physics=st.floats(min_value=0.0, max_value=1.0, **FINITE),
    r=st.lists(st.floats(min_value=-2.0, max_value=2.0, **FINITE), min_size=3, max_size=3),
    cap=st.floats(min_value=1.0, max_value=50_000.0, **FINITE),
)
@settings(max_examples=200, deadline=None)
def test_sorted_quantiles_never_cross(physics, r, cap):
    """Independently fitted heads DO cross. The sort is what makes the interval
    an interval; without it p10 can exceed p50 and reserve is sized backwards."""
    x = pd.DataFrame({"f": [0.0]})
    models = {0.1: _ConstantHead(r[0]), 0.5: _ConstantHead(r[1]), 0.9: _ConstantHead(r[2])}
    out = predict_residual(models, x, pd.Series([physics]), cap)
    p10, p50, p90 = out["p10_mw"].iloc[0], out["p50_mw"].iloc[0], out["p90_mw"].iloc[0]
    assert p10 <= p50 <= p90
    assert 0.0 <= p10 and p90 <= cap + 1e-9


@given(v=st.floats(min_value=-5.0, max_value=80.0, **FINITE))
@settings(max_examples=300, deadline=None)
def test_power_curve_stays_in_the_unit_interval(v):
    assert 0.0 <= power_curve(v) <= 1.0


@given(
    ws=st.floats(min_value=0.0, max_value=60.0, **FINITE),
    temp=st.floats(min_value=-40.0, max_value=55.0, **FINITE),
    pressure=st.floats(min_value=870.0, max_value=1085.0, **FINITE),
    direction=st.floats(min_value=0.0, max_value=360.0, **FINITE),
)
@settings(max_examples=150, deadline=None)
def test_physics_power_is_bounded_by_capacity(ws, temp, pressure, direction):
    site = site_master(load_region("BE"), "wind")[0]
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    wx = pd.DataFrame(
        {
            "wind_speed_100m_ms": [ws],
            "wind_speed_10m_ms": [ws * 0.75],
            "wind_gusts_10m_ms": [ws * 1.4],
            "temperature_2m_c": [temp],
            "surface_pressure_hpa": [pressure],
            "wind_direction_100m_deg": [direction],
        },
        index=idx,
    )
    p = wind_features(wx, site)["physics_power_mw"].iloc[0]
    assert 0.0 <= p <= site.capacity_mw
