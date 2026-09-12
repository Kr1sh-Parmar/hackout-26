"""India is the TRANSFER region: it proves the pipeline moves to a new place on
configuration alone, and it must never imply an accuracy claim it cannot back.

No Indian open dataset gives metered, technology-separated regional generation
overlapping the lead-stratified NWP archive, so there is no label to fit a
residual model to and none to calibrate a conformal band against. These tests
pin the two things that follow from that: the forecast still serves, and every
row of it says the band is uncalibrated.
"""

from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from src.core.config import load_region

RUN_TS = pd.Timestamp("2026-09-12T00:00:00Z")


def _weather(hours: int = 48) -> pd.DataFrame:
    """A regional weather frame shaped like the one `ingest.live` produces:
    `run_ts_utc` and `valid_ts_utc` as COLUMNS, one row per lead hour."""
    valid = pd.date_range(RUN_TS + pd.Timedelta("1h"), periods=hours, freq="h", tz="UTC")
    ramp = np.clip(np.sin(((np.arange(hours) % 24) - 6) / 12 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "region_id": "IN",
            "run_ts_utc": RUN_TS,
            "valid_ts_utc": valid,
            "lead_hours": np.arange(1, hours + 1, dtype=float),
            "ghi_wm2": 800 * ramp,
            "dni_wm2": 600 * ramp,
            "dhi_wm2": 180 * ramp,
            "temperature_2m_c": 30.0,
            "wind_speed_10m_ms": 4.0,
            "wind_speed_100m_ms": 7.0,
            "wind_direction_100m_deg": 210.0,
            "surface_pressure_hpa": 1005.0,
            "relative_humidity_2m_pct": 55.0,
            "wind_gusts_10m_ms": 11.0,
            "cloud_cover_pct": 30.0,
            "cloud_cover_low_pct": 15.0,
            "cloud_cover_mid_pct": 10.0,
            "cloud_cover_high_pct": 20.0,
            "dew_point_2m_c": 20.0,
            "precipitation_mm": 0.0,
            "ghi_wm2_model_std": 40.0,
            "ghi_wm2_model_range": 90.0,
            "wind_speed_100m_ms_model_std": 0.8,
        }
    )


def test_india_is_declared_physics_only_and_belgium_is_not():
    """Declared in config, not inferred from a missing artifact: "no model file"
    has to stay a loud error for a region that is supposed to have one."""
    assert load_region("IN").physics_only is True
    assert load_region("BE").physics_only is False


def test_a_missing_model_still_raises_for_a_region_that_should_have_one(monkeypatch):
    """`physics_only` must be the ONLY way into the unmodelled path. If a lost
    artifact silently degraded Belgium to physics, the platform would keep
    serving and quietly stop being the thing it was measured as."""
    from src.models import predict as predict_mod

    def _gone(*a, **k):
        raise FileNotFoundError("artifact deleted")

    monkeypatch.setattr(predict_mod, "load_model", _gone)
    with pytest.raises(RuntimeError, match="no model registered"):
        predict_mod.predict("BE", pd.Timestamp("2026-09-12T00:00:00Z"), weather=_weather())


def test_the_transfer_region_config_is_loadable_and_self_consistent():
    """The validators that matter for a hand-written region: archetype shares
    sum to 1 per technology, grid weights sum to 1, both technologies present."""
    cfg = load_region("IN")

    assert set(cfg.capacity_mw) == {"solar", "wind"}
    assert cfg.timezone == "Asia/Kolkata"
    assert sum(p.weight for p in cfg.weather_grid) == pytest.approx(1.0)
    for tech, arcs in cfg.archetypes.items():
        assert sum(a.share for a in arcs) == pytest.approx(1.0), tech
    # India's fleet is entirely onshore -- an offshore archetype here would be
    # a copied-from-Belgium mistake, not a modelling choice.
    assert all(a.id.startswith("onshore") for a in cfg.archetypes["wind"])


def test_physics_only_rows_are_never_marked_calibrated(monkeypatch):
    """The serving contract for a region with no labels. An uncalibrated band
    presented as calibrated is the one output here that could mislead an
    operator sizing a reserve."""

    from src.models import predict as predict_mod

    monkeypatch.setattr(predict_mod, "load_actuals", lambda *a, **k: None)

    fc = predict_mod.predict("IN", RUN_TS, weather=_weather())

    assert not fc.empty
    assert set(fc.tech.unique()) == {"solar", "wind"}
    assert not fc.calibrated.any(), "a physics-only region has nothing to calibrate against"
    assert (fc.model_version == "physics-only").all()
    assert (fc.p10_mw <= fc.p50_mw).all() and (fc.p50_mw <= fc.p90_mw).all()
    assert (fc.p50_mw <= fc.capacity_mw).all()
    assert fc[fc.tech == "solar"].p50_mw.max() > 0, "daylight hours must produce something"
