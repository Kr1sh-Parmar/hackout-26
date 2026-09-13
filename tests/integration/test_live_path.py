"""The live serving path, with the network stubbed.

CI must never call a live external API, so the adapter is monkeypatched. What is
actually under test is the SHAPE CONTRACT: a live frame must be
indistinguishable from a gold row to everything downstream, because that is the
only reason a backtest number says anything about live performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_region
from src.ingest import live as live_mod
from src.ingest.adapters.openmeteo import RENAME, WeatherUnavailable
from src.ingest.live import assert_serving_shape, live_weather, run_timestamp

RUN = pd.Timestamp("2026-09-12 00:00", tz="UTC")


def _fake_region(cfg, forecast_days: int = 4) -> pd.DataFrame:
    """A plausible long frame: every point x model x hour, canonical names."""
    hours = pd.date_range(RUN, periods=96, freq="h", tz="UTC")
    rows = []
    rng = np.random.default_rng(0)
    for point in cfg.weather_grid:
        for model in cfg.nwp_models:
            df = pd.DataFrame({"valid_ts_utc": hours})
            # is_day first: the irradiance columns are derived from it
            df["is_day"] = (hours.hour.isin(range(6, 20))).astype(int)
            for canon in RENAME.values():
                if canon == "is_day":
                    continue
                if canon in ("ghi_wm2", "dni_wm2", "dhi_wm2", "direct_radiation_wm2"):
                    df[canon] = np.where(df["is_day"] == 1, rng.uniform(50, 700, len(hours)), 0.0)
                elif canon == "surface_pressure_hpa":
                    df[canon] = rng.uniform(990, 1020, len(hours))
                else:
                    df[canon] = rng.uniform(0, 20, len(hours))
            df["grid_point_id"] = point.id
            df["weight"] = point.weight
            df["nwp_model"] = model
            df["latitude"], df["longitude"], df["elevation_m"] = point.lat, point.lon, 50.0
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["region_id"] = cfg.region_id
    return out


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(live_mod, "fetch_region", _fake_region)
    return load_region("BE")


def test_live_frame_covers_the_horizon(stubbed):
    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    assert len(wx) == 72
    assert wx["lead_hours"].min() == 1
    assert wx["lead_hours"].max() == 72


def test_live_frame_never_describes_its_own_past(stubbed):
    """A forecast issued at run_ts cannot say anything about hours before it."""
    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    assert (wx["valid_ts_utc"] > RUN).all()


def test_live_frame_carries_every_column_the_model_needs(stubbed):
    """THE shape contract. If this fails, serving and training have diverged."""
    from tests.conftest import read_built_parquet

    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    gold = read_built_parquet("data/gold/training_base_24_72h/part-0.parquet").head(50)
    missing = assert_serving_shape(wx, gold)
    assert not missing, f"live frame is missing {missing}"


def test_live_frame_has_model_disagreement(stubbed):
    """Disagreement is a feature, not a diagnostic -- it must survive to serving."""
    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    assert "ghi_wm2_model_std" in wx.columns
    assert wx["ghi_wm2_model_std"].notna().any()


def test_live_frame_is_marked_as_live(stubbed):
    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    assert wx["forecast_vintage"].eq("live").all()
    assert wx["run_ts_is_live"].all(), "a live run must be distinguishable from replay"


def test_timestamps_are_utc_aware(stubbed):
    wx = live_weather(stubbed, run_ts=RUN, horizon_hours=72)
    for c in ("run_ts_utc", "valid_ts_utc"):
        assert str(wx[c].dtype) == "datetime64[ns, UTC]", c


def test_run_timestamp_floors_to_the_hour():
    ts = run_timestamp(pd.Timestamp("2026-09-12 09:47:31", tz="UTC"))
    assert ts == pd.Timestamp("2026-09-12 09:00", tz="UTC")


def test_naive_run_timestamp_is_localised():
    assert run_timestamp(pd.Timestamp("2026-09-12 09:00")).tz is not None


def test_weather_failure_raises_rather_than_returning_partial(monkeypatch):
    """A silent partial forecast is worse than a loud failure: an operator acts
    on whatever number appears."""

    def boom(cfg, forecast_days: int = 4):
        raise WeatherUnavailable("simulated outage")

    monkeypatch.setattr(live_mod, "fetch_region", boom)
    with pytest.raises(WeatherUnavailable):
        live_weather(load_region("BE"), run_ts=RUN)
