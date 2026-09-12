"""Live forecast input: fetch today's weather and shape it exactly like training.

THE WHOLE POINT OF THIS MODULE is that the frame it returns is indistinguishable
in shape and semantics from a row of `gold/training_base_24_72h`. The feature
builder, the physics chain and the model then cannot tell whether they are
looking at history or at this morning -- which is the only way a backtest number
means anything about live performance.

One honest difference, and it is recorded in the data rather than hidden:
training weather comes from the Previous Runs archive (a forecast genuinely
issued 24-72 h earlier), while a live run necessarily uses the forecast issued
NOW. `forecast_vintage` says which, and `run_ts_is_live` marks it.
"""

from __future__ import annotations

import pandas as pd

from ..core.config import RegionConfig
from ..quality.schemas import weather_nwp_schema
from ..quality.validators import validate
from .adapters.openmeteo import fetch_region
from .regional import regionalise

HORIZON_HOURS = 72


def run_timestamp(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Floor to the hour, in UTC. A run is identified by when it was issued."""
    ts = pd.Timestamp(now or pd.Timestamp.utcnow())
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").floor("h")


def live_weather_with_points(
    cfg: RegionConfig,
    run_ts: pd.Timestamp | None = None,
    horizon_hours: int = HORIZON_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regional frame AND the per-grid-point rows behind it.

    The storm detector needs the points, not the regional mean: capacity-weighted
    averaging caps hub wind around 18 m/s against a 25-27 m/s cut-out, so the mean
    can never trigger the event the detector exists to catch.

    Raises:
        WeatherUnavailable: the network or the API failed. Callers should fall
            back to the last good run or to replay -- never to a silent default.
    """
    run_ts = run_timestamp(run_ts)
    raw = fetch_region(cfg)

    # only hours strictly after the run: a forecast cannot describe its own past
    raw = raw[raw["valid_ts_utc"] > run_ts].copy()
    raw["run_ts_utc"] = run_ts
    raw["forecast_vintage"] = "live"
    raw["lead_hours"] = (
        (raw["valid_ts_utc"] - raw["run_ts_utc"]).dt.total_seconds().div(3600).round().astype(int)
    )
    raw = raw[raw["lead_hours"].between(1, horizon_hours)]

    # The provider boundary. A live fetch is the one input nobody reviewed
    # before it reached the model, so the same contract the silver tables are
    # built against is applied to it here -- strict=False, because dropping a
    # bad hour and serving the rest beats refusing to forecast at all. Without
    # this, an Open-Meteo unit change or a renamed field arrives as a plausible
    # forecast rather than as an error.
    raw = validate(raw, weather_nwp_schema, "live_weather", strict=False)

    wide = regionalise(raw)
    wide["region_id"] = cfg.region_id
    wide["lead_hours"] = (
        (wide["valid_ts_utc"] - wide["run_ts_utc"]).dt.total_seconds().div(3600).round().astype(int)
    )
    wide = wide[wide["lead_hours"].between(1, horizon_hours)]

    # pandas 2 infers second resolution from a scalar Timestamp assign; gold is
    # nanosecond. Pin it so a live frame and a historical frame are the same dtype.
    for col in ("run_ts_utc", "valid_ts_utc"):
        wide[col] = pd.to_datetime(wide[col], utc=True).astype("datetime64[ns, UTC]")

    wide["run_ts_is_approx"] = False
    wide["run_ts_is_live"] = True
    if "is_day" in wide.columns:
        wide["is_day"] = wide["is_day"].fillna(0).astype("int8")

    front = [
        "region_id",
        "run_ts_utc",
        "valid_ts_utc",
        "lead_hours",
        "forecast_vintage",
        "run_ts_is_live",
    ]
    wide = wide[front + [c for c in wide.columns if c not in front]]
    return wide.sort_values("valid_ts_utc").reset_index(drop=True), raw.reset_index(drop=True)


def live_weather(
    cfg: RegionConfig,
    run_ts: pd.Timestamp | None = None,
    horizon_hours: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Regional weather for the next `horizon_hours`, shaped like the gold table."""
    return live_weather_with_points(cfg, run_ts, horizon_hours)[0]


def assert_serving_shape(live: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Columns the model needs that a live frame does not carry.

    Called before predicting so a schema gap fails loudly at the boundary rather
    than as a confusing NaN forecast three layers down.
    """
    needed = [
        c
        for c in reference.columns
        if not c.startswith(("y_", "cap_", "qc_ok_", "tso_", "sample_weight_", "demand_"))
    ]
    return [c for c in needed if c not in live.columns]
