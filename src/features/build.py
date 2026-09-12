"""CONTRACT M3 -- feature construction.

`build_features` is a PURE function: no I/O, no globals, no config reads, no
clock. That is not a style preference. The most common production failure in
forecasting systems is training and serving computing features differently -- a
different fill rule, a different timezone, a lag from a different origin. One
pure function removes that class of bug structurally rather than by discipline,
and `tests/integration/test_feature_parity.py` holds it in place.

Owner: ML / Physics.  Consumers: models/, evaluation/.
"""

from __future__ import annotations

import pandas as pd

from ..core.config import SiteMaster

# Index of the returned frame. Never collapse these two into one: a forecast is
# identified by the run that produced it AND the hour it describes.
FEATURE_INDEX = ["run_ts_utc", "valid_ts_utc"]

# Columns every downstream stage may rely on. Extend deliberately -- adding a
# feature here is a contract change, and the pandera schema moves with it.
FEATURE_COLUMNS: list[str] = [
    # provenance / horizon
    "lead_hours",
    "tech",
    "capacity_mw",
    # solar physics (dev-03 §2)
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "solar_elevation_deg",
    "airmass",
    "is_day",
    "clearsky_ghi_wm2",
    "clearsky_index_kt",
    "poa_global_wm2",
    "cell_temperature_c",
    "thermal_derate",
    "physics_pac_mw",
    "clipping_headroom",
    "mins_since_sunrise",
    "mins_to_sunset",
    # wind physics (dev-03 §3)
    "ws_hub_ms",
    "air_density_kgm3",
    "ws_density_corrected_ms",
    "power_curve_cf",
    "physics_power_mw",
    "dP_dv",
    "turbulence_proxy",
    "wind_dir_sin",
    "wind_dir_cos",
    "below_cutin_flag",
    "above_cutout_flag",
    # NWP quality -- these predict our own error, and are what turn a point
    # forecast into an honest interval
    "ghi_model_disagreement",
    "ws_model_disagreement",
    "ghi_model_range",
    "nwp_ghi_ramp",
    "nwp_ws_ramp",
    "nwp_bias_lag_7d",
    # temporal
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    # autoregressive -- 24 h MINIMUM. At lead hour 48 `power_lag_1h` does not
    # exist; any shorter lag scores brilliantly in a backtest and fails in
    # deployment, because the backtest leaked what the live system never has.
    "power_lag_24h",
    "power_lag_168h",
    "roll_mean_24h",
    "roll_std_24h",
    "smart_persistence_cf",
]

MIN_LAG_HOURS = 24


def build_features(
    weather: pd.DataFrame,
    site: SiteMaster,
    actuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the model matrix for one archetype.

    Args:
        weather: Layer-0 `weather_nwp` rows for this region. Must carry
            `run_ts_utc`, `valid_ts_utc`, `lead_hours`, `nwp_model` and the
            canonical unit-suffixed weather columns.
        site: one resolved archetype (see `core.config.site_master`).
        actuals: generation history for the lag family. `None` at cold start --
            and at serving time, where future actuals do not exist. Lags are
            emitted as NaN in that case; LightGBM handles missing natively.

    Returns:
        DataFrame indexed by (run_ts_utc, valid_ts_utc) with FEATURE_COLUMNS.
        Solar columns are NaN for wind sites and vice versa.

    Raises:
        ValueError: if a lag shorter than MIN_LAG_HOURS is requested.
    """
    raise NotImplementedError("M3 -- see dev-03 §2-§5")


def empty_features() -> pd.DataFrame:
    """The contract's shape, for building against before M3 lands."""
    idx = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], tz="UTC", name=FEATURE_INDEX[0]),
            pd.DatetimeIndex([], tz="UTC", name=FEATURE_INDEX[1]),
        ]
    )
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in FEATURE_COLUMNS}, index=idx)
