"""Open-Meteo live forecast adapter.

Serving weather. The training archive is fetched by `scripts/dl_openmeteo_*.py`;
this is the forward-looking twin, and it deliberately requests the SAME variable
names so training and serving cannot diverge at the source.

`wind_speed_unit=ms` is the single most consequential parameter here: the API
defaults to km/h and power goes as v**3, so a missed conversion is a ~47x error
in megawatts that still looks like a plausible number.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# identical to the training pull (scripts/dl_openmeteo_hist_forecast.py)
HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,pressure_msl,"
    "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
    "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
    "terrestrial_radiation,wind_speed_10m,wind_speed_100m,wind_direction_10m,"
    "wind_direction_100m,wind_gusts_10m,precipitation,rain,snowfall,snow_depth,"
    "visibility,is_day"
)

# raw Open-Meteo name -> canonical unit-suffixed name. Must match build_weather.py.
RENAME: dict[str, str] = {
    "shortwave_radiation": "ghi_wm2",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
    "direct_radiation": "direct_radiation_wm2",
    "terrestrial_radiation": "terrestrial_radiation_wm2",
    "temperature_2m": "temperature_2m_c",
    "dew_point_2m": "dew_point_2m_c",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "surface_pressure": "surface_pressure_hpa",
    "pressure_msl": "pressure_msl_hpa",
    "cloud_cover": "cloud_cover_pct",
    "cloud_cover_low": "cloud_cover_low_pct",
    "cloud_cover_mid": "cloud_cover_mid_pct",
    "cloud_cover_high": "cloud_cover_high_pct",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_speed_100m": "wind_speed_100m_ms",
    "wind_direction_10m": "wind_direction_10m_deg",
    "wind_direction_100m": "wind_direction_100m_deg",
    "wind_gusts_10m": "wind_gusts_10m_ms",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "snow_depth": "snow_depth_m",
    "visibility": "visibility_m",
    "is_day": "is_day",
}


class WeatherUnavailable(RuntimeError):
    """Live weather could not be fetched. The caller should fall back, not guess."""


def fetch_point(
    lat: float,
    lon: float,
    model: str,
    forecast_days: int = 4,
    timeout: int = 120,
    tries: int = 3,
) -> pd.DataFrame:
    """One grid point, one NWP model, hourly, out to `forecast_days`.

    4 days covers the 72 h horizon with margin for the run-time offset.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "models": model,
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",  # non-negotiable; see module docstring
        "timezone": "UTC",
    }
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(FORECAST_URL, params=params, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame(j["hourly"])
            df["valid_ts_utc"] = pd.to_datetime(df.pop("time"), utc=True)
            df["latitude"] = j.get("latitude")
            df["longitude"] = j.get("longitude")
            df["elevation_m"] = j.get("elevation")
            return df.rename(columns=RENAME)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * attempt)
    raise WeatherUnavailable(f"{model} @ ({lat},{lon}) after {tries} tries: {last}") from last


def fetch_region(cfg, forecast_days: int = 4) -> pd.DataFrame:
    """Every grid point x every configured NWP model, in long form.

    Returns the shape `src.ingest.regional.regionalise` expects: one row per
    (valid_ts, nwp_model, grid_point) carrying the point's capacity `weight`.
    """
    frames: list[pd.DataFrame] = []
    for point in cfg.weather_grid:
        for model in cfg.nwp_models:
            df = fetch_point(point.lat, point.lon, model, forecast_days=forecast_days)
            df["grid_point_id"] = point.id
            df["weight"] = point.weight
            df["nwp_model"] = model
            frames.append(df)
    if not frames:
        raise WeatherUnavailable("no grid points configured")
    # align to the union of columns first: concatenating frames with differing
    # all-NA columns leaves pandas to guess dtypes and warn about it
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    out = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    out["region_id"] = cfg.region_id
    return out
