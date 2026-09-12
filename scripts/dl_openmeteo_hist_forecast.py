"""Tier-1: Open-Meteo Historical FORECAST API -- archived past forecasts.

NOT the reanalysis archive. Training on reanalysis and serving on forecasts is
train/serve skew (data.md 5). Same variable names both sides is the whole point.

wind_speed_unit=ms is non-negotiable: the API defaults to km/h and power goes
as v**3, so a missed conversion is a ~47x error in MW -- silently.
"""

from __future__ import annotations
import itertools
import pathlib
import sys
import time
import pandas as pd
import requests

OUT = pathlib.Path("data/raw/openmeteo/hist_forecast")
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# capacity-weighted sample points; a single centroid misses frontal systems
GRID = [
    ("flanders_w", 51.05, 3.72, 0.30),
    ("antwerp", 51.22, 4.40, 0.25),
    ("brussels", 50.85, 4.35, 0.15),
    ("liege", 50.63, 5.57, 0.18),
    ("namur", 50.46, 4.87, 0.12),
]

HOURLY = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,pressure_msl,"
    "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
    "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
    "terrestrial_radiation,wind_speed_10m,wind_speed_100m,wind_direction_10m,"
    "wind_direction_100m,wind_gusts_10m,precipitation,rain,snowfall,snow_depth,"
    "visibility,is_day"
)

MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
START, END = "2023-01-01", "2026-09-10"


def fetch(name, lat, lon, w, model) -> bool:
    dest = OUT / f"{name}_{model}.parquet"
    if dest.exists():
        print(f"  skip {dest.name}")
        return True
    for attempt in (1, 2, 3):
        try:
            r = requests.get(
                URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": START,
                    "end_date": END,
                    "hourly": HOURLY,
                    "models": model,
                    "wind_speed_unit": "ms",  # non-negotiable
                    "timezone": "UTC",
                },
                timeout=300,
            )
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame(j["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["grid_point_id"], df["weight"], df["nwp_model"] = name, w, model
            df["latitude"], df["longitude"] = j["latitude"], j["longitude"]
            df["elevation_m"] = j.get("elevation")
            df.to_parquet(dest, index=False)
            nn = df["shortwave_radiation"].notna().sum()
            print(f"  {dest.name:34} {len(df):>7,} rows  ghi_notna={nn:,}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} {name}/{model}: {e}", file=sys.stderr)
            time.sleep(20 * attempt)
    return False


if __name__ == "__main__":
    failed = []
    for (name, lat, lon, w), model in itertools.product(GRID, MODELS):
        if not fetch(name, lat, lon, w, model):
            failed.append(f"{name}/{model}")
        time.sleep(1.2)  # be polite to the free tier
    print("\nFAILED:", failed or "none")
    sys.exit(1 if failed else 0)
