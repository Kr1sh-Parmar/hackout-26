"""Tier-1 (ADDITION to data.md): Open-Meteo PREVIOUS RUNS archive.

WHY THIS EXISTS -- a gap in data.md 5.
The Historical Forecast API returns the best/most-recent forecast for each hour,
i.e. roughly 0-24 h lead time. This project forecasts at 24-72 h. Training on
~6 h-lead weather and serving at 48 h lead is train/serve skew: the same class of
bug the docs name for reanalysis, one level subtler.

The Previous Runs API returns, for each valid hour, what the forecast said N days
earlier. previous_day1/2/3 map onto lead bands 24-47 h / 48-71 h / 72-95 h, which
is exactly the 24-72 h horizon. That is what makes lead_hours a real feature and
the (run_ts, valid_ts) primary key meaningful.

Archive starts 2024-03-05 (verified), so this is shorter than the main pull.
"""
from __future__ import annotations
import itertools, pathlib, sys, time
import pandas as pd, requests
from om_quota import fetch_json, date_chunks, DailyQuotaExhausted

OUT = pathlib.Path("data/raw/openmeteo/previous_runs"); OUT.mkdir(parents=True, exist_ok=True)
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

GRID = [("flanders_w", 51.05, 3.72, .30), ("antwerp", 51.22, 4.40, .25),
        ("brussels",   50.85, 4.35, .15), ("liege",   50.63, 5.57, .18),
        ("namur",      50.46, 4.87, .12)]
MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
START, END = "2024-03-05", "2026-09-10"
PREV_DAYS = (1, 2, 3)

# Open-Meteo weights a call by (variables x time span). 80 vars x 2.5 years blows
# the hourly quota, so request ONLY the _previous_dayN variants -- the day0 values
# are already covered by the Historical Forecast pull -- and only the variables the
# solar/wind physics actually consumes. This list is the canonical prev-run set and
# must match CORE_PREV in build_weather.py.
BASE = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "wind_speed_10m", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m"]
HOURLY = [f"{b}_previous_day{n}" for b in BASE for n in PREV_DAYS]


def fetch(name, lat, lon, w, model) -> bool:
    dest = OUT / f"{name}_{model}.parquet"
    if dest.exists():
        print(f"  skip {dest.name}"); return True

    # fetch in ~200-day chunks so each request is small enough to clear the bucket
    parts = []
    for a, b in date_chunks(START, END, days=200):
        j = fetch_json(URL, {
            "latitude": lat, "longitude": lon,
            "start_date": a, "end_date": b,
            "hourly": ",".join(HOURLY), "models": model,
            "wind_speed_unit": "ms", "timezone": "UTC",
        }, f"{name}/{model} {a}..{b}")
        if j is None:
            return False
        parts.append(pd.DataFrame(j["hourly"]))
        meta = j

    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["grid_point_id"], df["weight"], df["nwp_model"] = name, w, model
    df["latitude"], df["longitude"] = meta["latitude"], meta["longitude"]
    df["elevation_m"] = meta.get("elevation")
    df.to_parquet(dest, index=False)
    live = df["shortwave_radiation_previous_day2"].notna().sum()
    print(f"  {dest.name:34} {len(df):>7,} rows  prev_day2_notna={live:,}", flush=True)
    return True


if __name__ == "__main__":
    print(f"{len(HOURLY)} variables x {len(GRID)} points x {len(MODELS)} models")
    failed = []
    try:
        for (name, lat, lon, w), model in itertools.product(GRID, MODELS):
            if not fetch(name, lat, lon, w, model):
                failed.append(f"{name}/{model}")
    except DailyQuotaExhausted as e:
        print("DAILY QUOTA EXHAUSTED -- stopping rather than spinning.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("Resumable: re-run after UTC midnight; completed files are skipped.",
              file=sys.stderr)
        sys.exit(2)
    print("FAILED:", failed or "none")
    sys.exit(1 if failed else 0)
