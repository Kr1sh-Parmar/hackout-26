"""Tier-2: Open-Meteo Historical Forecast weather for the India transfer region.

Grid points are DERIVED from CEA_1x1_gridded_installed_{solar,wind}_cap.nc rather
than guessed -- top capacity cells, weighted by installed MW, per technology.

Window is 2021-01-01..2022-10-31: the overlap between the Historical Forecast
archive (most non-ECMWF models start 2021-22) and the Zenodo Indian series,
which ends 2022-10-31. Using historical FORECAST here, not ERA5, keeps the
India path free of the train/serve skew that reanalysis introduces.
"""
from __future__ import annotations
import json, pathlib, sys, time
import pandas as pd, requests, xarray as xr
from om_quota import fetch_json, DailyQuotaExhausted

RAWI = pathlib.Path("data/raw/india")
OUT = pathlib.Path("data/raw/openmeteo/india"); OUT.mkdir(parents=True, exist_ok=True)
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
START, END = "2021-01-01", "2022-10-31"
TOP_N = 6

# Trimmed to what the solar/wind physics actually consumes. Open-Meteo weights a
# call by (variables x days) and India is 36 calls, so dropping 9 unused variables
# (snow, visibility, rain, dew point, msl pressure, 10m direction, terrestrial and
# direct radiation) cuts the quota cost ~36% for no loss of modelling signal.
# Snow/visibility in particular carry nothing for an Indian fleet.
HOURLY = ("temperature_2m,relative_humidity_2m,surface_pressure,"
          "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
          "shortwave_radiation,diffuse_radiation,direct_normal_irradiance,"
          "wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m,"
          "precipitation,is_day")


def grid_points() -> pd.DataFrame:
    """Top-N capacity cells per tech, union'd, with per-tech capacity weights."""
    frames = {}
    for tech in ("solar", "wind"):
        ds = xr.open_dataset(RAWI / f"CEA_1x1_gridded_installed_{tech}_cap.nc")
        d = (ds["__xarray_dataarray_variable__"].to_dataframe(name="cap")
             .reset_index().dropna())
        d = d[d.cap > 0].nlargest(TOP_N, "cap")
        d[f"weight_{tech}"] = d.cap / d.cap.sum()
        frames[tech] = d[["latitude", "longitude", f"weight_{tech}"]]

    g = frames["solar"].merge(frames["wind"], on=["latitude", "longitude"], how="outer")
    g[["weight_solar", "weight_wind"]] = g[["weight_solar", "weight_wind"]].fillna(0.0)
    g["grid_point_id"] = ["in_%02d_%02d" % (r.latitude, r.longitude) for r in g.itertuples()]
    return g.sort_values("grid_point_id").reset_index(drop=True)


def fetch(row, model) -> bool:
    dest = OUT / f"{row.grid_point_id}_{model}.parquet"
    if dest.exists():
        print(f"  skip {dest.name}"); return True
    j = fetch_json(URL, {
        "latitude": float(row.latitude), "longitude": float(row.longitude),
        "start_date": START, "end_date": END,
        "hourly": HOURLY, "models": model,
        "wind_speed_unit": "ms", "timezone": "UTC",
    }, f"{row.grid_point_id}/{model}")
    if j is None:
        return False
    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["grid_point_id"] = row.grid_point_id
    df["weight_solar"], df["weight_wind"] = row.weight_solar, row.weight_wind
    df["nwp_model"] = model
    df["latitude"], df["longitude"] = j["latitude"], j["longitude"]
    df["elevation_m"] = j.get("elevation")
    df.to_parquet(dest, index=False)
    print(f"  {dest.name:32} {len(df):>7,} rows", flush=True)
    return True


if __name__ == "__main__":
    g = grid_points()
    (OUT / "grid_points.json").write_text(json.dumps(g.to_dict("records"), indent=2))
    print(f"{len(g)} india grid points (top-{TOP_N} solar + top-{TOP_N} wind cells)")
    failed = []
    try:
        for row in g.itertuples():
            for model in MODELS:
                if not fetch(row, model):
                    failed.append(f"{row.grid_point_id}/{model}")
    except DailyQuotaExhausted as e:
        print("DAILY QUOTA EXHAUSTED -- stopping rather than spinning.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("Resumable: re-run after UTC midnight; completed files are skipped.",
              file=sys.stderr)
        sys.exit(2)
    print("FAILED:", failed or "none")
    sys.exit(1 if failed else 0)
