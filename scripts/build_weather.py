"""Open-Meteo raw -> bronze -> silver/weather_nwp.

Two sources, unified into one table keyed by
(run_ts_utc, valid_ts_utc, nwp_model, grid_point_id):

  forecast_vintage='day0_best'  from the Historical Forecast API. Open-Meteo
      serves the most recent forecast for each hour, so lead is ~0-23 h. run_ts
      is APPROXIMATED as that day's 00Z run -- flagged by run_ts_is_approx.
  forecast_vintage='prev_day1|2|3'  from the Previous Runs API. run_ts is real:
      the run from N days earlier, giving lead bands 24-47 / 48-71 / 72-95 h.

Only the prev_day rows are legitimate training data for a 24-72 h model. The
day0 rows are kept for bias/skill reference, never as the 24-72 h training set.
"""

from __future__ import annotations
import glob
import pathlib
import pandas as pd

BRONZE = pathlib.Path("data/bronze")
SILVER = pathlib.Path("data/silver")
REGION_ID = "BE"

RENAME = {
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
META = ["grid_point_id", "weight", "nwp_model", "latitude", "longitude", "elevation_m"]


def _write(df, layer, name):
    dest = layer / name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote {layer.name}/{name:24} {len(df):>9,} rows  {len(df.columns)} cols")


def _load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["valid_ts_utc"] = pd.to_datetime(df["time"], utc=True)
    return df.drop(columns=["time"])


def build_day0() -> pd.DataFrame:
    df = _load("data/raw/openmeteo/hist_forecast/*.parquet")
    _write(df, BRONZE, "weather_hist_forecast")
    out = df.rename(columns=RENAME)
    keep = ["valid_ts_utc"] + META + [v for v in RENAME.values() if v in out.columns]
    out = out[keep].copy()
    out["forecast_vintage"] = "day0_best"
    out["run_ts_utc"] = out.valid_ts_utc.dt.normalize()  # that day's 00Z run
    out["run_ts_is_approx"] = True
    return out


# Canonical previous-run variable set. The first 7 files were fetched with a wider
# list before the quota forced a trim; restricting to this intersection keeps the
# silver schema uniform across all 15 without re-downloading the wide ones.
CORE_PREV = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
]


def build_prev() -> pd.DataFrame:
    df = _load("data/raw/openmeteo/previous_runs/*.parquet")
    if df.empty:
        print("  (no previous_runs files yet -- skipping)")
        return pd.DataFrame()
    _write(df, BRONZE, "weather_previous_runs")

    frames = []
    for n in (1, 2, 3):
        suf = f"_previous_day{n}"
        cols = {f"{b}{suf}": b for b in CORE_PREV if f"{b}{suf}" in df.columns}
        if not cols:
            continue
        sub = df[["valid_ts_utc", *META, *cols]].rename(columns=cols)
        sub = sub.rename(columns=RENAME)
        sub["forecast_vintage"] = f"prev_day{n}"
        # the run from N days earlier, at 00Z
        sub["run_ts_utc"] = sub.valid_ts_utc.dt.normalize() - pd.Timedelta(days=n)
        sub["run_ts_is_approx"] = False
        # drop rows with no forecast at all for this vintage
        wx = [c for c in RENAME.values() if c in sub.columns and c != "is_day"]
        sub = sub[sub[wx].notna().any(axis=1)]
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parts = [p for p in (build_day0(), build_prev()) if not p.empty]
    out = pd.concat(parts, ignore_index=True)

    out.insert(0, "region_id", REGION_ID)
    out["lead_hours"] = (
        (out.valid_ts_utc - out.run_ts_utc).dt.total_seconds().div(3600).round().astype("int32")
    )
    # is_day is geometric -- it depends on valid time and location only, never on
    # which forecast run produced the row. The prev-run pull omits it, so fill it
    # from the day0 rows at the same (valid_ts, grid_point). Leaving it NaN->0 would
    # silently mark every 24-72 h row as night and erase the whole solar training set.
    if "is_day" in out:
        src = (
            out.loc[
                out.forecast_vintage == "day0_best", ["valid_ts_utc", "grid_point_id", "is_day"]
            ]
            .dropna()
            .drop_duplicates(["valid_ts_utc", "grid_point_id"])
        )
        lut = src.set_index(["valid_ts_utc", "grid_point_id"])["is_day"]
        idx = pd.MultiIndex.from_arrays([out.valid_ts_utc, out.grid_point_id])
        out["is_day"] = out["is_day"].fillna(pd.Series(lut.reindex(idx).values, index=out.index))
        missing = int(out.is_day.isna().sum())
        if missing:
            print(f"  WARNING is_day still null on {missing:,} rows")
        out["is_day"] = out["is_day"].fillna(0).astype("int8")

    key = ["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"]
    dup = out.duplicated(key).sum()
    if dup:
        print(f"  WARNING {dup:,} duplicate key rows -> keeping last")
        out = out.drop_duplicates(key, keep="last")

    # drop columns that are entirely null across every model
    empty = [c for c in out.columns if out[c].notna().sum() == 0]
    if empty:
        print("  dropping all-null columns:", empty)
        out = out.drop(columns=empty)

    front = [
        "region_id",
        "run_ts_utc",
        "valid_ts_utc",
        "lead_hours",
        "forecast_vintage",
        "run_ts_is_approx",
        "nwp_model",
        "grid_point_id",
        "weight",
    ]
    out = out[front + [c for c in out.columns if c not in front]]
    out = out.sort_values(key).reset_index(drop=True)
    _write(out, SILVER, "weather_nwp")

    print("\n  vintage / lead-hour coverage:")
    g = out.groupby("forecast_vintage").agg(
        n=("valid_ts_utc", "size"),
        lead_min=("lead_hours", "min"),
        lead_max=("lead_hours", "max"),
        t0=("valid_ts_utc", "min"),
        t1=("valid_ts_utc", "max"),
    )
    print(g.to_string())


if __name__ == "__main__":
    main()
