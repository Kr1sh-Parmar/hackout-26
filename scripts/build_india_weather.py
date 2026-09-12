"""India weather raw -> bronze -> silver/weather_nwp_india.

India uses the plain Historical Forecast API (day-0 vintage, ~0-23 h lead), NOT
the previous-runs archive, because the Zenodo Indian series ends 2022-10-31 and
the lead-stratified archive only starts 2024-03-05 -- they do not overlap.

Consequence, and it must travel with every Indian number: the India path can
demonstrate that the pipeline TRANSFERS to another region and fleet, but it
cannot demonstrate 24-72 h accuracy. Two separate caveats compound here --
day-0 lead, and a reanalysis-modelled (not metered) target.

Grid weights are per-technology (weight_solar / weight_wind), derived from the
CEA 1x1 installed-capacity raster, because India's solar and wind fleets sit in
different places -- unlike Belgium, where one weight serves both.
"""

from __future__ import annotations
import glob
import pathlib
import pandas as pd

BRONZE = pathlib.Path("data/bronze")
SILVER = pathlib.Path("data/silver")
REGION_ID = "IN"

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
META = [
    "grid_point_id",
    "weight_solar",
    "weight_wind",
    "nwp_model",
    "latitude",
    "longitude",
    "elevation_m",
]


def _write(df, layer, name):
    dest = layer / name
    dest.mkdir(parents=True, exist_ok=True)
    for c in ("valid_ts_utc", "run_ts_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote {layer.name}/{name:26} {len(df):>9,} rows  {len(df.columns)} cols")


def main() -> None:
    files = sorted(glob.glob("data/raw/openmeteo/india/*.parquet"))
    if not files:
        print("  no India weather files -- run scripts/dl_openmeteo_india.py first")
        return
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["valid_ts_utc"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"])
    _write(df, BRONZE, "weather_india")

    out = df.rename(columns=RENAME)
    keep = ["valid_ts_utc"] + META + [v for v in RENAME.values() if v in out.columns]
    out = out[keep].copy()
    out.insert(0, "region_id", REGION_ID)
    out["forecast_vintage"] = "day0_best"
    out["run_ts_utc"] = out.valid_ts_utc.dt.normalize()
    out["run_ts_is_approx"] = True
    out["lead_hours"] = (
        (out.valid_ts_utc - out.run_ts_utc).dt.total_seconds().div(3600).round().astype("int32")
    )
    if "is_day" in out:
        out["is_day"] = out["is_day"].fillna(0).astype("int8")

    key = ["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"]
    dup = int(out.duplicated(key).sum())
    if dup:
        print(f"  WARNING {dup:,} duplicate key rows -> keeping last")
        out = out.drop_duplicates(key, keep="last")
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
        "weight_solar",
        "weight_wind",
    ]
    out = out[front + [c for c in out.columns if c not in front]]
    _write(out.sort_values(key).reset_index(drop=True), SILVER, "weather_nwp_india")

    print(f"    {out.grid_point_id.nunique()} grid points x {out.nwp_model.nunique()} models")
    print(f"    {out.valid_ts_utc.min()} -> {out.valid_ts_utc.max()}")
    for t in ("solar", "wind"):
        w = out.drop_duplicates("grid_point_id")[f"weight_{t}"].sum()
        print(f"    weight_{t} sums to {w:.4f}")


if __name__ == "__main__":
    main()
