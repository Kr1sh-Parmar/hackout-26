"""OPSD (Germany, 4 TSO zones) -> bronze -> silver. Cross-check region only.

NOTE: this release (2020-10-06) contains NO per-TSO solar/wind FORECAST columns,
contrary to data.md 10 -- only load forecasts. Germany is therefore a second
generation cross-check region, not a second forecast benchmark. Elia remains the
only external generation-forecast benchmark in the project.
"""

from __future__ import annotations
import pathlib
import pandas as pd

RAW = pathlib.Path("data/raw/opsd")
BRONZE = pathlib.Path("data/bronze")
SILVER = pathlib.Path("data/silver")

ZONES = {
    "DE": "DE",
    "DE_50hertz": "DE-50HERTZ",
    "DE_amprion": "DE-AMPRION",
    "DE_tennet": "DE-TENNET",
    "DE_transnetbw": "DE-TRANSNETBW",
}


def _write(df, layer, name):
    dest = layer / name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote {layer.name}/{name:28} {len(df):>9,} rows")


def build(freq: str) -> None:
    f = RAW / f"time_series_{freq}_singleindex.csv"
    keep_prefix = tuple(ZONES) + ("utc_timestamp",)
    d = pd.read_csv(f, low_memory=False)
    cols = ["utc_timestamp"] + [
        c for c in d.columns if c.startswith(keep_prefix) and c != "utc_timestamp"
    ]
    d = d[cols]
    d["ts_utc"] = pd.to_datetime(d.utc_timestamp, utc=True, format="ISO8601")
    d = d.drop(columns=["utc_timestamp"])
    _write(d, BRONZE, f"opsd_{freq}")

    rows = []
    for zone, region_id in ZONES.items():
        for tech, gen, cap in [
            ("solar", f"{zone}_solar_generation_actual", f"{zone}_solar_capacity"),
            ("wind", f"{zone}_wind_generation_actual", f"{zone}_wind_capacity"),
            (
                "wind_onshore",
                f"{zone}_wind_onshore_generation_actual",
                f"{zone}_wind_onshore_capacity",
            ),
            (
                "wind_offshore",
                f"{zone}_wind_offshore_generation_actual",
                f"{zone}_wind_offshore_capacity",
            ),
        ]:
            if gen not in d.columns:
                continue
            r = pd.DataFrame(
                {
                    "region_id": region_id,
                    "ts_utc": d.ts_utc,
                    "tech": tech,
                    "power_mw": d[gen],
                    "monitored_capacity_mw": d[cap] if cap in d.columns else pd.NA,
                }
            ).dropna(subset=["power_mw"])
            rows.append(r)
    gen = pd.concat(rows, ignore_index=True).sort_values(["region_id", "tech", "ts_utc"])
    _write(gen, SILVER, f"opsd_generation_{freq}")

    ld = []
    for zone, region_id in ZONES.items():
        a, fc = (
            f"{zone}_load_actual_entsoe_transparency",
            f"{zone}_load_forecast_entsoe_transparency",
        )
        if a not in d.columns:
            continue
        ld.append(
            pd.DataFrame(
                {
                    "region_id": region_id,
                    "ts_utc": d.ts_utc,
                    "demand_mw": d[a],
                    "demand_forecast_mw": d[fc] if fc in d.columns else pd.NA,
                }
            ).dropna(subset=["demand_mw"])
        )
    _write(pd.concat(ld, ignore_index=True), SILVER, f"opsd_load_{freq}")


if __name__ == "__main__":
    for freq in ("60min", "15min"):
        print(f"== {freq} ==")
        build(freq)
