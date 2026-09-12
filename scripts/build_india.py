"""India transfer region: Zenodo 7824872 -> bronze -> silver.

Caveat that must travel with every Indian number: the hourly series is
reanalysis-MODELLED, not metered. Scoring against it partly scores the model
that produced it. India demonstrates TRANSFER, not accuracy. The POSOCO daily
figures are the only genuine observations in the record.
"""
from __future__ import annotations
import pathlib
import pandas as pd, xarray as xr

RAW = pathlib.Path("data/raw/india")
BRONZE = pathlib.Path("data/bronze"); SILVER = pathlib.Path("data/silver")
REGION_ID = "IN"


def _write(df, layer, name):
    dest = layer / name; dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote {layer.name}/{name:30} {len(df):>9,} rows")


def build_modelled():
    ds = xr.open_dataset(RAW / "modelled-historical-hourly-renewable_output.nc")
    df = ds["modelled_grid_output"].to_dataframe().reset_index()
    df["ts_utc"] = pd.to_datetime(df["time"], utc=True)
    out = df[["ts_utc", "modelled_grid_output"]].rename(
        columns={"modelled_grid_output": "modelled_output"})
    out.insert(0, "region_id", REGION_ID)
    # single combined series -- NOT separable into solar vs wind
    out["tech"] = "combined"
    out["source"] = "zenodo_7824872_modelled"
    _write(out, BRONZE, "india_modelled_hourly")
    _write(out, SILVER, "india_modelled_hourly")
    print(f"    range {out.ts_utc.min()} -> {out.ts_utc.max()}")


def build_posoco():
    """Daily reported generation in MU (million units = GWh) by regional load
    despatch centre. The only genuine observations in this record."""
    frames = []
    for tech in ("solar", "wind"):
        d = pd.read_csv(RAW / f"POSOCO_reported_{tech}_MU_daily.csv", parse_dates=["Date"])
        long = d.melt(id_vars="Date", var_name="rldc", value_name="energy_gwh")
        long["tech"] = tech
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"Date": "date"})
    out["region_id"] = REGION_ID
    out["is_national_total"] = out.rldc.eq("Total")   # do not sum with the parts
    _write(out, SILVER, "india_posoco_daily")
    for t in ("solar", "wind"):
        s = out[(out.tech == t) & out.is_national_total]
        print(f"    {t}: {s.date.min().date()} -> {s.date.max().date()}  {len(s):,} days")


def build_capacity():
    inst = pd.read_csv(RAW / "installed-by-state-oct2022.csv")
    inst.columns = [c.strip() for c in inst.columns]
    _write(inst, SILVER, "india_installed_by_state")

    rows = []
    for tech in ("solar", "wind"):
        ds = xr.open_dataset(RAW / f"CEA_1x1_gridded_installed_{tech}_cap.nc")
        d = (ds["__xarray_dataarray_variable__"].to_dataframe(name="capacity_gw")
             .reset_index().dropna())
        d = d[d.capacity_gw > 0]; d["tech"] = tech
        rows.append(d)
    grid = pd.concat(rows, ignore_index=True)
    grid["region_id"] = REGION_ID
    _write(grid, SILVER, "india_gridded_capacity")
    print("    national totals (GW):",
          grid.groupby("tech").capacity_gw.sum().round(1).to_dict())

    byd = pd.read_csv(RAW / "tabulated-installed-by-date.csv")
    _write(byd, SILVER, "india_installed_by_date")


if __name__ == "__main__":
    print("== modelled hourly =="); build_modelled()
    print("== POSOCO daily ==");    build_posoco()
    print("== capacity ==");        build_capacity()
