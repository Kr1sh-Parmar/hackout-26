"""silver -> gold/training_base : one analysis-ready row per (run_ts, valid_ts).

This is the model matrix MINUS derived physics features. Feature engineering
(clear-sky index, POA, power curve) belongs to build_features() in the ML
pipeline and is deliberately NOT done here -- that function is contracted to be
pure and to run identically in training and serving.

What this does do:
  * capacity-weighted spatial aggregation of the 5 grid points -> regional weather
  * per-NWP-model columns kept, plus the blend and the model disagreement, which
    is one of the strongest available predictors of our own forecast error
  * Elia 15-min -> hourly (the modelling resolution; NWP is hourly natively)
  * joins generation truth, demand and the Elia benchmark onto the same key
  * y is CAPACITY FACTOR; sample_weight is 0 on censored rows
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd

SILVER = pathlib.Path("data/silver"); GOLD = pathlib.Path("data/gold")
REGION_ID = "BE"

WX_VARS = ["ghi_wm2", "dni_wm2", "dhi_wm2", "direct_radiation_wm2",
           "temperature_2m_c", "dew_point_2m_c", "relative_humidity_2m_pct",
           "surface_pressure_hpa", "cloud_cover_pct", "cloud_cover_low_pct",
           "cloud_cover_mid_pct", "cloud_cover_high_pct",
           "wind_speed_10m_ms", "wind_speed_100m_ms", "wind_direction_100m_deg",
           "wind_gusts_10m_ms", "precipitation_mm"]
KEY = ["run_ts_utc", "valid_ts_utc", "forecast_vintage"]


def _write(df, name):
    dest = GOLD / name; dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote gold/{name:22} {len(df):>9,} rows  {len(df.columns)} cols")


def regional_weather() -> pd.DataFrame:
    """Capacity-weighted mean across grid points, renormalised over non-null
    weights so one missing point does not silently shrink the regional value."""
    wx = pd.read_parquet(SILVER / "weather_nwp" / "part-0.parquet")
    vars_ = [v for v in WX_VARS if v in wx.columns]
    g = KEY + ["nwp_model"]

    num = wx[g].copy()
    for v in vars_:
        num[v] = wx[v] * wx.weight
    wsum = wx[g].copy()
    for v in vars_:
        wsum[v] = wx.weight.where(wx[v].notna())

    agg_n = num.groupby(g, observed=True)[vars_].sum(min_count=1)
    agg_w = wsum.groupby(g, observed=True)[vars_].sum(min_count=1)
    reg = (agg_n / agg_w.replace(0, np.nan)).reset_index()

    # is_day is geometric, identical across points
    if "is_day" in wx.columns:
        day = wx.groupby(g, observed=True)["is_day"].max().reset_index()
        reg = reg.merge(day, on=g, how="left")

    # wide: one column block per NWP model
    wide = reg.pivot_table(index=KEY, columns="nwp_model", values=vars_, observed=True)
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    wide = wide.reset_index()

    # blend + disagreement: disagreement predicts our own error
    for v in vars_:
        cols = [c for c in wide.columns if c.startswith(f"{v}__")]
        if not cols:
            continue
        wide[v] = wide[cols].mean(axis=1)
        wide[f"{v}_model_std"] = wide[cols].std(axis=1)
        wide[f"{v}_model_range"] = wide[cols].max(axis=1) - wide[cols].min(axis=1)

    if "is_day" in reg.columns:
        d = reg.groupby(KEY, observed=True)["is_day"].max().reset_index()
        wide = wide.merge(d, on=KEY, how="left")
    return wide


def hourly_generation(tech: str) -> pd.DataFrame:
    """15-min -> hourly. An hour containing any non-OK interval is not clean."""
    g = pd.read_parquet(SILVER / f"generation_actuals_{tech}" / "part-0.parquet")
    g["hour"] = g.ts_utc.dt.floor("h")
    agg = g.groupby("hour").agg(
        power_mw=("power_mw", "mean"),
        monitored_capacity_mw=("monitored_capacity_mw", "mean"),
        n_intervals=("power_mw", "size"),
        n_ok=("qc_flag", lambda s: (s == "OK").sum()),
    ).reset_index().rename(columns={"hour": "valid_ts_utc"})
    agg["qc_ok"] = agg.n_ok == agg.n_intervals
    agg[f"y_{tech}_mw"] = agg.power_mw
    agg[f"cap_{tech}_mw"] = agg.monitored_capacity_mw
    agg[f"y_{tech}_cf"] = agg.power_mw / agg.monitored_capacity_mw
    agg[f"qc_ok_{tech}"] = agg.qc_ok
    return agg[["valid_ts_utc", f"y_{tech}_mw", f"cap_{tech}_mw",
                f"y_{tech}_cf", f"qc_ok_{tech}"]]


def hourly_tso(tech: str) -> pd.DataFrame:
    t = pd.read_parquet(SILVER / f"tso_forecast_{tech}" / "part-0.parquet")
    t = t[t.horizon == "day_ahead_6pm"]
    t["valid_ts_utc"] = t.ts_utc.dt.floor("h")
    a = t.groupby("valid_ts_utc").agg(
        **{f"tso_{tech}_p10_mw": ("p10_mw", "mean"),
           f"tso_{tech}_p50_mw": ("p50_mw", "mean"),
           f"tso_{tech}_p90_mw": ("p90_mw", "mean")}).reset_index()
    return a


def hourly_load() -> pd.DataFrame:
    l = pd.read_parquet(SILVER / "load" / "part-0.parquet")
    l["valid_ts_utc"] = l.ts_utc.dt.floor("h")
    return l.groupby("valid_ts_utc").agg(
        demand_mw=("demand_mw", "mean"),
        demand_da_mw=("demand_da_6pm_mw", "mean"),
        demand_da_p10_mw=("demand_da_p10_mw", "mean"),
        demand_da_p90_mw=("demand_da_p90_mw", "mean")).reset_index()


def main() -> None:
    print("== regional weather ==")
    base = regional_weather()
    print(f"  {len(base):,} (run,valid,vintage) rows")

    for tech in ("solar", "wind"):
        base = base.merge(hourly_generation(tech), on="valid_ts_utc", how="left")
        base = base.merge(hourly_tso(tech), on="valid_ts_utc", how="left")
    base = base.merge(hourly_load(), on="valid_ts_utc", how="left")

    base.insert(0, "region_id", REGION_ID)
    base["lead_hours"] = ((base.valid_ts_utc - base.run_ts_utc)
                          .dt.total_seconds().div(3600).round().astype("int32"))

    # censored labels carry no weight: curtailed / missing / frozen / out-of-range
    for tech in ("solar", "wind"):
        ok = base[f"qc_ok_{tech}"].fillna(False) & base[f"y_{tech}_cf"].notna()
        base[f"sample_weight_{tech}"] = ok.astype(float)

    front = ["region_id", "run_ts_utc", "valid_ts_utc", "lead_hours", "forecast_vintage"]
    base = base[front + [c for c in base.columns if c not in front]]
    base = base.sort_values(["run_ts_utc", "valid_ts_utc"]).reset_index(drop=True)
    _write(base, "training_base")

    train = base[base.forecast_vintage.str.startswith("prev_day")
                 & base.lead_hours.between(24, 72)]
    _write(train, "training_base_24_72h")

    print("\n  trainable rows by vintage (sample_weight=1):")
    for tech in ("solar", "wind"):
        s = train.groupby("forecast_vintage")[f"sample_weight_{tech}"].sum().astype(int)
        print(f"    {tech:6}", s.to_dict(), f"total={int(s.sum()):,}")


if __name__ == "__main__":
    main()
