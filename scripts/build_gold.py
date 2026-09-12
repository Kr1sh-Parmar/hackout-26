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
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest.regional import regionalise  # noqa: E402

SILVER = pathlib.Path("data/silver")
GOLD = pathlib.Path("data/gold")
REGION_ID = "BE"


def _write(df, name):
    dest = GOLD / name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote gold/{name:22} {len(df):>9,} rows  {len(df.columns)} cols")


def regional_weather() -> pd.DataFrame:
    """Per-point, per-model silver weather -> one regional row per key.

    The aggregation itself lives in `src/ingest/regional.py` because the LIVE
    path must do it identically. Duplicating it here would let the two drift,
    and a model served differently-aggregated weather than it trained on is
    train/serve skew that no backtest can see.
    """
    wx = pd.read_parquet(SILVER / "weather_nwp" / "part-0.parquet")
    return regionalise(wx)


def hourly_generation(tech: str) -> pd.DataFrame:
    """15-min -> hourly. An hour containing any non-OK interval is not clean."""
    g = pd.read_parquet(SILVER / f"generation_actuals_{tech}" / "part-0.parquet")
    g["hour"] = g.ts_utc.dt.floor("h")
    agg = (
        g.groupby("hour")
        .agg(
            power_mw=("power_mw", "mean"),
            monitored_capacity_mw=("monitored_capacity_mw", "mean"),
            n_intervals=("power_mw", "size"),
            n_ok=("qc_flag", lambda s: (s == "OK").sum()),
        )
        .reset_index()
        .rename(columns={"hour": "valid_ts_utc"})
    )
    agg["qc_ok"] = agg.n_ok == agg.n_intervals
    agg[f"y_{tech}_mw"] = agg.power_mw
    agg[f"cap_{tech}_mw"] = agg.monitored_capacity_mw
    agg[f"y_{tech}_cf"] = agg.power_mw / agg.monitored_capacity_mw
    agg[f"qc_ok_{tech}"] = agg.qc_ok
    return agg[["valid_ts_utc", f"y_{tech}_mw", f"cap_{tech}_mw", f"y_{tech}_cf", f"qc_ok_{tech}"]]


# Elia publishes several forecast vintages, each with its own information cutoff.
# Carrying only one makes the benchmark unfair in a direction that changes with
# lead hour: `day_ahead_6pm` is issued ~18:00 D-1, so it is effectively a 6-30 h
# forecast, while `week_ahead` is issued ~7 days out. Our 24-72 h model sits
# BETWEEN them, so the honest comparison brackets it rather than picking one.
TSO_VINTAGES = {"da": "day_ahead_6pm", "wa": "week_ahead"}


def hourly_tso(tech: str) -> pd.DataFrame:
    t = pd.read_parquet(SILVER / f"tso_forecast_{tech}" / "part-0.parquet")
    t["valid_ts_utc"] = t.ts_utc.dt.floor("h")
    out = None
    for short, horizon in TSO_VINTAGES.items():
        sel = t[t.horizon == horizon]
        if sel.empty:
            continue
        agg = (
            sel.groupby("valid_ts_utc")
            .agg(
                **{
                    f"tso_{tech}_{short}_p10_mw": ("p10_mw", "mean"),
                    f"tso_{tech}_{short}_p50_mw": ("p50_mw", "mean"),
                    f"tso_{tech}_{short}_p90_mw": ("p90_mw", "mean"),
                }
            )
            .reset_index()
        )
        out = agg if out is None else out.merge(agg, on="valid_ts_utc", how="outer")

    # keep the original unsuffixed names as aliases for the day-ahead vintage so
    # existing callers keep working
    for q in ("p10", "p50", "p90"):
        src = f"tso_{tech}_da_{q}_mw"
        if src in out.columns:
            out[f"tso_{tech}_{q}_mw"] = out[src]
    return out


def hourly_load() -> pd.DataFrame:
    load = pd.read_parquet(SILVER / "load" / "part-0.parquet")
    load["valid_ts_utc"] = load.ts_utc.dt.floor("h")
    return (
        load.groupby("valid_ts_utc")
        .agg(
            demand_mw=("demand_mw", "mean"),
            demand_da_mw=("demand_da_6pm_mw", "mean"),
            demand_da_p10_mw=("demand_da_p10_mw", "mean"),
            demand_da_p90_mw=("demand_da_p90_mw", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    print("== regional weather ==")
    base = regional_weather()
    print(f"  {len(base):,} (run,valid,vintage) rows")

    for tech in ("solar", "wind"):
        base = base.merge(hourly_generation(tech), on="valid_ts_utc", how="left")
        base = base.merge(hourly_tso(tech), on="valid_ts_utc", how="left")
    base = base.merge(hourly_load(), on="valid_ts_utc", how="left")

    base.insert(0, "region_id", REGION_ID)
    base["lead_hours"] = (
        (base.valid_ts_utc - base.run_ts_utc).dt.total_seconds().div(3600).round().astype("int32")
    )

    # censored labels carry no weight: curtailed / missing / frozen / out-of-range
    for tech in ("solar", "wind"):
        ok = base[f"qc_ok_{tech}"].fillna(False) & base[f"y_{tech}_cf"].notna()
        base[f"sample_weight_{tech}"] = ok.astype(float)

    front = ["region_id", "run_ts_utc", "valid_ts_utc", "lead_hours", "forecast_vintage"]
    base = base[front + [c for c in base.columns if c not in front]]
    base = base.sort_values(["run_ts_utc", "valid_ts_utc"]).reset_index(drop=True)
    _write(base, "training_base")

    train = base[base.forecast_vintage.str.startswith("prev_day") & base.lead_hours.between(24, 72)]
    _write(train, "training_base_24_72h")

    print("\n  trainable rows by vintage (sample_weight=1):")
    for tech in ("solar", "wind"):
        s = train.groupby("forecast_vintage")[f"sample_weight_{tech}"].sum().astype(int)
        print(f"    {tech:6}", s.to_dict(), f"total={int(s.sum()):,}")


if __name__ == "__main__":
    main()
