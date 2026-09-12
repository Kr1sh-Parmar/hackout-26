"""Elia raw CSV -> bronze -> silver.

bronze/  parsed, typed, UTC, canonical names. No business logic.
silver/  validated, deduplicated, quality-flagged, nationally aggregated.

Two aggregation rules, and they are opposites -- confirmed by the discovery call:
  ods032 (solar) HAS a 'Belgium' national row plus regions AND provinces.
                 Summing would triple-count => FILTER to region == 'Belgium'.
  ods031 (wind)  has NO national row. Rows are 5 (region, on/offshore, grid) combos.
                 => SUM all 5 to get the national total.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, pandas as pd

RAW = pathlib.Path("data/raw/elia")
BRONZE = pathlib.Path("data/bronze"); SILVER = pathlib.Path("data/silver")
YEARS = [2023, 2024, 2025, 2026]
REGION_ID = "BE"

# Elia publishes four forecast vintages; each has its own information cutoff.
# Scoring against the wrong one makes the benchmark meaningless (01 6).
HORIZONS = {
    "most_recent":     ("mostrecentforecast",   "mostrecentconfidence10",  "mostrecentconfidence90"),
    "day_ahead_11am":  ("dayahead11hforecast",  "dayahead11hconfidence10", "dayahead11hconfidence90"),
    "day_ahead_6pm":   ("dayaheadforecast",     "dayaheadconfidence10",    "dayaheadconfidence90"),
    "week_ahead":      ("weekaheadforecast",    "weekaheadconfidence10",   "weekaheadconfidence90"),
}


def _read(prefix: str) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        f = RAW / f"{prefix}_{y}.csv"
        if not f.exists():
            print(f"  WARN missing {f.name}", file=sys.stderr); continue
        d = pd.read_csv(f, encoding="utf-8-sig")          # files carry a UTF-8 BOM
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["ts_utc"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.drop(columns=["datetime"])


TS_COLS = ("ts_utc", "run_ts_utc", "valid_ts_utc")


def _write(df: pd.DataFrame, layer: pathlib.Path, name: str) -> None:
    # Series.values on a tz-aware datetime silently strips the timezone, which is
    # how naive timestamps get into a layer that requires UTC. Re-assert at the
    # boundary rather than relying on every caller remembering.
    for c in TS_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
    dest = layer / name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  wrote {layer.name}/{name:26} {len(df):>9,} rows  {df.memory_usage(deep=True).sum()/1e6:>6.1f} MB")


def qc_flag(power: pd.Series, cap: pd.Series, curtailed: pd.Series) -> pd.Series:
    """Physics-based curtailment detection needs pvlib and happens at the feature
    stage; here we flag only what is determinable from the source itself."""
    f = pd.Series("OK", index=power.index, dtype=object)
    # frozen sensor: zero variation over 1h while output is materially non-zero
    roll_std = power.rolling(4, min_periods=4).std()
    f[(roll_std == 0) & (power > cap * 0.01)] = "FROZEN"
    f[(power < 0) | (power > cap * 1.05)] = "OUT_OF_RANGE"
    f[curtailed > 0] = "CURTAILED"
    f[power.isna()] = "MISSING"
    return f


def build_solar() -> None:
    df = _read("ods032_solar_historical")
    _write(df, BRONZE, "elia_solar")                      # all 14 regions preserved

    nat = df[df.region == "Belgium"].copy().sort_values("ts_utc")
    dup = nat.duplicated("ts_utc").sum()
    if dup:
        print(f"  dropping {dup:,} duplicate solar timestamps")
        nat = nat.drop_duplicates("ts_utc", keep="last")

    out = pd.DataFrame({
        "region_id": REGION_ID, "ts_utc": nat.ts_utc.values, "tech": "solar",
        "power_mw": nat.measured.values,
        "monitored_capacity_mw": nat.monitoredcapacity.values,
        "load_factor": nat.loadfactor.values,
        "curtailed_mw": 0.0,            # Elia publishes no PV curtailment flag
        "availability_pct": np.nan,
    })
    out["qc_flag"] = qc_flag(out.power_mw, out.monitored_capacity_mw, out.curtailed_mw)
    _write(out, SILVER, "generation_actuals_solar")
    _write(_tso(nat, "solar"), SILVER, "tso_forecast_solar")


def build_wind() -> None:
    df = _read("ods031_wind_historical")
    _write(df, BRONZE, "elia_wind")

    # the column is 100% non-null but ~99.2% EMPTY STRINGS; only a real bid id
    # means downward redispatch. .notna() would flag 99.6% of the fleet as curtailed.
    # Elia's CSV export writes an empty text field as the two-character string "''"
    # (two apostrophes), not as empty or null -- so a naive .notna() or .ne("")
    # test flags 99.6% of the fleet as curtailed. Strip quotes before testing.
    _bid = (df.decrementalbidid.fillna("").astype(str)
              .str.strip().str.strip("'\"").str.strip())
    df["is_curtailed"] = _bid.ne("")
    seg_keys = ["offshoreonshore", "gridconnectiontype", "region"]
    n_seg = df.groupby("ts_utc", observed=True).size()

    # per-segment table: the config defines separate onshore/offshore archetypes
    # (different hub heights and shear), so keep the split fittable independently.
    seg = (df.groupby(["ts_utc", "offshoreonshore"], observed=True)
             .agg(power_mw=("measured", "sum"),
                  monitored_capacity_mw=("monitoredcapacity", "sum"),
                  n_rows=("measured", "size"),
                  n_curtailed=("is_curtailed", "sum"))
             .reset_index())
    seg.insert(0, "region_id", REGION_ID)
    _write(seg, SILVER, "generation_actuals_wind_segment")

    # national total = sum of all 5 combos
    g = df.groupby("ts_utc", observed=True)
    nat = pd.DataFrame({
        "power_mw": g.measured.sum(min_count=1),
        "monitored_capacity_mw": g.monitoredcapacity.sum(min_count=1),
        "n_segments": g.size(),
        "n_curtailed_segments": g.is_curtailed.sum(),
    }).reset_index()

    expected = int(n_seg.mode().iat[0])
    partial = nat.n_segments < expected
    print(f"  wind: expected {expected} segments/interval; "
          f"{partial.sum():,} intervals are partial")

    out = pd.DataFrame({
        "region_id": REGION_ID, "ts_utc": nat.ts_utc, "tech": "wind",
        "power_mw": nat.power_mw, "monitored_capacity_mw": nat.monitored_capacity_mw,
        "load_factor": nat.power_mw / nat.monitored_capacity_mw,
        "curtailed_mw": 0.0,
        "availability_pct": np.nan,
    })
    out["qc_flag"] = qc_flag(out.power_mw, out.monitored_capacity_mw, out.curtailed_mw)
    # a partially-reported interval is not a real national total
    out.loc[partial.values, "qc_flag"] = "MISSING"
    # decremental bid == downward redispatch == curtailment
    out.loc[(nat.n_curtailed_segments > 0).values & (out.qc_flag == "OK"), "qc_flag"] = "CURTAILED"
    _write(out, SILVER, "generation_actuals_wind")

    # TSO benchmark: sum the forecast columns over the same 5 combos
    fc_cols = [c for trio in HORIZONS.values() for c in trio]
    wf = df.groupby("ts_utc", observed=True)[fc_cols].sum(min_count=1).reset_index()
    _write(_tso(wf, "wind"), SILVER, "tso_forecast_wind")


def _tso(src: pd.DataFrame, tech: str) -> pd.DataFrame:
    """Long-format TSO forecast -- one row per (ts, horizon). This is the external
    benchmark; keeping the vintages separate is what lets us match the cutoff."""
    rows = []
    for horizon, (p50, p10, p90) in HORIZONS.items():
        r = pd.DataFrame({
            "region_id": REGION_ID, "ts_utc": src.ts_utc.values, "tech": tech,
            "horizon": horizon,
            "p10_mw": src[p10].values, "p50_mw": src[p50].values, "p90_mw": src[p90].values,
        })
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["p50_mw"]).sort_values(["ts_utc", "horizon"])


def build_load() -> None:
    df = _read("ods001_load_historical")
    _write(df, BRONZE, "elia_load")
    df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    out = pd.DataFrame({
        "region_id": REGION_ID, "ts_utc": df.ts_utc.values,
        "demand_mw": df.totalload.values,
        "demand_da_6pm_mw": df.dayaheadforecast.values,
        "demand_da_p10_mw": df.dayaheadconfidence10.values,
        "demand_da_p90_mw": df.dayaheadconfidence90.values,
        "demand_recent_mw": df.mostrecentforecast.values,
        "demand_week_ahead_mw": df.weekaheadforecast.values,
    })
    _write(out, SILVER, "load")


if __name__ == "__main__":
    print("== solar =="); build_solar()
    print("== wind ==");  build_wind()
    print("== load ==");  build_load()
