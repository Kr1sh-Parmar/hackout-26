"""Integration checks from data.md 15 and dev-04 6.

Each of these has caught a real bug in this class of project. Run before building
any features. Exits non-zero if any FAIL.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, pandas as pd

SILVER = pathlib.Path("data/silver")
RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def rd(name: str) -> pd.DataFrame:
    return pd.read_parquet(SILVER / name / "part-0.parquet")


def main() -> int:
    wx = rd("weather_nwp")
    sol = rd("generation_actuals_solar")
    wnd = rd("generation_actuals_wind")
    load = rd("load")
    tso_s = rd("tso_forecast_solar")

    print("== units and physical range ==")
    check("wind in m/s not km/h (max 100m < 60)",
          wx.wind_speed_100m_ms.max() < 60, f"max={wx.wind_speed_100m_ms.max():.1f} m/s")
    check("irradiance in physical range (< 1400 W/m2)",
          wx.ghi_wm2.max() < 1400, f"max={wx.ghi_wm2.max():.0f} W/m2")
    check("GHI plausible for Belgium (700-1000 W/m2 peak)",
          700 <= wx.ghi_wm2.max() <= 1100, f"max={wx.ghi_wm2.max():.0f}")
    check("temperature in range",
          wx.temperature_2m_c.dropna().between(-60, 60).all(),
          f"{wx.temperature_2m_c.min():.1f}..{wx.temperature_2m_c.max():.1f} C")
    check("surface pressure in range",
          wx.surface_pressure_hpa.dropna().between(800, 1100).all(),
          f"{wx.surface_pressure_hpa.min():.0f}..{wx.surface_pressure_hpa.max():.0f} hPa")
    check("wind direction 0-360",
          wx.wind_direction_100m_deg.dropna().between(0, 360).all())

    print("== timezone and keys ==")
    check("weather ts is tz-aware UTC", str(wx.valid_ts_utc.dt.tz) == "UTC")
    check("generation ts is tz-aware UTC", str(sol.ts_utc.dt.tz) == "UTC")
    key = ["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"]
    check("weather PK unique (run,valid,model,point)",
          not wx.duplicated(key).any(),
          f"{int(wx.duplicated(key).sum())} dups")
    check("no duplicate solar timestamps", not sol.ts_utc.duplicated().any())
    check("no duplicate wind timestamps", not wnd.ts_utc.duplicated().any())

    print("== NWP model coverage ==")
    per = wx.groupby(["grid_point_id", "nwp_model"]).size().unstack(fill_value=0)
    check("all 3 NWP models present at every grid point",
          (per > 0).all().all(), f"{per.shape[0]} points x {per.shape[1]} models")
    check("grid point weights sum to 1",
          abs(wx.drop_duplicates("grid_point_id").weight.sum() - 1.0) < 1e-6,
          f"sum={wx.drop_duplicates('grid_point_id').weight.sum():.4f}")

    print("== spatial coverage per vintage ==")
    # the regional value is a capacity-weighted mean; if a vintage is missing grid
    # points the weights renormalise silently and the region is under-sampled.
    cov = (wx.groupby(["forecast_vintage", "nwp_model"])["grid_point_id"]
             .nunique().unstack(fill_value=0))
    n_points = wx.grid_point_id.nunique()
    worst = int(cov.values.min())
    print(cov.to_string())
    check("every vintage x model covers all grid points",
          worst == n_points, f"min {worst}/{n_points} points")
    wsum = (wx.drop_duplicates(["forecast_vintage", "nwp_model", "grid_point_id"])
              .groupby(["forecast_vintage", "nwp_model"]).weight.sum())
    check("capacity weight covered >= 0.99 for every vintage x model",
          wsum.min() > 0.99, f"min weight covered={wsum.min():.2f}")

    print("== lead hours ==")
    tr = wx[wx.forecast_vintage.str.startswith("prev_day")]
    check("24-72 h training vintages exist", len(tr) > 0, f"{len(tr):,} rows")
    if len(tr):
        check("lead_hours covers the 24-72 h horizon",
              tr.lead_hours.min() <= 24 and tr.lead_hours.max() >= 71,
              f"{tr.lead_hours.min()}..{tr.lead_hours.max()} h")
        check("real (non-approximated) run_ts on training vintages",
              (~tr.run_ts_is_approx).all())

    print("== no solar at night (timezone alignment) ==")
    s = sol.set_index("ts_utc")
    night = s.between_time("23:00", "02:00").power_mw
    cap = sol.monitored_capacity_mw.max()
    check("no solar generation at night",
          night.max() < cap * 0.001,
          f"max night={night.max():.2f} MW vs cap {cap:,.0f} MW")

    print("== generation sanity ==")
    for nm, d in (("solar", sol), ("wind", wnd)):
        m = d.power_mw.notna() & d.monitored_capacity_mw.notna()
        ok = bool((d.loc[m, "power_mw"] <= d.loc[m, "monitored_capacity_mw"] * 1.05).all())
        worst = (d.power_mw / d.monitored_capacity_mw).max()
        check(f"{nm} power <= monitored capacity (+5% slack)", ok,
              f"max load factor={worst:.3f}")
        # small negatives are real (parasitic consumption when becalmed); they must
        # be flagged and tiny, not silently rewritten.
        neg = d[d.power_mw < 0]
        cap = d.monitored_capacity_mw.max()
        check(f"{nm} negatives are flagged and small",
              neg.empty or (neg.qc_flag.eq("OUT_OF_RANGE").all()
                            and neg.power_mw.min() > -0.01 * cap),
              f"{len(neg):,} rows, min={d.power_mw.min():.2f} MW ({100*d.power_mw.min()/cap:.3f}% of cap)")
        frac = d.qc_flag.value_counts(normalize=True).mul(100).round(2).to_dict()
        check(f"{nm} mostly OK (<5% flagged)",
              frac.get("OK", 0) > 95, str(frac))

    print("== overlap window ==")
    ov_lo = max(wx.valid_ts_utc.min(), sol.ts_utc.min(), wnd.ts_utc.min(), load.ts_utc.min())
    ov_hi = min(wx.valid_ts_utc.max(), sol.ts_utc.max(), wnd.ts_utc.max(), load.ts_utc.max())
    check("weather/generation/load overlap > 1 year",
          (ov_hi - ov_lo).days > 365, f"{ov_lo.date()} -> {ov_hi.date()} ({(ov_hi-ov_lo).days} days)")
    if len(tr):
        t_lo = max(tr.valid_ts_utc.min(), sol.ts_utc.min())
        t_hi = min(tr.valid_ts_utc.max(), sol.ts_utc.max())
        check("24-72h trainable overlap > 365 days",
              (t_hi - t_lo).days > 365, f"{t_lo.date()} -> {t_hi.date()} ({(t_hi-t_lo).days} days)")

    print("== gap census ==")
    for nm, d in (("solar", sol), ("wind", wnd), ("load", load)):
        idx = pd.DatetimeIndex(d.ts_utc)
        full = pd.date_range(idx.min(), idx.max(), freq="15min", tz="UTC")
        missing = full.difference(idx)
        check(f"{nm} interval completeness", len(missing) / len(full) < 0.01,
              f"{len(missing):,} of {len(full):,} missing ({100*len(missing)/len(full):.3f}%)",
              warn_only=True)

    print("== TSO benchmark ==")
    check("Elia forecast vintages present",
          set(tso_s.horizon.unique()) >= {"day_ahead_6pm", "day_ahead_11am", "week_ahead"},
          str(sorted(tso_s.horizon.unique())))
    j = sol.merge(tso_s[tso_s.horizon == "day_ahead_6pm"], on=["region_id", "ts_utc"], how="inner")
    j = j[(j.qc_flag == "OK") & (j.power_mw.notna()) & (j.p50_mw.notna())]
    day = j[j.p50_mw + j.power_mw > 0]
    nrmse = np.sqrt(((day.p50_mw - day.power_mw) ** 2).mean()) / day.monitored_capacity_mw.mean()
    check("Elia day-ahead solar nRMSE is a sane benchmark",
          0.01 < nrmse < 0.20, f"nRMSE={nrmse*100:.2f}% of capacity over {len(day):,} daylight intervals")

    print("== gold: physical coupling ==")
    gp = pathlib.Path("data/gold/training_base_24_72h/part-0.parquet")
    if gp.exists():
        g = pd.read_parquet(gp)
        d = g[(g.sample_weight_solar == 1) & (g.is_day == 1)]
        check("daylight solar training rows exist", len(d) > 1000, f"{len(d):,} rows")
        # the strongest single test that weather and generation are aligned in time
        c_s = d.ghi_wm2.corr(d.y_solar_cf)
        check("GHI predicts solar capacity factor (r > 0.75)", c_s > 0.75, f"r={c_s:.4f}")
        w = g[g.sample_weight_wind == 1]
        c_w = w.wind_speed_100m_ms.corr(w.y_wind_cf)
        check("100m wind predicts wind capacity factor (r > 0.75)", c_w > 0.75, f"r={c_w:.4f}")
        # disagreement must grow with horizon, else the vintages are not really
        # different forecast runs
        near = d[d.lead_hours <= 47].ghi_wm2_model_std.mean()
        far = d[d.lead_hours > 47].ghi_wm2_model_std.mean()
        check("NWP disagreement grows with lead time", far > near,
              f"lead<=47h: {near:.1f} W/m2 vs lead>47h: {far:.1f} W/m2")
        check("no lead hour below 24 in the training set",
              g.lead_hours.min() >= 24, f"min={g.lead_hours.min()} h")
        check("target is capacity factor in [0, 1.05]",
              g.y_solar_cf.dropna().between(0, 1.05).all()
              and g.y_wind_cf.dropna().between(-0.01, 1.05).all())
    else:
        check("gold layer built", False, "data/gold/training_base_24_72h missing")

    print()
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    print(f"{len(RESULTS)} checks: {len(RESULTS)-n_fail-n_warn} pass, {n_warn} warn, {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
