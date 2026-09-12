"""One forecast cycle: features -> predict -> outlook -> events -> actions -> gold.

    python scripts/run_cycle.py --region BE
    python scripts/run_cycle.py --region BE --run-ts 2026-08-01T00:00:00Z

This is the batch equivalent of the scheduler job in dev-01 §7. Under lean infra
there is no APScheduler: the same call sequence runs here, invoked manually, and
the API stays a thin read layer over what this writes. A request that has to
compute a forecast is a design failure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import GridState, load_region  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.decisions.events import scan_events  # noqa: E402
from src.decisions.net_load import build_outlook  # noqa: E402
from src.decisions.recommend import recommend  # noqa: E402
from src.decisions.storage_sim import simulate  # noqa: E402
from src.evaluation.drift import (  # noqa: E402
    DRIFT_COLUMNS,
    comparable_features,
    error_drift,
    feature_drift,
    is_expected_drift,
    seasonal_window,
    should_retrain,
)
from src.ingest.adapters.elia import (  # noqa: E402
    LoadUnavailable,
    climatology_load,
    fetch_load_forecast,
)
from src.ingest.adapters.openmeteo import WeatherUnavailable  # noqa: E402
from src.ingest.live import live_weather_with_points, run_timestamp  # noqa: E402
from src.models.explain import explain_run  # noqa: E402
from src.models.registry import META_NAME, model_dir  # noqa: E402
from src.models.predict import predict  # noqa: E402

GOLD = pathlib.Path("data/gold")
TRAINING_BASE = GOLD / "training_base_24_72h" / "part-0.parquet"

# How far back "recent" reaches for the drift monitor. A month is long enough to
# carry a few hundred scored hours and short enough that a regime change is not
# averaged away by the weeks before it.
DRIFT_WINDOW_DAYS = 30


def write(
    df: pd.DataFrame,
    table: str,
    region: str,
    run_ts: pd.Timestamp,
    provenance: dict | None = None,
) -> None:
    """Partition by region and run_date so DuckDB prunes files instead of scanning.

    Every table carries the model version and calibration date, not just the
    forecast. An outlook or an action is a CONSEQUENCE of a specific model run,
    and after a bad recommendation the first question is which model produced it.
    """
    dest = GOLD / table / f"region_id={region}" / f"run_date={run_ts:%Y-%m-%d}"
    dest.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["region_id"] = region
    out["run_ts_utc"] = run_ts
    for k, v in (provenance or {}).items():
        out[k] = v
    out.to_parquet(dest / "part-0.parquet", index=False)
    print(f"  {table:9} {len(out):>5,} rows -> {dest.as_posix()}")


def latest_run(region: str) -> pd.Timestamp:
    path = pathlib.Path("data/gold/training_base_24_72h/part-0.parquet")
    if not path.exists():
        raise SystemExit(f"no training matrix at {path}; run scripts/build_gold.py first")
    g = pd.read_parquet(path, columns=["region_id", "run_ts_utc"])
    runs = g.loc[g.region_id == region, "run_ts_utc"]
    if runs.empty:
        known = sorted(g.region_id.unique())
        raise SystemExit(f"no runs for region {region!r} in the training matrix; have: {known}")
    return pd.Timestamp(runs.max())


# Demand is a per-region feed, not a platform-wide one. Elia publishes Belgium's
# and nobody else's; wiring it in unconditionally would have attached Belgian
# load to an Indian forecast and produced a net-load curve that is arithmetic
# performed on two different countries.
DEMAND_ADAPTERS = {"BE": fetch_load_forecast}


def load_demand(region: str, horizon: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Demand for the forecast horizon, or an empty frame if the region has none.

    Historical replay reads the silver table. A LIVE run needs demand for hours
    that have not happened yet, so it fetches the region's forward load forecast
    and falls back to an hour-of-week climatology if that is unavailable. Every
    row carries `demand_vintage`, because an operator must be able to tell the
    TSO's own number from our substitute for it.

    A region with no demand feed gets an empty frame, and `build_outlook`
    already returns the contract shape for that -- the generation forecast still
    serves, the net-load layer above it simply says nothing rather than saying
    something invented.
    """
    path = pathlib.Path("data/silver/load/part-0.parquet")
    hist = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["region_id"])
    hist = hist[hist["region_id"] == region]
    if horizon is None:
        return hist

    fetch = DEMAND_ADAPTERS.get(region)
    if fetch is None:
        print(f"  demand   no feed configured for {region} -- no net-load layer", file=sys.stderr)
        return pd.DataFrame()

    try:
        fwd = fetch(horizon.min(), horizon.max())
        got = set(fwd["valid_ts_utc"])
        if len(got & set(horizon)) >= len(horizon) * 0.9:
            print(f"  demand   Elia forward forecast ({fwd.demand_vintage.mode().iat[0]})")
            return fwd
        print(
            f"  demand   Elia covered only {len(got & set(horizon))}/{len(horizon)} h",
            file=sys.stderr,
        )
    except LoadUnavailable as exc:
        print(f"  demand   Elia unavailable: {exc}", file=sys.stderr)

    if hist.empty:
        print("  demand   no history to build a climatology from", file=sys.stderr)
        return pd.DataFrame()
    print("  demand   FALLBACK hour-of-week climatology", file=sys.stderr)
    return climatology_load(hist, horizon)


def load_wx_points(region: str, run_ts: pd.Timestamp) -> pd.DataFrame | None:
    """Per-grid-point weather for the storm detector.

    Storm shutdown is a LOCAL extreme: the capacity-weighted regional mean tops
    out at 18.5 m/s against a 25-27 m/s cut-out, so averaging erases exactly the
    event this is meant to catch. The detector needs the max across points.
    """
    path = pathlib.Path("data/silver/weather_nwp/part-0.parquet")
    if not path.exists():
        return None
    w = pd.read_parquet(path)
    sel = w[(w.region_id == region) & (w.run_ts_utc == run_ts)]
    return sel if len(sel) else None


def baseline_rmse_cf(region: str, tech: str) -> float:
    """The backtest nRMSE this model was published at, in capacity-factor units.

    Comes from `artifacts/backtest.json` rather than a hardcoded constant, so a
    retrained model raises its own bar instead of being judged against whatever
    number happened to be in the README.
    """
    path = pathlib.Path("artifacts/backtest.json")
    if not path.exists():
        return float("nan")
    for row in json.loads(path.read_text()):
        if row.get("region") == region and row.get("tech") == tech:
            return float(row.get("nrmse_mean", float("nan")))
    return float("nan")


def drift_check(region: str, run_ts: pd.Timestamp, cfg, window_days: int = DRIFT_WINDOW_DAYS):
    """Is the model still operating on the world it was fitted to?

    Two independent questions, per `evaluation.drift`: have the INPUTS moved
    (PSI against the training window), and has the ERROR got worse (recent
    issued forecasts against the backtest baseline). Computed here in the batch
    cycle -- /health surfaces the verdict but must never compute it, for the
    same reason it never runs the model.

    Both windows come from the training matrix, which already carries the
    regionalised inputs, `is_day`, the realised capacity factor and the sample
    weights. Reading silver instead would mean re-deriving all four, and any
    difference in how would make the monitor disagree with the model it watches.
    """
    if not TRAINING_BASE.exists():
        return pd.DataFrame(columns=DRIFT_COLUMNS)

    base = pd.read_parquet(TRAINING_BASE)
    base = base[base.region_id == region]
    if base.empty:
        return pd.DataFrame(columns=DRIFT_COLUMNS)
    cutoff = run_ts - pd.Timedelta(days=window_days)
    current = base[base.run_ts_utc > cutoff]
    issued = read_recent_forecasts(region, cutoff)

    rows = []
    for tech in sorted(cfg.capacity_mw):
        meta_file = model_dir(region, tech) / META_NAME
        if not meta_file.exists():
            continue
        train_end = pd.Timestamp(json.loads(meta_file.read_text())["train_end"])
        if train_end.tz is None:
            train_end = train_end.tz_localize("UTC")
        # Same calendar period as the current window, not the whole history --
        # otherwise every autumn reads as drift. See `drift.seasonal_window`.
        reference = seasonal_window(base[base.valid_ts_utc <= train_end], current["valid_ts_utc"])

        feats = comparable_features(reference, current) if len(current) else []
        table = feature_drift(reference, current, feats)
        # Count and rank only the UNEXPECTED drift, so the stored numbers say the
        # same thing the verdict does. Installed capacity grows every quarter and
        # carries the largest PSI in the table; reporting it as the headline
        # number would put a 6.6 next to a verdict that deliberately ignores it.
        unexpected = table[table.drifted & ~table.feature.map(is_expected_drift)]

        errors = scored_residuals(issued, base, tech)
        err = error_drift(errors, baseline_rmse_cf(region, tech))
        retrain, reason = should_retrain(table, err)
        rows.append(
            {
                "tech": tech,
                "n_features_drifted": len(unexpected),
                "drifted_features": ", ".join(unexpected.feature),
                "max_psi": float(unexpected.psi.max()) if len(unexpected) else float("nan"),
                "n_scored_hours": int(len(errors)),
                "recent_rmse_cf": err["recent_rmse"],
                "baseline_rmse_cf": err["baseline_rmse"],
                "relative_increase": err["relative_increase"],
                "retrain": bool(retrain),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS)


def read_recent_forecasts(region: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Forecasts this platform actually issued since `cutoff`.

    Operational error, not a re-scored backtest: these are the numbers an
    operator was shown. An empty frame is the normal state on a fresh install
    and means error drift simply has nothing to say yet.
    """
    files = sorted((GOLD / "forecast" / f"region_id={region}").glob("*/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[pd.to_datetime(df.run_ts_utc, utc=True) > cutoff]
    return df


def scored_residuals(issued: pd.DataFrame, base: pd.DataFrame, tech: str) -> pd.Series:
    """Residuals in capacity-factor units, on the rows the model is SCORED on.

    Solar is daylight-only here for the same reason it is daylight-only in the
    backtest: half a solar series is night zeros every model predicts perfectly,
    so including them reads as a 20% improvement over the published baseline and
    would suppress a retrain signal rather than raise one. Censored intervals
    (`sample_weight == 0`) are excluded -- a curtailed hour is not a miss.
    """
    if issued.empty or "tech" not in issued.columns:
        return pd.Series(dtype=float)
    fc = issued[issued.tech == tech]
    if fc.empty:
        return pd.Series(dtype=float)

    cols = ["run_ts_utc", "valid_ts_utc", "is_day", f"y_{tech}_cf", f"sample_weight_{tech}"]
    truth = base[[c for c in cols if c in base.columns]]
    m = fc.merge(truth, on=["run_ts_utc", "valid_ts_utc"], how="inner")
    if m.empty:
        return pd.Series(dtype=float)
    if f"sample_weight_{tech}" in m.columns:
        m = m[m[f"sample_weight_{tech}"] > 0]
    if tech == "solar" and "is_day" in m.columns:
        m = m[m.is_day == 1]
    m = m[m.capacity_mw > 0]
    return (m.p50_mw / m.capacity_mw - m[f"y_{tech}_cf"]).dropna()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--run-ts", default=None, help="ISO timestamp; default = latest available run")
    ap.add_argument(
        "--live",
        action="store_true",
        help="fetch weather from now instead of replaying a historical run",
    )
    args = ap.parse_args()

    setup_logging("INFO")
    region = args.region
    cfg = load_region(region)
    wx = wx_points = None
    if args.live:
        run_ts = run_timestamp()
        print(f"== LIVE cycle {region} @ {run_ts.isoformat()} ==")
        try:
            wx, wx_points = live_weather_with_points(cfg, run_ts=run_ts)
        except WeatherUnavailable as exc:
            # Degrade to the last good run and SAY SO. Serving a stale forecast
            # as if it were current is worse than serving nothing.
            print(f"  live weather unavailable: {exc}", file=sys.stderr)
            print("  falling back to the latest historical run", file=sys.stderr)
            run_ts, wx, wx_points = latest_run(region), None, None
        print(f"  weather  {len(wx):,} rows" if wx is not None else "  weather  from gold")
    else:
        run_ts = pd.Timestamp(args.run_ts) if args.run_ts else latest_run(region)
        if run_ts.tz is None:
            run_ts = run_ts.tz_localize("UTC")
        print(f"== cycle {region} @ {run_ts.isoformat()} ==")

    fc = predict(region, run_ts, weather=wx)
    write(fc, "forecast", region, run_ts)
    # carry the producing model onto every derived table
    prov = {
        "model_version": str(fc["model_version"].iloc[0]) if len(fc) else "unknown",
        "calibration_date": str(fc["calibration_date"].iloc[0]) if len(fc) else "unknown",
    }

    # Attribution is computed HERE, not in the API. A request that has to run
    # SHAP is the same design failure as one that has to run the model, and
    # precomputing is also what lets /explain work under REPLAY_MODE offline.
    write(explain_run(region, run_ts, weather=wx), "explain", region, run_ts, prov)

    horizon = pd.DatetimeIndex(sorted(fc["valid_ts_utc"].unique())) if len(fc) else None
    demand = load_demand(region, horizon if args.live else None)
    outlook = build_outlook(fc, demand, cfg)
    write(outlook, "outlook", region, run_ts, prov)

    points = wx_points if wx_points is not None else load_wx_points(region, run_ts)
    events = scan_events(outlook, wx_points=points, cfg=cfg)
    write(events, "events", region, run_ts, prov)

    soc = cfg.storage.energy_capacity_mwh * 0.5
    grid_state = GridState(
        region_id=region,
        must_run_mw=cfg.decision_thresholds.must_run_mw,
        storage_soc_mwh=soc,
        storage=cfg.storage,
        ramp_limit_mw_per_h=cfg.decision_thresholds.ramp_limit_mw_per_h,
    )
    # pass the events we already scanned: rescanning inside recommend() would
    # drop STORM_SHUTDOWN, which needs the per-grid-point weather held here
    actions = recommend(outlook, grid_state, cfg, events=events)
    write(actions, "actions", region, run_ts, prov)

    drift = drift_check(region, run_ts, cfg)
    if len(drift):
        write(drift, "drift", region, run_ts, prov)
        for _, d in drift.iterrows():
            print(f"  drift/{d.tech:5} {'RETRAIN' if d.retrain else 'ok':>7}: {d.reason}")

    if outlook.empty:
        # A generation forecast without a demand feed is a complete answer to a
        # smaller question. Inventing a demand curve to fill the net-load layer
        # would make every downstream action a statement about nothing.
        print(
            f"\n  {region}: generation forecast only -- no demand feed "
            "configured, so no net-load outlook, events or actions. "
            "That is the honest output, not a failure."
        )
        return

    sim = simulate(outlook, cfg.storage)
    print(f"\n  horizon {outlook.valid_ts_utc.min()} -> {outlook.valid_ts_utc.max()}")
    print(f"  over-generation hours : {(outlook.headroom_mw < 0).sum()}")
    print(f"  storage charged       : {sim.charge_mw.sum():,.0f} MWh")
    if len(events):
        print(f"  events                : {events.flag.value_counts().to_dict()}")
    if len(actions):
        print(f"  actions               : {actions.action.value_counts().to_dict()}")
        top = actions.iloc[0]
        print(
            f"  top action            : {top.action} {top.mwh:,.0f} MWh "
            f"Rs{top.value_inr:,.0f} {'DECISIVE' if top.decisive else 'indicative'}"
        )
    else:
        print("  actions               : none (a quiet horizon is a valid outcome)")


if __name__ == "__main__":
    main()
