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
from src.models.predict import predict  # noqa: E402

GOLD = pathlib.Path("data/gold")


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
    g = pd.read_parquet(
        "data/gold/training_base_24_72h/part-0.parquet", columns=["region_id", "run_ts_utc"]
    )
    return pd.Timestamp(g[g.region_id == region].run_ts_utc.max())


def load_demand(region: str) -> pd.DataFrame:
    d = pd.read_parquet("data/silver/load/part-0.parquet")
    return d[d["region_id"] == region]


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--run-ts", default=None, help="ISO timestamp; default = latest available run")
    args = ap.parse_args()

    setup_logging("INFO")
    region = args.region
    cfg = load_region(region)
    run_ts = pd.Timestamp(args.run_ts) if args.run_ts else latest_run(region)
    if run_ts.tz is None:
        run_ts = run_ts.tz_localize("UTC")
    print(f"== cycle {region} @ {run_ts.isoformat()} ==")

    fc = predict(region, run_ts)
    write(fc, "forecast", region, run_ts)
    # carry the producing model onto every derived table
    prov = {
        "model_version": str(fc["model_version"].iloc[0]) if len(fc) else "unknown",
        "calibration_date": str(fc["calibration_date"].iloc[0]) if len(fc) else "unknown",
    }

    outlook = build_outlook(fc, load_demand(region), cfg)
    write(outlook, "outlook", region, run_ts, prov)

    events = scan_events(outlook, wx_points=load_wx_points(region, run_ts), cfg=cfg)
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
