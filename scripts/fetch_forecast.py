"""Fetch live weather for the next 72 h and cache it as a run.

    python scripts/fetch_forecast.py --region BE

Writes `data/silver/live_weather/region_id=BE/run_date=YYYY-MM-DD/part-0.parquet`
so a cycle can be re-run without re-hitting the API, and so the exact inputs
behind a published forecast stay auditable after the fact.

This is the serving twin of `scripts/dl_openmeteo_*.py`. It asks for the same
variables, in the same units, and routes through the same regional aggregation,
because a model served differently-shaped weather than it trained on fails in a
way no backtest can show.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.ingest.adapters.openmeteo import WeatherUnavailable  # noqa: E402
from src.ingest.live import live_weather, run_timestamp  # noqa: E402

SILVER = pathlib.Path("data/silver/live_weather")


def cache_path(region: str, run_ts: pd.Timestamp) -> pathlib.Path:
    return SILVER / f"region_id={region}" / f"run_date={run_ts:%Y-%m-%d}" / "part-0.parquet"


def latest_cached(region: str) -> pd.DataFrame | None:
    """Most recent cached live run, for offline re-runs and for degraded mode."""
    root = SILVER / f"region_id={region}"
    parts = sorted(root.glob("run_date=*/part-0.parquet"))
    if not parts:
        return None
    df = pd.read_parquet(parts[-1])
    return df.sort_values("valid_ts_utc").reset_index(drop=True)


def fetch(region: str, horizon_hours: int = 72) -> pd.DataFrame:
    cfg = load_region(region)
    run_ts = run_timestamp()
    wx = live_weather(cfg, run_ts=run_ts, horizon_hours=horizon_hours)

    dest = cache_path(region, run_ts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(dest, index=False)

    print(f"  run_ts   {run_ts.isoformat()}")
    print(f"  rows     {len(wx):,} ({wx.lead_hours.min()}..{wx.lead_hours.max()} h lead)")
    print(f"  horizon  {wx.valid_ts_utc.min()} -> {wx.valid_ts_utc.max()}")
    print(f"  models   {sorted({c.split('__')[1] for c in wx.columns if '__' in c})}")
    print(f"  cached   {dest.as_posix()}")
    return wx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--horizon-hours", type=int, default=72)
    args = ap.parse_args()

    print(f"== live weather {args.region} ==")
    try:
        fetch(args.region, args.horizon_hours)
    except WeatherUnavailable as exc:
        # Fail loudly and point at the fallback rather than emitting a silent
        # partial forecast: a wrong number an operator acts on is worse than none.
        print(f"  LIVE WEATHER UNAVAILABLE: {exc}", file=sys.stderr)
        cached = latest_cached(args.region)
        if cached is not None:
            print(
                f"  last cached run is {cached.run_ts_utc.iloc[0]} -- "
                "re-run the cycle against it, or use REPLAY_MODE=true",
                file=sys.stderr,
            )
        sys.exit(2)


if __name__ == "__main__":
    main()
