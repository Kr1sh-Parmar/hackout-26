"""Freeze whatever gold runs exist for a region into `artifacts/replay/`.

    python scripts/snapshot_replay.py --region BE
    python scripts/snapshot_replay.py --region BE --run-ts 2026-09-09T00:00:00Z

Demo from cache, and say which you are showing: after this runs, set
REPLAY_MODE=true and every `read_table` call in the app serves the frozen
copy instead of hitting `data/gold/` -- no network, no live pipeline required.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest.replay import snapshot  # noqa: E402

GOLD = pathlib.Path("data/gold")


def latest_run(region: str) -> pd.Timestamp:
    """The newest run present in the tables this script actually freezes.

    It used to read `training_base_24_72h` instead, which is the TRAINING
    matrix: it ends wherever the last archive build did, so "latest available
    run" resolved to a historical replay run and quietly snapshotted that while
    a fresher live cycle sat in gold unfrozen. The demo then served a forecast
    from whichever model produced the older run.
    """
    runs = []
    for table in ("forecast", "outlook"):
        part = GOLD / table / f"region_id={region}"
        for f in part.glob("*/*.parquet"):
            runs.append(pd.read_parquet(f, columns=["run_ts_utc"]).run_ts_utc.max())
    if runs:
        return pd.Timestamp(max(runs))
    # nothing has run a cycle yet -- fall back to the newest weather we hold
    g = pd.read_parquet(
        GOLD / "training_base_24_72h" / "part-0.parquet", columns=["region_id", "run_ts_utc"]
    )
    return pd.Timestamp(g[g.region_id == region].run_ts_utc.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--run-ts", default=None, help="ISO timestamp; default = latest available run")
    args = ap.parse_args()

    region = args.region
    run_ts = pd.Timestamp(args.run_ts) if args.run_ts else latest_run(region)
    if run_ts.tzinfo is None:
        run_ts = run_ts.tz_localize("UTC")

    written = snapshot(region, run_ts, gold_root=GOLD)
    if not written:
        print(f"no gold tables found for {region} @ {run_ts.isoformat()} -- nothing snapshotted")
        sys.exit(1)
    print(f"snapshotted {region} @ {run_ts:%Y%m%dT%H} -> {', '.join(written)}")


if __name__ == "__main__":
    main()
