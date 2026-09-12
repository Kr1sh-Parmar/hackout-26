"""Replay mode -- the most valuable 30 lines in the repo.

Live APIs fail during demos far more often than models do. `snapshot` freezes
one good gold cycle to disk; `load_replay` reads it back; `read_table` is the
one-line switch a caller uses to say "give me the replay copy when
REPLAY_MODE is on, otherwise ask the real store" -- so the live path and the
demo path are the same code, and which one served a given request is never a
mystery (`api/main.py`'s `/health` already reports `replay_mode`).

No cache framework: one directory per run, one parquet per table, most-recent-wins
unless a run is pinned.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd

from ..core.config import Settings, get_settings

# Tables written once per forecast cycle, partitioned by run.
TABLES: tuple[str, ...] = ("forecast", "outlook", "events", "actions", "explain", "drift")
# Tables that belong to the MODEL, not to a cycle: `backtest.py` rewrites them
# per training run, and they carry no `run_ts_utc` to filter on. They still have
# to be frozen, or an offline demo shows a forecast it cannot vouch for.
STATIC_TABLES: tuple[str, ...] = ("backtest",)

_RUN_DIR = "%Y%m%dT%H"


def _replay_root(region_id: str, settings: Settings | None = None) -> pathlib.Path:
    settings = settings or get_settings()
    return settings.artifact_root / "replay" / region_id


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def snapshot(
    region_id: str,
    run_ts: pd.Timestamp,
    gold_root: pathlib.Path = pathlib.Path("data/gold"),
    settings: Settings | None = None,
) -> list[str]:
    """Freeze one gold cycle (forecast/outlook/events/actions) to
    `artifacts/replay/{region}/{YYYYMMDDTHH}/{table}.parquet`.

    Reads whichever tables actually have a run for `run_ts` and skips the
    rest -- a demo snapshot works with three tables just as well as four.
    Returns the list of table names actually written.
    """
    run_ts = _utc(run_ts)
    dest = _replay_root(region_id, settings) / f"{run_ts:{_RUN_DIR}}"
    dest.mkdir(parents=True, exist_ok=True)

    written = []
    for table in TABLES:
        pattern = gold_root / table / f"region_id={region_id}" / f"run_date={run_ts:%Y-%m-%d}"
        files = sorted(pattern.glob("*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "run_ts_utc" in df.columns:
            df = df[pd.to_datetime(df["run_ts_utc"], utc=True) == run_ts]
        if df.empty:
            continue
        df.to_parquet(dest / f"{table}.parquet", index=False)
        written.append(table)

    for table in STATIC_TABLES:
        files = sorted((gold_root / table).glob(f"region_id={region_id}*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(dest / f"{table}.parquet", index=False)
        written.append(table)
    return written


def load_replay(
    region_id: str,
    table: str,
    settings: Settings | None = None,
    run_ts: pd.Timestamp | dt.datetime | None = None,
) -> pd.DataFrame:
    """Read `table` from the snapshot of `run_ts`, or from the most recent
    snapshot holding it. Empty frame if none exists -- never raises, so a missing
    snapshot degrades rather than crashes a demo.
    """
    root = _replay_root(region_id, settings)
    if not root.exists():
        return pd.DataFrame()
    if run_ts is not None and table not in STATIC_TABLES:
        # A pinned run reads THAT snapshot or nothing. Falling through to the newest
        # one would put an old run's label on a new run's numbers. Model tables are
        # run-independent, so they still come from the latest snapshot below.
        f = root / f"{_utc(run_ts):{_RUN_DIR}}" / f"{table}.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()
    snapshots = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    for snap in snapshots:
        f = snap / f"{table}.parquet"
        if f.exists():
            return pd.read_parquet(f)
    return pd.DataFrame()


def list_replay_runs(region_id: str, settings: Settings | None = None) -> list[pd.Timestamp]:
    """Every frozen cycle, newest first. The snapshot directory name IS the run hour."""
    root = _replay_root(region_id, settings)
    if not root.exists():
        return []
    runs = []
    for p in root.iterdir():
        if not p.is_dir() or not any((p / f"{t}.parquet").exists() for t in TABLES):
            continue
        try:
            runs.append(pd.Timestamp(dt.datetime.strptime(p.name, _RUN_DIR), tz="UTC"))
        except ValueError:
            continue  # not a snapshot directory
    return sorted(runs, reverse=True)


def latest_replay_run(region_id: str, settings: Settings | None = None) -> pd.Timestamp | None:
    """The run timestamp frozen in the most recent snapshot, or None.

    Replay needs its own freshness answer: `ParquetStore.latest_run` reads
    `data/gold/`, which on a demo machine may be empty or may hold a DIFFERENT
    (newer) cycle than the one being replayed. Reporting that one would put a
    provenance timestamp on screen that belongs to data nobody is looking at.
    """
    for table in TABLES:
        df = load_replay(region_id, table, settings)
        if not df.empty and "run_ts_utc" in df.columns:
            return pd.Timestamp(pd.to_datetime(df["run_ts_utc"], utc=True).max())
    return None


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the live store's query arguments to a frozen frame.

    `run_ts` has already chosen the snapshot in `load_replay`, so it is not
    applied again here. Every other filter must still bite: a demo that cannot
    narrow to solar or raise the severity floor is a screenshot, not a working
    system.
    """
    if df.empty:
        return df
    horizon = filters.get("horizon")
    if horizon is not None and "lead_hours" in df.columns:
        df = df[df["lead_hours"] <= horizon]
    tech = filters.get("tech")
    if tech is not None and "tech" in df.columns:
        df = df[df["tech"] == tech]
    min_severity = filters.get("min_severity")
    if min_severity is not None and "severity" in df.columns:
        df = df[df["severity"] >= min_severity]
    valid_ts = filters.get("valid_ts")
    if valid_ts is not None and "valid_ts_utc" in df.columns:
        df = df[pd.to_datetime(df["valid_ts_utc"], utc=True) == pd.Timestamp(valid_ts)]
    return df.reset_index(drop=True)


def read_table(region_id: str, table: str, store, **filters) -> pd.DataFrame:
    """Replay data when `REPLAY_MODE` is on, otherwise delegate to `store`.

    `store` is a `ParquetStore`-shaped object; the live-path method name is
    `read_{table}` (e.g. `read_forecast`), matching `core.store.ParquetStore`.
    This is the ONLY switch -- routes call it instead of the store directly, so
    the demo path and the live path cannot drift apart.
    """
    if get_settings().replay_mode:
        return _apply_filters(load_replay(region_id, table, run_ts=filters.get("run_ts")), filters)
    return getattr(store, f"read_{table}")(region_id, **filters)
