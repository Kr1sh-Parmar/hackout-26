"""Replay mode -- the most valuable 30 lines in the repo.

Live APIs fail during demos far more often than models do. `snapshot` freezes
one good gold cycle to disk; `load_replay` reads it back; `read_table` is the
one-line switch a caller uses to say "give me the replay copy when
REPLAY_MODE is on, otherwise ask the real store" -- so the live path and the
demo path are the same code, and which one served a given request is never a
mystery (`api/main.py`'s `/health` already reports `replay_mode`).

No cache framework: one directory, one parquet per table, most-recent-wins.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from ..core.config import Settings, get_settings

TABLES: tuple[str, ...] = ("forecast", "outlook", "events", "actions")


def _replay_root(region_id: str, settings: Settings | None = None) -> pathlib.Path:
    settings = settings or get_settings()
    return settings.artifact_root / "replay" / region_id


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
    run_ts = pd.Timestamp(run_ts)
    if run_ts.tzinfo is None:
        run_ts = run_ts.tz_localize("UTC")
    dest = _replay_root(region_id, settings) / f"{run_ts:%Y%m%dT%H}"
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
    return written


def load_replay(region_id: str, table: str, settings: Settings | None = None) -> pd.DataFrame:
    """Read the most recent snapshot for `table`. Empty frame if none exists --
    never raises, so a missing snapshot degrades rather than crashes a demo.
    """
    root = _replay_root(region_id, settings)
    if not root.exists():
        return pd.DataFrame()
    snapshots = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    for snap in snapshots:
        f = snap / f"{table}.parquet"
        if f.exists():
            return pd.read_parquet(f)
    return pd.DataFrame()


def read_table(region_id: str, table: str, store) -> pd.DataFrame:
    """Replay data when `REPLAY_MODE` is on, otherwise delegate to `store`.

    `store` is a `ParquetStore`-shaped object; the live-path method name is
    `read_{table}` (e.g. `read_forecast`), matching `core.store.ParquetStore`.
    """
    settings = get_settings()
    if settings.replay_mode:
        return load_replay(region_id, table, settings)
    return getattr(store, f"read_{table}")(region_id)
