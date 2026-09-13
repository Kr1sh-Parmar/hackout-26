"""DuckDB-over-parquet reader for the gold layer `run_cycle.py` writes.

Every read method must return an empty DataFrame -- never raise -- when the
table does not exist yet, because the API has to work before a single gold
parquet file has landed. DuckDB is asked with parameterised `?` placeholders
throughout; nothing user-supplied is ever interpolated into SQL text.
"""

from __future__ import annotations

import datetime as dt
import glob
import pathlib
import threading

import duckdb
import pandas as pd


def _posix(path: pathlib.Path) -> str:
    """DuckDB's read_parquet glob wants forward slashes even on Windows."""
    return str(path).replace("\\", "/")


class ParquetStore:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)
        self.con = duckdb.connect(":memory:")
        # One connection is shared by every request thread (see `_query`).
        self._lock = threading.Lock()
        # DuckDB renders TIMESTAMPTZ in the SESSION timezone, so on a machine set
        # to IST every timestamp came back as +05:30 -- the same instant, but the
        # API is not the presentation layer and must speak UTC. Pin it here so the
        # served value does not depend on where the server happens to be running.
        self.con.execute("SET TimeZone='UTC'")

    def _glob(self, layer: str, name: str) -> str:
        return _posix(self.root / layer / name / "**" / "*.parquet")

    def _exists(self, layer: str, name: str) -> bool:
        return len(glob.glob(self._glob(layer, name), recursive=True)) > 0

    def _query(self, sql: str, params: list) -> pd.DataFrame:
        try:
            # FastAPI runs sync routes in a threadpool, all sharing this store. Overlapping
            # execute/fetchdf on one DuckDB connection returned None, so every route 500'd
            # the moment a browser loaded its panels in parallel (live path only -- replay
            # barely queries). ponytail: serialised reads; they are millisecond parquet
            # scans. Per-thread cursors (each re-pinning TimeZone) if this ever queues.
            with self._lock:
                return self.con.execute(sql, params).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()

    def _read_run(
        self,
        name: str,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None,
        extra_sql: str = "",
        extra_params: list | None = None,
    ) -> pd.DataFrame:
        """Read one gold table, pinned to a specific run or the latest one."""
        if not self._exists("gold", name):
            return pd.DataFrame()
        pattern = self._glob("gold", name)
        sql = "SELECT * FROM read_parquet(?, hive_partitioning=1) WHERE region_id = ?"
        params: list = [pattern, region_id]
        if run_ts is not None:
            sql += " AND run_ts_utc = ?"
            params.append(pd.Timestamp(run_ts))
        else:
            sql += (
                " AND run_ts_utc = (SELECT max(run_ts_utc) FROM "
                "read_parquet(?, hive_partitioning=1) WHERE region_id = ?)"
            )
            params += [pattern, region_id]
        sql += extra_sql
        params += extra_params or []
        return self._query(sql, params)

    def read_forecast(
        self,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None = None,
        horizon: int = 72,
        tech: str | None = None,
    ) -> pd.DataFrame:
        extra_sql = " AND lead_hours <= ?"
        params = [horizon]
        if tech is not None:
            extra_sql += " AND tech = ?"
            params.append(tech)
        return self._read_run("forecast", region_id, run_ts, extra_sql, params)

    def read_outlook(
        self,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None = None,
        horizon: int = 72,
    ) -> pd.DataFrame:
        return self._read_run("outlook", region_id, run_ts, " AND lead_hours <= ?", [horizon])

    def read_events(
        self,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None = None,
        min_severity: int = 1,
    ) -> pd.DataFrame:
        return self._read_run("events", region_id, run_ts, " AND severity >= ?", [min_severity])

    def read_actions(
        self,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self._read_run("actions", region_id, run_ts)

    def read_explain(
        self,
        region_id: str,
        run_ts: dt.datetime | pd.Timestamp | None = None,
        tech: str | None = None,
        valid_ts: dt.datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        extra_sql = ""
        params: list = []
        if tech is not None:
            extra_sql += " AND tech = ?"
            params.append(tech)
        if valid_ts is not None:
            extra_sql += " AND valid_ts_utc = ?"
            params.append(pd.Timestamp(valid_ts))
        extra_sql += " ORDER BY valid_ts_utc, tech, rank"
        return self._read_run("explain", region_id, run_ts, extra_sql, params)

    def read_drift(
        self, region_id: str, run_ts: dt.datetime | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """The retrain verdict from the latest cycle. Empty until one has run."""
        return self._read_run("drift", region_id, run_ts)

    def read_backtest(self, region_id: str, tech: str) -> pd.DataFrame:
        """Per-lead-hour accuracy. One row per lead hour, aggregated over every
        walk-forward fold -- there is no `valid_ts_utc` here and so no time
        window to filter by; see `api.routes.backtest`.
        """
        if not self._exists("gold", "backtest"):
            return pd.DataFrame()
        sql = "SELECT * FROM read_parquet(?, hive_partitioning=1) WHERE region_id = ? AND tech = ?"
        return self._query(sql, [self._glob("gold", "backtest"), region_id, tech])

    def list_runs(self, region_id: str) -> list[pd.Timestamp]:
        """Every forecast run in gold for a region, newest first. Every cycle writes
        `forecast`, so its runs are the runs."""
        if not self._exists("gold", "forecast"):
            return []
        df = self._query(
            "SELECT DISTINCT run_ts_utc AS r FROM read_parquet(?, hive_partitioning=1) "
            "WHERE region_id = ? ORDER BY r DESC",
            [self._glob("gold", "forecast"), region_id],
        )
        if df.empty:
            return []
        stamps = (pd.Timestamp(t) for t in df["r"])
        return [t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC") for t in stamps]

    def latest_run(self, region_id: str) -> dt.datetime | None:
        for name in ("forecast", "outlook"):
            if not self._exists("gold", name):
                continue
            df = self._query(
                "SELECT max(run_ts_utc) AS m FROM read_parquet(?, hive_partitioning=1) "
                "WHERE region_id = ?",
                [self._glob("gold", name), region_id],
            )
            if not df.empty and pd.notna(df.loc[0, "m"]):
                ts = pd.Timestamp(df.loc[0, "m"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                return ts.to_pydatetime()
        return None
