"""The API shares ONE ParquetStore across FastAPI's threadpool.

A DuckDB connection is not safe for concurrent use: overlapping requests made
`fetchdf()` return None, and every route 500'd with "'NoneType' object has no
attribute 'empty'". It never showed in replay mode, which barely touches DuckDB --
only on the live path, the first time a browser fired its panels in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.core.store import ParquetStore

RUNS = [pd.Timestamp("2026-09-09T00:00:00Z"), pd.Timestamp("2026-09-12T17:00:00Z")]


def _write_forecast_gold(root) -> None:
    for run in RUNS:
        dest = root / "gold" / "forecast" / "region_id=BE" / f"run_date={run:%Y-%m-%d}"
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "run_ts_utc": [run] * 3,
                "lead_hours": [1, 2, 3],
                "tech": ["solar"] * 3,
                "p50_mw": [1.0, 2.0, 3.0],
            }
        ).to_parquet(dest / "part-0.parquet", index=False)


def test_concurrent_reads_on_one_store_all_succeed(tmp_path):
    _write_forecast_gold(tmp_path)
    store = ParquetStore(tmp_path)

    def work(i: int):
        kind = i % 3
        if kind == 0:
            return store.list_runs("BE")
        if kind == 1:
            return store.latest_run("BE")
        return store.read_forecast("BE")

    # Worker exceptions re-raise here, so a None from fetchdf fails the test.
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(600)))

    assert all(r == RUNS[::-1] for r in results[0::3])
    assert all(pd.Timestamp(r) == RUNS[1] for r in results[1::3])
    assert all(isinstance(r, pd.DataFrame) and len(r) == 3 for r in results[2::3])
