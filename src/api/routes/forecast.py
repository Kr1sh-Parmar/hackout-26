from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from ...core.store import ParquetStore
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import ForecastResponse, Tech
from ._common import as_utc, provenance_fields, records, require_fresh_run, resolve_region

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    region_id: str,
    run_ts: dt.datetime | None = None,
    horizon_hours: int = Query(72, ge=1, le=168),
    tech: Tech | None = None,
    store: ParquetStore = Depends(get_store),
) -> ForecastResponse:
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = read_table(
        region_id,
        "forecast",
        store,
        run_ts=run_ts,
        horizon=horizon_hours,
        tech=tech.value if tech else None,
    )
    data = records(
        df,
        [
            "valid_ts_utc",
            "lead_hours",
            "tech",
            "p10_mw",
            "p50_mw",
            "p90_mw",
            "capacity_mw",
            "calibrated",
        ],
    )
    return ForecastResponse(**provenance_fields(region_id, store, cfg, df), data=data)
