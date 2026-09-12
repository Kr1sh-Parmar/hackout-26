from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from ...core.store import ParquetStore
from ...decisions.events import EVENT_COLUMNS
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import EventsResponse
from ._common import as_utc, provenance_fields, records, require_fresh_run, resolve_region

router = APIRouter()


@router.get("/events", response_model=EventsResponse)
def get_events(
    region_id: str,
    run_ts: dt.datetime | None = None,
    min_severity: int = Query(1, ge=1, le=5),
    store: ParquetStore = Depends(get_store),
) -> EventsResponse:
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = read_table(region_id, "events", store, run_ts=run_ts, min_severity=min_severity)
    data = records(df, EVENT_COLUMNS)
    return EventsResponse(**provenance_fields(region_id, store, cfg, df), data=data)
