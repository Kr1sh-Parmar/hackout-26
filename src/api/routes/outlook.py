from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from ...core.store import ParquetStore
from ...decisions.net_load import OUTLOOK_COLUMNS
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import OutlookResponse
from ._common import as_utc, provenance_fields, records, require_fresh_run, resolve_region

router = APIRouter()

_POINT_COLUMNS = [c for c in OUTLOOK_COLUMNS if c not in ("region_id",)]


@router.get("/outlook", response_model=OutlookResponse)
def get_outlook(
    region_id: str,
    run_ts: dt.datetime | None = None,
    horizon_hours: int = Query(72, ge=1, le=168),
    store: ParquetStore = Depends(get_store),
) -> OutlookResponse:
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = read_table(region_id, "outlook", store, run_ts=run_ts, horizon=horizon_hours)
    data = records(df, _POINT_COLUMNS)
    return OutlookResponse(**provenance_fields(region_id, store, cfg, df), data=data)
