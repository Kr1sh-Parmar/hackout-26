from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ...decisions.recommend import RECOMMEND_COLUMNS
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import ActionsResponse
from ._common import as_utc, provenance_fields, records, require_fresh_run, resolve_region

router = APIRouter()

_POINT_COLUMNS = [c for c in RECOMMEND_COLUMNS if c != "region_id"]


@router.get("/actions", response_model=ActionsResponse)
def get_actions(
    region_id: str,
    run_ts: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> ActionsResponse:
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = read_table(region_id, "actions", store, run_ts=run_ts)
    data = records(df, _POINT_COLUMNS)
    return ActionsResponse(**provenance_fields(region_id, store, cfg, df), data=data)
