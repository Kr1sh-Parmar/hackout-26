from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ...decisions.recommend import RECOMMEND_COLUMNS
from ..deps import get_store
from ..schemas import ActionsResponse
from ._common import as_utc, provenance_fields, records, resolve_region

router = APIRouter()

_POINT_COLUMNS = [c for c in RECOMMEND_COLUMNS if c != "region_id"]


@router.get("/actions", response_model=ActionsResponse)
def get_actions(
    region_id: str,
    run_ts: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> ActionsResponse:
    cfg = resolve_region(region_id)
    df = store.read_actions(region_id, run_ts=as_utc(run_ts))
    data = records(df, _POINT_COLUMNS)
    return ActionsResponse(**provenance_fields(region_id, store, cfg, df), data=data)
