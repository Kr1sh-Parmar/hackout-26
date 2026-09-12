from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ..deps import get_store
from ..schemas import BacktestResponse, Tech
from ._common import as_utc, provenance_fields, records, resolve_region

router = APIRouter()


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(
    region_id: str,
    tech: Tech,
    window_start: dt.datetime | None = None,
    window_end: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> BacktestResponse:
    cfg = resolve_region(region_id)
    df = store.read_backtest(
        region_id, tech.value, window_start=as_utc(window_start), window_end=as_utc(window_end)
    )
    data = records(df, list(df.columns))
    return BacktestResponse(**provenance_fields(region_id, store, cfg, df), data=data)
