"""Live storage-capacity sweep -- not a gold table, computed on request
against whatever outlook is currently available.

The sweep is computed here, but its INPUT goes through the replay switch like
every other operational route: reading the store directly would have served a
sizing curve off live gold while the rest of the screen showed replayed data."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...core.store import ParquetStore
from ...decisions.storage_sim import sweep
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import SweepResponse
from ._common import provenance_fields, records, resolve_region

router = APIRouter()


@router.get("/storage/sweep", response_model=SweepResponse)
def get_storage_sweep(
    region_id: str,
    max_mwh: float = Query(1000, gt=0),
    steps: int = Query(25, ge=2, le=200),
    store: ParquetStore = Depends(get_store),
) -> SweepResponse:
    cfg = resolve_region(region_id)
    outlook = read_table(region_id, "outlook", store)
    if outlook.empty:
        data = []
    else:
        sw = sweep(outlook, cfg.storage, max_mwh=max_mwh, steps=steps)
        data = records(sw, list(sw.columns))
    return SweepResponse(**provenance_fields(region_id, store, cfg, outlook), data=data)
