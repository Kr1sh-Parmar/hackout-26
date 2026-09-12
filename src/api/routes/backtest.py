from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import BacktestResponse, Tech
from ._common import provenance_fields, records, resolve_region

router = APIRouter()


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(
    region_id: str,
    tech: Tech,
    store: ParquetStore = Depends(get_store),
) -> BacktestResponse:
    """Accuracy per lead hour, from the walk-forward backtest.

    No time-window parameters, deliberately, though dev-01 4 lists them: this
    table is aggregated ACROSS the whole walk-forward and indexed by lead hour,
    so it carries no `valid_ts_utc` to slice on. Accepting a window would have
    quietly returned an empty frame for every caller that passed one.
    """
    cfg = resolve_region(region_id)
    df = read_table(region_id, "backtest", store, tech=tech.value)
    data = records(df, list(df.columns))
    return BacktestResponse(**provenance_fields(region_id, store, cfg, df), data=data)
