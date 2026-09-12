from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from ...core.store import ParquetStore
from ...ingest.replay import read_table
from ...models.explain import EXPLAIN_COLUMNS
from ..deps import get_store
from ..schemas import ExplainResponse, Tech
from ._common import as_utc, provenance_fields, records, require_fresh_run, resolve_region

router = APIRouter()


@router.get("/explain", response_model=ExplainResponse)
def get_explain(
    region_id: str,
    tech: Tech | None = None,
    valid_ts: dt.datetime | None = None,
    run_ts: dt.datetime | None = None,
    top_n: int = Query(5, ge=1, le=20),
    store: ParquetStore = Depends(get_store),
) -> ExplainResponse:
    """Why is this forecast where it is? -- ranked drivers of the correction.

    Reads what `run_cycle.py` precomputed; the API never runs SHAP itself.
    """
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = read_table(
        region_id,
        "explain",
        store,
        run_ts=run_ts,
        tech=tech.value if tech else None,
        valid_ts=as_utc(valid_ts),
    )
    if not df.empty:
        df = df[df["rank"] < top_n]
    data = records(df, EXPLAIN_COLUMNS)
    return ExplainResponse(**provenance_fields(region_id, store, cfg, df, run_ts), data=data)
