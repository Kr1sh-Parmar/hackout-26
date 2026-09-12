from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ...evaluation.report import summarise
from ...ingest.replay import read_table
from ..deps import get_store
from ..schemas import BacktestResponse, BacktestSummary, Tech
from ._common import provenance_fields, records, resolve_region

router = APIRouter()

# The design target for the 80% band: coverage within 2 pp of nominal.
PICP_TARGET = (0.78, 0.82)


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(
    region_id: str,
    tech: Tech,
    store: ParquetStore = Depends(get_store),
) -> BacktestResponse:
    """Accuracy per lead hour, from the walk-forward backtest, plus its headline.

    No time-window parameters, deliberately, though dev-01 4 lists them: this
    table is aggregated ACROSS the whole walk-forward and indexed by lead hour,
    so it carries no `valid_ts_utc` to slice on. Accepting a window would have
    quietly returned an empty frame for every caller that passed one.
    """
    cfg = resolve_region(region_id)
    df = read_table(region_id, "backtest", store, tech=tech.value)
    if not df.empty and "lead_hours" in df.columns:
        df = df.sort_values("lead_hours", ignore_index=True)
    return BacktestResponse(
        **provenance_fields(region_id, store, cfg, df),
        data=records(df, list(df.columns)),
        summary=_summary(df),
    )


def _summary(df: pd.DataFrame) -> BacktestSummary | None:
    if df.empty or "n_rows" not in df.columns:
        return None
    headline = summarise(df)
    if not headline:
        return None

    scored = df[df["n_rows"] > 0]
    picp = scored["picp_80"] if "picp_80" in scored.columns else pd.Series(dtype=float)
    folds = df["folds"].dropna() if "folds" in df.columns else pd.Series(dtype=float)
    finite = {k: (v if math.isfinite(v) else None) for k, v in headline.items()}
    return BacktestSummary(
        **{**finite, "n_rows": int(headline["n_rows"])},
        # Absent on tables written before backtest.py recorded it; unknown, not zero.
        folds=int(folds.iloc[0]) if not folds.empty else None,
        leads_scored=int(len(scored)),
        leads_in_band=int(picp.between(*PICP_TARGET).sum()),
        picp_target_low=PICP_TARGET[0],
        picp_target_high=PICP_TARGET[1],
    )
