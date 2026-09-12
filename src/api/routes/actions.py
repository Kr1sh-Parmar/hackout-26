from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import APIRouter, Depends

from ...core.store import ParquetStore
from ...decisions.recommend import RECOMMEND_COLUMNS
from ...ingest.replay import read_table
from .. import acks
from ..deps import get_store
from ..errors import ActionNotFound
from ..schemas import AckResponse, ActionsResponse
from ._common import (
    as_utc,
    provenance_fields,
    records,
    require_fresh_run,
    require_run,
    resolve_region,
)

router = APIRouter()

_POINT_COLUMNS = [c for c in RECOMMEND_COLUMNS if c != "region_id"] + [
    "action_id",
    "acknowledged_at",
]


def _with_ack_state(region_id: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(action_id=pd.Series(dtype=str), acknowledged_at=pd.Series(dtype=str))
    ids = [
        acks.action_id(region_id, run, action, start)
        for run, action, start in zip(df["run_ts_utc"], df["action"], df["valid_from"])
    ]
    df = df.assign(action_id=ids)
    return df.assign(acknowledged_at=df["action_id"].map(acks.load(region_id)))


@router.get("/actions", response_model=ActionsResponse)
def get_actions(
    region_id: str,
    run_ts: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> ActionsResponse:
    cfg = resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_fresh_run(region_id, store, run_ts)
    df = _with_ack_state(region_id, read_table(region_id, "actions", store, run_ts=run_ts))
    data = records(df, _POINT_COLUMNS)
    return ActionsResponse(**provenance_fields(region_id, store, cfg, df, run_ts), data=data)


def _require_action(region_id: str, aid: str, run_ts: dt.datetime | None, store) -> None:
    """Only an action the API is actually serving can be acknowledged -- an unknown
    id is a stale screen or a typo, and silently storing it would report success."""
    resolve_region(region_id)
    run_ts = as_utc(run_ts)
    require_run(region_id, store, run_ts)
    served = _with_ack_state(region_id, read_table(region_id, "actions", store, run_ts=run_ts))
    if served.empty or aid not in set(served["action_id"]):
        raise ActionNotFound(region_id, aid)


@router.post("/actions/{action_id}/ack", response_model=AckResponse)
def acknowledge_action(
    action_id: str,
    region_id: str,
    run_ts: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> AckResponse:
    _require_action(region_id, action_id, run_ts, store)
    return AckResponse(
        region_id=region_id,
        action_id=action_id,
        acknowledged_at=acks.acknowledge(region_id, action_id),
    )


@router.delete("/actions/{action_id}/ack", response_model=AckResponse)
def withdraw_acknowledgement(
    action_id: str,
    region_id: str,
    run_ts: dt.datetime | None = None,
    store: ParquetStore = Depends(get_store),
) -> AckResponse:
    _require_action(region_id, action_id, run_ts, store)
    acks.withdraw(region_id, action_id)
    return AckResponse(region_id=region_id, action_id=action_id, acknowledged_at=None)
