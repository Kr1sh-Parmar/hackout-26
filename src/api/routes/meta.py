from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import APIRouter, Depends

from ...core.config import load_region
from ...core.store import ParquetStore
from ...ingest.replay import latest_replay_run, read_table
from ..deps import get_settings, get_store
from ..schemas import Health, RunsResponse, SiteInfo
from ._common import available_runs, list_region_ids, resolve_region

router = APIRouter()

_CALIBRATION_STALE_AFTER_DAYS = 30


@router.get("/sites", response_model=list[SiteInfo])
def get_sites() -> list[SiteInfo]:
    return [
        SiteInfo(
            region_id=cfg.region_id,
            timezone=cfg.timezone,
            capacity_mw=cfg.capacity_mw,
            nwp_models=cfg.nwp_models,
            physics_only=cfg.physics_only,
        )
        for cfg in (load_region(rid) for rid in list_region_ids())
    ]


@router.get("/runs", response_model=RunsResponse)
def get_runs(region_id: str, store: ParquetStore = Depends(get_store)) -> RunsResponse:
    """The forecast runs that can be served, newest first -- what a run picker
    offers. Any of them is a valid `run_ts` for the operational endpoints."""
    resolve_region(region_id)
    runs = available_runs(region_id, store)
    return RunsResponse(
        region_id=region_id,
        replay_mode=get_settings().replay_mode,
        latest=runs[0] if runs else None,
        runs=runs,
    )


@router.get("/health", response_model=Health)
def get_health(store: ParquetStore = Depends(get_store)) -> Health:
    settings = get_settings()
    regions = list_region_ids()
    now = dt.datetime.now(dt.timezone.utc)
    warnings: list[str] = []

    for region_id in regions:
        # Freshness has to come from whichever source is actually being served.
        # Reading gold here while the endpoints serve a snapshot reported "no
        # data yet" on precisely the machine replay mode exists for.
        replayed = latest_replay_run(region_id) if settings.replay_mode else None
        latest = replayed or store.latest_run(region_id)
        if latest is None:
            warnings.append(f"{region_id}: no gold forecast data yet")
            continue
        lag_min = (now - latest).total_seconds() / 60
        # Frozen data is old ON PURPOSE; saying so every time would train the
        # operator to ignore the warnings list. `replay_mode` already says it.
        if lag_min > settings.stale_warn_after_minutes and replayed is None:
            warnings.append(f"{region_id}: latest run is {lag_min:.0f} min old (ingest lag)")

        fc = read_table(region_id, "forecast", store, horizon=1)
        if not fc.empty and "calibration_date" in fc.columns:
            try:
                cal = pd.Timestamp(fc["calibration_date"].iloc[0])
                if cal.tzinfo is None:
                    cal = cal.tz_localize("UTC")
                age_days = (now - cal).days
                if age_days > _CALIBRATION_STALE_AFTER_DAYS:
                    warnings.append(f"{region_id}: calibration is {age_days}d old")
            except (ValueError, TypeError):
                pass

        for _, d in read_table(region_id, "drift", store).iterrows():
            # The monitor already decided AND said why; /health repeats the
            # reason rather than re-deriving a verdict from the numbers, so an
            # operator and a retraining job can never disagree about the call.
            if d.get("retrain"):
                warnings.append(f"{region_id}/{d['tech']}: {d['reason']}")

    if settings.replay_mode:
        warnings.append("replay_mode is active -- serving replayed, not live, data")

    return Health(
        status="ok" if not warnings else "degraded",
        replay_mode=settings.replay_mode,
        regions=regions,
        warnings=warnings,
        timestamp=now,
    )
