from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import APIRouter, Depends

from ...core.config import load_region
from ...core.store import ParquetStore
from ..deps import get_settings, get_store
from ..schemas import Health, SiteInfo
from ._common import list_region_ids

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
        )
        for cfg in (load_region(rid) for rid in list_region_ids())
    ]


@router.get("/health", response_model=Health)
def get_health(store: ParquetStore = Depends(get_store)) -> Health:
    settings = get_settings()
    regions = list_region_ids()
    now = dt.datetime.now(dt.timezone.utc)
    warnings: list[str] = []

    for region_id in regions:
        latest = store.latest_run(region_id)
        if latest is None:
            warnings.append(f"{region_id}: no gold forecast data yet")
            continue
        lag_min = (now - latest).total_seconds() / 60
        if lag_min > settings.stale_warn_after_minutes:
            warnings.append(f"{region_id}: latest run is {lag_min:.0f} min old (ingest lag)")

        fc = store.read_forecast(region_id, horizon=1)
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

    if settings.replay_mode:
        warnings.append("replay_mode is active -- serving replayed, not live, data")

    return Health(
        status="ok" if not warnings else "degraded",
        replay_mode=settings.replay_mode,
        regions=regions,
        warnings=warnings,
        timestamp=now,
    )
