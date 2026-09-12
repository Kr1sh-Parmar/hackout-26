"""Small helpers shared by the route modules -- not a route itself."""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import yaml

from ...core.config import RegionConfig, load_region
from ...ingest.replay import latest_replay_run
from ..deps import get_settings
from ..errors import ForecastUnavailable, RegionNotConfigured, StaleForecast

REGION_DIR = pathlib.Path("config/regions")


def list_region_ids() -> list[str]:
    return sorted(
        str(yaml.safe_load(p.read_text(encoding="utf-8")).get("region_id", "?"))
        for p in REGION_DIR.glob("*.yaml")
    )


def resolve_region(region_id: str) -> RegionConfig:
    try:
        return load_region(region_id)
    except FileNotFoundError as exc:
        raise RegionNotConfigured(region_id, list_region_ids()) from exc


def as_utc(ts: dt.datetime | None) -> dt.datetime | None:
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts


def require_fresh_run(region_id: str, store, run_ts: dt.datetime | None = None) -> None:
    """Refuse to answer an operational question we have no current answer to.

    An empty 200 says "nothing is happening on the grid"; "no model has ever run
    here" says something completely different, and an operator sizing reserves is
    entitled to tell them apart. Two deliberate exemptions:

    - `run_ts` pinned -- the caller asked for one specific historical run, which
      is a replay request, not a request for current conditions.
    - replay mode -- the data is frozen ON PURPOSE. Erroring on its age would
      break the offline demo path, which is an integrity feature, not a bug.
    """
    if get_settings().replay_mode:
        # Freshness in replay is "is there a snapshot", never "how old is it".
        if latest_replay_run(region_id) is None:
            raise ForecastUnavailable(region_id)
        return
    latest = store.latest_run(region_id)
    if latest is None:
        raise ForecastUnavailable(region_id)
    if run_ts is not None:
        return
    age_min = _age_minutes(latest)
    if age_min > get_settings().stale_after_minutes:
        raise StaleForecast(region_id, age_min)


def _age_minutes(latest: dt.datetime | None) -> float | None:
    if latest is None:
        return None
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - latest).total_seconds() / 60


def provenance_fields(
    region_id: str, store, cfg: RegionConfig, df: pd.DataFrame | None = None
) -> dict:
    # In replay, `issued_at` must be the SNAPSHOT's run, not whatever happens to
    # sit in data/gold -- otherwise the screen shows a timestamp belonging to
    # data nobody is looking at.
    latest = latest_replay_run(region_id) if get_settings().replay_mode else None
    latest = latest or store.latest_run(region_id)
    issued_at = latest or dt.datetime.now(dt.timezone.utc)
    model_version = "unavailable"
    calibration_date = None
    if df is not None and not df.empty:
        if "model_version" in df.columns:
            model_version = str(df["model_version"].iloc[0])
        if "calibration_date" in df.columns:
            calibration_date = str(df["calibration_date"].iloc[0])
    return {
        "region_id": region_id,
        "issued_at": issued_at,
        "model_version": model_version,
        "calibration_date": calibration_date,
        "nwp_models": cfg.nwp_models,
        "replay_mode": get_settings().replay_mode,
        # Served alongside the data so a consumer can render "3 h old" itself.
        # Age is a property of the answer, not only a reason to withhold it.
        "age_minutes": _age_minutes(latest),
    }


def records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """DataFrame -> JSON-safe records.

    NaN is a float that json.dumps refuses ("Out of range float values are not
    JSON compliant"), so any genuinely-absent value must become None before it
    reaches the encoder. Absent values are legitimate here -- `ramp_mw_per_h` is
    a first difference and has no value on the first row of a horizon -- so the
    fix is to encode them as null, not to invent a zero that reads as "flat".
    """
    if df.empty:
        return []
    return df[columns].astype(object).where(pd.notna(df[columns]), None).to_dict("records")
