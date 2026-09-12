"""Small helpers shared by the route modules -- not a route itself."""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import yaml

from ...core.config import RegionConfig, load_region
from ..deps import get_settings
from ..errors import RegionNotConfigured

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


def provenance_fields(
    region_id: str, store, cfg: RegionConfig, df: pd.DataFrame | None = None
) -> dict:
    latest = store.latest_run(region_id)
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
