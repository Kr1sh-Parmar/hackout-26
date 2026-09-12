"""API response models. Pydantic v2.

Every list response extends Provenance: a consumer must always be able to
answer "which model, which calibration, was this replayed?" even when `data`
is empty because no gold table has landed yet.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict

from ..decisions.recommend import Action, Flag

__all__ = [
    "Tech",
    "Flag",
    "Action",
    "Provenance",
    "ForecastPoint",
    "OutlookPoint",
    "GridEvent",
    "Recommendation",
    "SweepPoint",
    "LeadHourMetric",
    "Health",
    "SiteInfo",
    "ForecastResponse",
    "OutlookResponse",
    "EventsResponse",
    "ActionsResponse",
    "SweepResponse",
    "BacktestResponse",
]


class Tech(str, Enum):
    solar = "solar"
    wind = "wind"


class Provenance(BaseModel):
    region_id: str
    issued_at: dt.datetime
    model_version: str = "unavailable"
    calibration_date: str | None = None
    nwp_models: list[str] = []
    replay_mode: bool = False


class ForecastPoint(BaseModel):
    valid_ts_utc: dt.datetime
    lead_hours: int
    tech: Tech
    p10_mw: float
    p50_mw: float
    p90_mw: float
    capacity_mw: float


class OutlookPoint(BaseModel):
    valid_ts_utc: dt.datetime
    lead_hours: int
    demand_mw: float
    solar_p10_mw: float
    solar_p50_mw: float
    solar_p90_mw: float
    wind_p10_mw: float
    wind_p50_mw: float
    wind_p90_mw: float
    net_load_mw: float
    net_load_p10_mw: float
    net_load_p90_mw: float
    must_run_mw: float
    headroom_mw: float
    ramp_mw_per_h: float | None = None


class GridEvent(BaseModel):
    event_id: str
    flag: Flag
    severity: int
    valid_from: dt.datetime
    valid_to: dt.datetime
    lead_hours: int
    magnitude_mw: float
    energy_mwh: float
    description: str


class Recommendation(BaseModel):
    action: Action
    flag: Flag
    valid_from: dt.datetime
    valid_to: dt.datetime
    power_mw: float
    mwh: float
    value_inr: float
    confidence: float
    decisive: bool
    rationale: str
    linked_event_id: str | None = None


class SweepPoint(BaseModel):
    energy_capacity_mwh: float
    curtailment_avoided_gwh_yr: float
    value_inr_yr: float
    cycles_per_year: float


class LeadHourMetric(BaseModel):
    """The evaluation track owns the backtest gold table; its exact metric
    set may grow. `extra="allow"` lets new columns pass through unvalidated
    rather than 500ing the whole endpoint the day it lands."""

    model_config = ConfigDict(extra="allow")

    lead_hours: int
    tech: Tech
    mae_mw: float | None = None
    rmse_mw: float | None = None
    bias_mw: float | None = None
    n_obs: int | None = None


class Health(BaseModel):
    status: str
    replay_mode: bool
    regions: list[str]
    warnings: list[str] = []
    timestamp: dt.datetime


class SiteInfo(BaseModel):
    region_id: str
    timezone: str
    capacity_mw: dict[str, float]
    nwp_models: list[str]


class ForecastResponse(Provenance):
    data: list[ForecastPoint]


class OutlookResponse(Provenance):
    data: list[OutlookPoint]


class EventsResponse(Provenance):
    data: list[GridEvent]


class ActionsResponse(Provenance):
    data: list[Recommendation]


class SweepResponse(Provenance):
    data: list[SweepPoint]


class BacktestResponse(Provenance):
    data: list[LeadHourMetric]
