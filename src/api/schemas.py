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
    "BacktestSummary",
    "Driver",
    "Health",
    "SiteInfo",
    "ForecastResponse",
    "OutlookResponse",
    "EventsResponse",
    "ActionsResponse",
    "AckResponse",
    "RunsResponse",
    "SweepResponse",
    "BacktestResponse",
    "ExplainResponse",
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
    age_minutes: float | None = None


class ForecastPoint(BaseModel):
    valid_ts_utc: dt.datetime
    lead_hours: int
    tech: Tech
    p10_mw: float
    p50_mw: float
    p90_mw: float
    capacity_mw: float
    # Whether THIS row's band was conformally calibrated. Not decoration: an
    # uncalibrated band is a decoration, a calibrated one is a reserve
    # requirement, and the consumer is entitled to know which it just received.
    # False below the trained lead band, and false for every row of a
    # physics-only region, which has no labels to calibrate against.
    calibrated: bool = False


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
    # Stable for this action in this run: what an acknowledgement is keyed by.
    # Derived, not stored -- see `api.acks.action_id`.
    action_id: str
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
    acknowledged_at: dt.datetime | None = None


class AckResponse(BaseModel):
    region_id: str
    action_id: str
    # None after an acknowledgement is withdrawn.
    acknowledged_at: dt.datetime | None = None


class RunsResponse(BaseModel):
    region_id: str
    replay_mode: bool
    latest: dt.datetime | None = None
    # Newest first. Any of these is a valid `run_ts` for the operational routes.
    runs: list[dt.datetime]


class SweepPoint(BaseModel):
    energy_capacity_mwh: float
    curtailment_avoided_gwh_yr: float
    value_inr_yr: float
    cycles_per_year: float


class LeadHourMetric(BaseModel):
    """One lead hour of the walk-forward backtest.

    The evaluation track owns the gold table and its metric set may grow, so
    `extra="allow"` lets a new column through rather than 500ing the endpoint
    the day it lands. The named fields are the ones a client can rely on --
    everything here is NORMALISED by installed capacity, which is why there is
    no `_mw` metric in the list: megawatt errors are not comparable between two
    regions, two technologies, or the same fleet a year apart.
    """

    model_config = ConfigDict(extra="allow")

    lead_hours: int
    tech: Tech
    n_rows: int | None = None
    nrmse_model: float | None = None
    nrmse_persistence: float | None = None
    nrmse_physics: float | None = None
    nrmse_tso: float | None = None
    skill_vs_persistence: float | None = None
    picp_80: float | None = None
    mean_width_frac: float | None = None


class BacktestSummary(BaseModel):
    """The headline over the whole walk-forward.

    Computed by `evaluation.report.summarise` -- the same function that writes
    `artifacts/backtest.json` -- so the served headline and the published one
    cannot disagree. Clients must not re-derive it from the per-lead rows: the
    obvious pooled RMS gives 6.05% solar nRMSE where the headline (a row-weighted
    mean, skill from the aggregated errors) is 5.36%.
    """

    folds: int | None = None
    n_rows: int
    leads_scored: int
    leads_in_band: int
    picp_target_low: float
    picp_target_high: float
    nrmse_mean: float | None = None
    nrmse_persistence: float | None = None
    nrmse_physics: float | None = None
    nrmse_tso: float | None = None
    nrmse_tso_wa: float | None = None
    skill_mean: float | None = None
    skill_vs_physics: float | None = None
    mbe_mean: float | None = None
    picp_mean: float | None = None


class Driver(BaseModel):
    """One feature's signed contribution to the p50 residual correction.

    Units are capacity factor, and the sign is meaningful: positive means this
    feature pushed the forecast ABOVE what physics alone predicted.
    """

    valid_ts_utc: dt.datetime
    lead_hours: int
    tech: Tech
    rank: int
    feature: str
    value: float | None = None
    contribution: float
    base_cf: float | None = None
    prediction_cf: float | None = None


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
    # No labels to train or calibrate against: a transfer region, which serves a
    # physics-only forecast and makes no accuracy claim.
    physics_only: bool = False


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
    summary: BacktestSummary | None = None


class ExplainResponse(Provenance):
    data: list[Driver]
