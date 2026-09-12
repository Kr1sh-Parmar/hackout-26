"""Turn an hourly outlook into grid events: contiguous runs of one condition,
not one row per hour.

No pinned contract existed for this shape anywhere in the docs, so
EVENT_COLUMNS is defined here and exported -- `recommend()`'s
`linked_event_id` and the `/events` API route both join against it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import structlog

from ..core.config import RegionConfig

logger = structlog.get_logger(__name__)

EVENT_COLUMNS: list[str] = [
    "event_id",
    "flag",
    "severity",
    "valid_from",
    "valid_to",
    "lead_hours",
    "magnitude_mw",
    "energy_mwh",
    "description",
]


def empty_events() -> pd.DataFrame:
    """The contract's shape, for building against before a real run lands."""
    df = pd.DataFrame(columns=EVENT_COLUMNS)
    for c in ("valid_from", "valid_to"):
        df[c] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def _event_id(region_id: str, flag: str, valid_from: pd.Timestamp) -> str:
    """Deterministic id so re-running a cycle on the same data doesn't churn
    ids downstream (actions, UI selections, etc.)."""
    raw = f"{region_id}|{flag}|{valid_from.isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _severity(ratio: float) -> int:
    """Map a magnitude/threshold ratio to a 1-5 severity bucket."""
    return int(np.clip(np.ceil(max(ratio, 0.0)), 1, 5))


def _runs(mask: pd.Series) -> list[pd.Index]:
    """Indices of each contiguous True run in a boolean Series."""
    if not mask.any():
        return []
    groups = (mask != mask.shift()).cumsum()
    return [idx for _, idx in mask[mask].groupby(groups[mask]).groups.items()]


def _rows_for_flag(
    outlook: pd.DataFrame,
    region_id: str,
    flag: str,
    mask: pd.Series,
    magnitude_col: str,
    scale: float,
    describe,
) -> list[dict]:
    rows = []
    for idx in _runs(mask):
        window = outlook.loc[idx]
        valid_from, valid_to = window["valid_ts_utc"].min(), window["valid_ts_utc"].max()
        magnitude = float(window[magnitude_col].abs().max())
        duration_h = max(len(idx), 1)
        energy_mwh = float(window[magnitude_col].abs().sum())
        rows.append(
            {
                "event_id": _event_id(region_id, flag, valid_from),
                "flag": flag,
                "severity": _severity(magnitude / scale) if scale else 1,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "lead_hours": int(window["lead_hours"].min()) if "lead_hours" in window else 0,
                "magnitude_mw": magnitude,
                "energy_mwh": energy_mwh,
                "description": describe(magnitude, duration_h),
            }
        )
    return rows


def scan_events(
    outlook: pd.DataFrame,
    wx_points: pd.DataFrame | None = None,
    cfg: RegionConfig | None = None,
) -> pd.DataFrame:
    """Detect grid events from a net-load outlook (and, for storms, raw
    per-grid-point weather).

    Args:
        outlook: build_outlook()'s shape.
        wx_points: raw `weather_nwp`-shaped rows (per grid point, not the
            capacity-weighted regional mean) for the same run. Capacity
            weighting caps hub wind well below cut-out, so STORM_SHUTDOWN can
            only be seen by taking the max across grid points. When None,
            storm detection is skipped -- this returns no STORM_SHUTDOWN
            events, but that means "not checked", not "no storms occurred".
        cfg: region thresholds and archetypes. Required.

    Returns:
        EVENT_COLUMNS, one row per contiguous run of a condition.
    """
    if cfg is None:
        raise ValueError("scan_events requires cfg (RegionConfig)")
    if outlook.empty:
        return empty_events()

    ordered = outlook.sort_values("valid_ts_utc").reset_index(drop=True)
    region_id = ordered["region_id"].iloc[0] if "region_id" in ordered.columns else cfg.region_id
    th = cfg.decision_thresholds
    total_capacity = cfg.capacity_mw.get("solar", 0) + cfg.capacity_mw.get("wind", 0)

    rows: list[dict] = []

    # OVER_GENERATION: headroom < 0 -- net load has dropped below the
    # must-run floor. Measured at ~4% of hours for BE at must_run_mw=2800.
    rows += _rows_for_flag(
        ordered,
        region_id,
        "OVER_GENERATION",
        ordered["headroom_mw"] < 0,
        "headroom_mw",
        th.must_run_mw or 1.0,
        lambda mag, dur: f"Over-generation: net load below must-run floor by up to {mag:.0f} MW "
        f"for {dur}h",
    )

    # STEEP_RAMP: the threshold is configuration, not a literal in this
    # module -- ramp_limit_mw_per_h: 400 in belgium.yaml fires on ~45% of
    # hours (measured p50 355, p95 ~1200, max ~2900 MW/h); whoever tunes the
    # YAML should read the threshold off cfg, not off a number typed here.
    rows += _rows_for_flag(
        ordered,
        region_id,
        "STEEP_RAMP",
        ordered["ramp_mw_per_h"].abs() > th.ramp_limit_mw_per_h,
        "ramp_mw_per_h",
        th.ramp_limit_mw_per_h or 1.0,
        lambda mag, dur: f"Steep ramp: net load moving {mag:.0f} MW/h for {dur}h",
    )

    # DEFICIT_RISK: the pessimistic (p90, renewable-P10) bound crosses the
    # must-run floor while the central (p50) estimate does not -- a risk
    # that is invisible in the point forecast and only visible in the band.
    # This is exactly the case fact A's inversion exists to catch: get the
    # band backwards and this flag silently stops firing.
    deficit_mask = (ordered["net_load_p90_mw"] > th.must_run_mw) & (
        ordered["net_load_mw"] <= th.must_run_mw
    )
    risk = (ordered["net_load_p90_mw"] - th.must_run_mw).clip(lower=0)
    rows += _rows_for_flag(
        ordered.assign(_deficit_risk_mw=risk),
        region_id,
        "DEFICIT_RISK",
        deficit_mask,
        "_deficit_risk_mw",
        th.must_run_mw or 1.0,
        lambda mag, dur: f"Deficit risk: worst-case net load up to {mag:.0f} MW above the "
        f"must-run floor for {dur}h (central forecast does not cross it)",
    )

    # LOW_CONFIDENCE: band width relative to installed capacity.
    band = ordered["net_load_p90_mw"] - ordered["net_load_p10_mw"]
    rows += _rows_for_flag(
        ordered.assign(_band_mw=band),
        region_id,
        "LOW_CONFIDENCE",
        band > th.band_limit_frac * total_capacity,
        "_band_mw",
        (th.band_limit_frac * total_capacity) or 1.0,
        lambda mag, dur: f"Low confidence: forecast band {mag:.0f} MW wide for {dur}h",
    )

    # STORM_SHUTDOWN: per-grid-point max wind speed vs. the lowest wind
    # archetype cut-out. The regional capacity-weighted mean caps hub wind
    # around 18-19 m/s against a 25-27 m/s cut-out and will never trigger
    # this -- must use wx_points, not outlook.
    if wx_points is not None and not wx_points.empty:
        wind_archetypes = cfg.archetypes.get("wind", [])
        cutouts = [a.cut_out_ms for a in wind_archetypes if a.cut_out_ms]
        cutout = min(cutouts) if cutouts else 25.0
        peak = wx_points.groupby("valid_ts_utc")["wind_speed_100m_ms"].max()
        peak = peak.reindex(ordered["valid_ts_utc"])
        storm_mask = (peak > cutout).fillna(False).reset_index(drop=True)

        def wind_share_at_risk(max_wind: float) -> float:
            return sum(
                a.share for a in wind_archetypes if a.cut_out_ms and a.cut_out_ms <= max_wind
            )

        storm_frame = ordered.assign(_peak_wind=peak.to_numpy())
        for idx in _runs(storm_mask):
            window = storm_frame.loc[idx]
            valid_from, valid_to = window["valid_ts_utc"].min(), window["valid_ts_utc"].max()
            peak_wind = float(window["_peak_wind"].max())
            share = wind_share_at_risk(peak_wind)
            magnitude_mw = cfg.capacity_mw.get("wind", 0.0) * share
            duration_h = max(len(idx), 1)
            rows.append(
                {
                    "event_id": _event_id(region_id, "STORM_SHUTDOWN", valid_from),
                    "flag": "STORM_SHUTDOWN",
                    "severity": 5 if share > 0.5 else 4,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "lead_hours": int(window["lead_hours"].min()) if "lead_hours" in window else 0,
                    "magnitude_mw": magnitude_mw,
                    "energy_mwh": magnitude_mw * duration_h,
                    "description": f"Storm shutdown: peak wind {peak_wind:.1f} m/s exceeds "
                    f"cut-out, {magnitude_mw:.0f} MW of wind at risk for {duration_h}h",
                }
            )
    else:
        logger.info("storm_detection_skipped", reason="wx_points not provided")

    if not rows:
        return empty_events()
    return pd.DataFrame(rows)[EVENT_COLUMNS]
