"""CONTRACT M4 + M5 -- forecast generation.

Returns calibrated quantiles, never a point forecast alone. Point forecasts are
operationally useless for reserve sizing: an operator needs to know how wrong
this might be, at each hour.

`calibrated` is part of the contract, not decoration. An uncalibrated band is a
decoration; a conformally calibrated one is a reserve requirement, and the
consumer is entitled to know which it just received.

Owner: ML / Physics.  Consumers: decisions/, api/, evaluation/.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from ..core.config import get_settings, load_region
from ..features.build import build_all_features
from ..models.physics import physics_forecast
from ..models.registry import load_model
from ..models.residual_gbdt import predict_residual

# Below this lead the residual model has never seen a training row: the gold
# table is `training_base_24_72h` and excludes the day0 vintage entirely. Serve
# physics only there rather than extrapolating a model outside its support.
MIN_TRAINED_LEAD_H = 24

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

PREDICT_COLUMNS: list[str] = [
    "region_id",
    "run_ts_utc",
    "valid_ts_utc",
    "lead_hours",
    "tech",
    "p10_mw",
    "p50_mw",
    "p90_mw",
    "capacity_mw",
    "model_version",
    "calibration_date",
    "calibrated",
]

_DTYPES: dict[str, str] = {
    "region_id": "object",
    "lead_hours": "int32",
    "tech": "object",
    "p10_mw": "float64",
    "p50_mw": "float64",
    "p90_mw": "float64",
    "capacity_mw": "float64",
    "model_version": "object",
    "calibration_date": "object",
    "calibrated": "bool",
}


def predict(
    region_id: str,
    run_ts: pd.Timestamp,
    horizons: range = range(1, 73),
    quantiles: tuple[float, ...] = QUANTILES,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce calibrated quantile forecasts for one region and one NWP run.

    Args:
        region_id: configured region, e.g. "BE".
        run_ts: model initialisation time, tz-aware UTC.
        horizons: lead hours to emit. The platform's operative band is 24-72 h;
            shorter leads are served but carry no autoregressive skill.
        quantiles: must be sorted ascending.

    Returns:
        One row per (valid_ts, tech) with PREDICT_COLUMNS. Quantiles are sorted
        per row -- independently fitted quantile heads can cross (p10 > p50),
        which is a real and embarrassing failure mode.

    Raises:
        FileNotFoundError: region not configured.
        RuntimeError: no model artifact registered for this region and tech.
    """
    cfg = load_region(region_id)  # raises FileNotFoundError for an unknown region
    run_ts = pd.Timestamp(run_ts)
    if run_ts.tz is None:
        raise ValueError("run_ts must be tz-aware UTC")
    # `weather` is the LIVE path: a frame from src.ingest.live shaped exactly
    # like the gold table. Absent it, replay a historical run from gold.
    wx = load_run_weather(region_id, run_ts, horizons) if weather is None else weather

    frames = []
    for tech in sorted(cfg.capacity_mw):
        try:
            artifact = load_model(region_id, tech)
        except FileNotFoundError as exc:
            raise RuntimeError(f"no model registered for {region_id}/{tech}") from exc
        frames.append(_forecast_tech(cfg, tech, wx, artifact, quantiles))

    if not frames:
        return empty_forecast()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["tech", "valid_ts_utc"]).reset_index(drop=True)[PREDICT_COLUMNS]


def load_actuals(region_id: str, tech: str) -> pd.DataFrame | None:
    """Generation history for the lag family.

    Serving DOES have history -- everything observed up to run_ts. Passing None
    here would leave every lag NaN at inference while training saw real values,
    which is train/serve skew in the one family the parity test cannot check
    (it compares non-lag columns by construction). `lags.lag_origin` keeps this
    honest: it only ever reads `valid_ts - 24*ceil(lead/24)`, which is at or
    before the run time.
    """
    path = pathlib.Path(get_settings().data_root) / f"silver/generation_actuals_{tech}"
    part = path / "part-0.parquet"
    if not part.exists():
        return None
    df = pd.read_parquet(part)
    return df[df["qc_flag"] == "OK"] if "qc_flag" in df.columns else df


def _forecast_tech(cfg, tech, wx, artifact, quantiles) -> pd.DataFrame:
    x = build_all_features(cfg.region_id, wx, actuals=load_actuals(cfg.region_id, tech), tech=tech)
    physics_cf = physics_forecast(cfg, wx, tech).to_numpy(dtype=float)
    capacity = float(cfg.capacity_mw[tech])
    lead = x["lead_hours"].to_numpy(dtype=float)
    meta = artifact.get("meta", {})

    bands = predict_residual(artifact["models"], x, physics_cf, capacity)
    lo = bands["p10_mw"].to_numpy(dtype=float)
    mid = bands["p50_mw"].to_numpy(dtype=float)
    hi = bands["p90_mw"].to_numpy(dtype=float)

    conformal = artifact.get("conformal")
    trained = lead >= MIN_TRAINED_LEAD_H
    if conformal is not None:
        clo, chi = conformal.apply(lo, hi, np.clip(lead, 0, None))
        lo, hi = np.where(trained, clo, lo), np.where(trained, chi, hi)

    # Physics only below the trained band -- no residual, no calibration claim.
    phys_mw = physics_cf * capacity
    mid = np.where(trained, mid, phys_mw)
    lo = np.where(trained, lo, phys_mw * 0.75)
    hi = np.where(trained, hi, phys_mw * 1.25)

    stack = np.sort(np.clip(np.column_stack([lo, mid, hi]), 0.0, capacity), axis=1)
    idx = x.index

    return pd.DataFrame(
        {
            "region_id": cfg.region_id,
            "run_ts_utc": idx.get_level_values("run_ts_utc"),
            "valid_ts_utc": idx.get_level_values("valid_ts_utc"),
            "lead_hours": lead.astype("int32"),
            "tech": tech,
            "p10_mw": stack[:, 0],
            "p50_mw": stack[:, 1],
            "p90_mw": stack[:, 2],
            "capacity_mw": capacity,
            "model_version": meta.get("model_version", "unknown"),
            "calibration_date": meta.get("calibration_date", "unknown"),
            "calibrated": trained & (conformal is not None),
        }
    )


def load_run_weather(region_id: str, run_ts: pd.Timestamp, horizons: range) -> pd.DataFrame:
    """Weather rows for one historical run, from the gold table.

    This is the REPLAY path. The live path passes `weather=` into predict()
    instead -- see `src.ingest.live.live_weather`, which produces a frame with
    the same columns so neither the feature builder nor the model can tell the
    difference.

    ponytail: reads the parquet directly rather than via core.store; same frame,
    one import to change if the store ever grows a training-matrix reader.
    """
    root = pathlib.Path(get_settings().data_root)
    path = root / "gold" / "training_base_24_72h" / "part-0.parquet"
    if not path.exists():
        raise RuntimeError(f"no training matrix at {path}; run scripts/build_gold.py")
    df = pd.read_parquet(path)
    sel = df[
        (df["region_id"] == region_id)
        & (df["run_ts_utc"] == run_ts)
        & (df["lead_hours"].isin(list(horizons)))
    ]
    if sel.empty:
        available = df.loc[df["region_id"] == region_id, "run_ts_utc"]
        hint = f" latest available run is {available.max()}" if len(available) else ""
        raise RuntimeError(
            f"no weather rows for {region_id} run {run_ts.isoformat()}.{hint} "
            "For a forecast from now, use the live path (scripts/run_cycle.py --live)."
        )
    return sel.sort_values("valid_ts_utc").reset_index(drop=True)


def empty_forecast() -> pd.DataFrame:
    """The contract's shape, for building against before M4 lands."""
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES.get(c, "float64")) for c in PREDICT_COLUMNS})
    for c in ("run_ts_utc", "valid_ts_utc"):
        df[c] = pd.Series(dtype="datetime64[ns, UTC]")
    return df[PREDICT_COLUMNS]
