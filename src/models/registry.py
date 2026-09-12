"""Model artifacts on disk. No MLflow -- a directory and a JSON sidecar.

The sidecar exists so that "which model produced this number?" is answerable
after a bad forecast, which is the only time anyone asks.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import joblib

from ..core.config import get_settings

ARTIFACT_SUBDIR = "models"
PICKLE_NAME = "model.joblib"
META_NAME = "metadata.json"


def model_dir(region: str, tech: str) -> pathlib.Path:
    root = pathlib.Path(get_settings().artifact_root)
    return root / ARTIFACT_SUBDIR / f"{region}_{tech}"


def save_model(models, conformal, meta: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "conformal": conformal}, path / PICKLE_NAME)

    meta = dict(meta)
    meta.setdefault("model_version", dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"))
    meta.setdefault("calibration_date", dt.datetime.now(dt.UTC).date().isoformat())
    (path / META_NAME).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_model(region: str, tech: str) -> dict:
    path = model_dir(region, tech)
    blob = path / PICKLE_NAME
    if not blob.exists():
        raise FileNotFoundError(f"no model artifact at {path}; train {region}/{tech} first")
    payload = joblib.load(blob)
    meta_file = path / META_NAME
    payload["meta"] = (
        json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    )
    return payload


def promote(candidate: dict, incumbent: dict | None) -> bool:
    """Promote only on a real improvement that is still honestly calibrated.

    0.5% is the noise floor of a seasonal backtest; anything smaller is a
    coin flip dressed as progress. A sharper model with a broken band is a
    regression, so PICP gates independently of nRMSE.
    """
    calibrated = abs(candidate["picp_mean"] - 0.80) < 0.06
    if incumbent is None:
        return calibrated
    return calibrated and candidate["nrmse_mean"] < incumbent["nrmse_mean"] * 0.995
