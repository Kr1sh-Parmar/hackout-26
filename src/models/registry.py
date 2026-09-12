"""Model artifacts on disk. No MLflow -- a directory and a JSON sidecar.

The sidecar exists so that "which model produced this number?" is answerable
after a bad forecast, which is the only time anyone asks.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib

import joblib

from ..core.config import get_settings

ARTIFACT_SUBDIR = "models"
PICKLE_NAME = "model.joblib"
META_NAME = "metadata.json"

# Provenance that has to survive the person who trained the model. Every one of
# these answers a question asked only after a bad forecast, when the training
# session is long gone: what was it fitted on, in what order, over which window,
# with which hyperparameters, and did those come from tuning or from the defaults.
REQUIRED_META = (
    "model_version",
    "calibration_date",
    "features",
    "train_end",
    "calibrate_end",
    "test_start",
    "params",
    "params_source",
)


def json_safe(obj):
    """Replace non-finite floats with None, recursively.

    `json.dumps` happily emits a bare `NaN`, and `json.loads` reads it back, so
    a Python round-trip never notices. Every other reader rejects it:
    `JSON.parse`, `jq` and Go's decoder all fail on the file. A metric that has
    no value for this run -- `nrmse_tso_wa` when there is no week-ahead TSO
    forecast to compare against -- is null, which every reader can hold.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def model_dir(region: str, tech: str) -> pathlib.Path:
    root = pathlib.Path(get_settings().artifact_root)
    return root / ARTIFACT_SUBDIR / f"{region}_{tech}"


def save_model(models, conformal, meta: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "conformal": conformal}, path / PICKLE_NAME)

    meta = dict(meta)
    missing = [
        k for k in REQUIRED_META if k not in meta and k not in ("model_version", "calibration_date")
    ]
    if missing:
        raise ValueError(f"refusing to save an unprovenanced artifact; metadata missing {missing}")
    meta.setdefault("model_version", dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"))
    meta.setdefault("calibration_date", dt.datetime.now(dt.UTC).date().isoformat())
    # allow_nan=False so a non-finite that json_safe missed fails loudly here
    # rather than shipping a file the frontend cannot parse.
    (path / META_NAME).write_text(
        json.dumps(json_safe(meta), indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
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

    One exception, and it is an outage rather than a judgment call: an incumbent
    that needs a feature the builder no longer produces cannot score a single
    live row -- `residual_gbdt._align` refuses, correctly. Defending its nRMSE
    then keeps a model that only works on paper and wedges the repo in a state
    where nothing can be promoted and /forecast 500s. Calibration still gates.
    """
    calibrated = abs(candidate["picp_mean"] - 0.80) < 0.06
    if incumbent is None:
        return calibrated
    if not calibrated:
        return False
    if not set(incumbent.get("features", [])) <= set(candidate.get("features", [])):
        return True

    sharper = candidate["nrmse_mean"] < incumbent["nrmse_mean"] * 0.995
    # A gate that only knows about nRMSE cannot ship a calibration improvement:
    # conditioning the conformal band on solar elevation left accuracy identical
    # and moved PICP 0.777 -> 0.808 at a NARROWER width, and this refused it.
    # Coverage is half of what the platform promises, so it gates in both
    # directions -- and a candidate that ties on both is still promoted, which
    # keeps the served artifact in step with the code that produced it.
    no_worse = candidate["nrmse_mean"] <= incumbent["nrmse_mean"] * 1.005
    covers_better = abs(candidate["picp_mean"] - 0.80) <= abs(incumbent["picp_mean"] - 0.80) + 0.005
    return sharper or (no_worse and covers_better)
