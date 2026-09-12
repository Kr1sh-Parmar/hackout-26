"""NWP disagreement and ramp features (dev-03 SS4).

Cross-model spread is already computed in ETL (`<var>_model_std` / `_model_range`).
These features predict our own error, which is what turns a point forecast into
an honest interval.

`nwp_bias_lag_7d` used to live here as a constant 0.0: a pure feature function
has no error history to compute a bias from, so it never carried information.
Re-measured after removal, every backtest number is bit-identical -- LightGBM
discards a zero-variance column at binning, so the feature was only ever
decorating the metadata sidecar with a driver that drove nothing. The upgrade
path if it is ever wanted: pass a lead-keyed bias table in as an argument and
look it up here. That keeps the function pure; it just needs the table, and the
table has to be built identically on the training and serving paths or it
becomes train/serve skew in the one family the parity test cannot check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NWP_COLUMNS = [
    "ghi_model_disagreement",
    "ws_model_disagreement",
    "ghi_model_range",
    "nwp_ghi_ramp",
    "nwp_ws_ramp",
]

_MAP = {
    "ghi_model_disagreement": "ghi_wm2_model_std",
    "ws_model_disagreement": "wind_speed_100m_ms_model_std",
    "ghi_model_range": "ghi_wm2_model_range",
}


def _ramp(wx: pd.DataFrame, col: str) -> np.ndarray:
    """Hour-over-hour change within a single NWP run, never across runs."""
    s = wx[col].astype(float)
    if "run_ts_utc" in wx.columns:
        return s.groupby(wx["run_ts_utc"].to_numpy()).diff().fillna(0.0).to_numpy()
    return s.diff().fillna(0.0).to_numpy()


def nwp_quality_features(wx: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.DatetimeIndex(wx.index))
    for name, src in _MAP.items():
        out[name] = wx[src].to_numpy(dtype=float) if src in wx.columns else np.nan
    out["nwp_ghi_ramp"] = _ramp(wx, "ghi_wm2")
    out["nwp_ws_ramp"] = _ramp(wx, "wind_speed_100m_ms")
    return out
