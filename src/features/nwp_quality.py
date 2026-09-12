"""NWP disagreement and ramp features (dev-03 SS4).

Cross-model spread is already computed in ETL (`<var>_model_std` / `_model_range`).
These features predict our own error, which is what turns a point forecast into
an honest interval.
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
    "nwp_bias_lag_7d",
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
    # ponytail: no error history reaches a pure feature function, so the bias
    # term is a structural zero. Upgrade path: pass a lead-keyed bias table in
    # and look it up here -- it stays pure, it just needs the table as an arg.
    out["nwp_bias_lag_7d"] = 0.0
    return out
