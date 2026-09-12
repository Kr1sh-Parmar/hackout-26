"""Calendar features (dev-03 SS4). Cyclic, so hour 23 sits next to hour 0."""

from __future__ import annotations

import numpy as np
import pandas as pd

TEMPORAL_COLUMNS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend"]


def temporal_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(idx)
    hour = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (idx.dayofweek >= 5).astype(float),
        },
        index=idx,
    )
