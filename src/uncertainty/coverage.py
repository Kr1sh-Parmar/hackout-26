"""Did the interval mean what it claimed? (dev-03 SS7)"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_report(lo, hi, y, lead_hours, nominal: float = 0.8) -> pd.DataFrame:
    """PICP and mean width per lead hour.

    `ace` is the average coverage error, signed: negative means the band is too
    narrow, which is the failure that costs money.
    """
    df = pd.DataFrame(
        {
            "lead_hours": np.asarray(lead_hours).astype(int),
            "inside": (np.asarray(y, dtype=float) >= np.asarray(lo, dtype=float))
            & (np.asarray(y, dtype=float) <= np.asarray(hi, dtype=float)),
            "width": np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float),
            "ok": np.isfinite(np.asarray(y, dtype=float)),
        }
    )
    df = df[df["ok"]]
    out = (
        df.groupby("lead_hours")
        .agg(picp=("inside", "mean"), mean_width=("width", "mean"), n=("inside", "size"))
        .reset_index()
    )
    out["ace"] = out["picp"] - nominal
    return out[["lead_hours", "picp", "mean_width", "n", "ace"]]
