"""PV curtailment detection.

Wind curtailment comes free from Elia's `decrementalbidid` flag. PV has no such
flag, so curtailed intervals have to be inferred from the shape of the signal:
output flat-lining well below what the physics chain says the weather allowed.

Curtailed hours are CENSORED LABELS -- the measured output is not what the
weather would have produced -- so training on them as-is teaches the model to
under-forecast exactly during the events this platform exists to predict.
`sample_weights` zeroes them (and other untrustworthy rows) out instead of
dropping them, which keeps the row's other features (lags, calendar) available
without letting its label pull the fit down.

Thresholds are heuristic constants, not `RegionConfig` fields: they describe
the *shape of a curtailment signature* (flat, sunny, underperforming), not a
region's economics or hardware -- picking a different `capacity_mw` should not
change what counts as "flat".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# good generating conditions: bright and daytime
KT_GOOD_THRESHOLD = 0.6
# output barely moving over the trailing window (rolling std < 0.2% of capacity)
FLATLINE_STD_FRAC = 0.002
# output under 75% of what physics says the weather should have allowed
RATIO_THRESHOLD = 0.75
# floor on the physics denominator so near-zero physics estimates don't blow up
# the ratio (e.g. right at sunrise/sunset)
PHYSICS_FLOOR_FRAC = 0.02
ROLLING_WINDOW = 4

BAD_QC_FLAGS = {"MISSING", "FROZEN", "OUT_OF_RANGE", "CURTAILED"}
LOW_AVAILABILITY_PCT = 90.0


def curtailment_flag(bid_id: pd.Series) -> pd.Series:
    """Wind curtailment from Elia's `decrementalbidid` column.

    Lived inline in `scripts/build_elia.py` as a chain of six string calls,
    which is why the trap it defuses went unnoticed once already and could not
    be tested at all.

    The trap: the column is 100% NON-NULL and ~99.2% empty, because Elia's CSV
    export writes an empty text field as the two-character string `''` -- two
    apostrophes -- rather than as an empty field or a null. A `.notna()` test
    therefore flags 99.6% of the wind fleet as curtailed, against a real rate of
    2.96%, and every downstream sample weight is wrong in a way that looks
    plausible. Strip the quotes before testing for emptiness.
    """
    cleaned = bid_id.fillna("").astype(str).str.strip().str.strip("'\"").str.strip()
    return cleaned.ne("")


def detect_curtailment(
    actual_mw: pd.Series,
    physics_mw: pd.Series,
    capacity: float,
    kt: pd.Series,
    is_day: pd.Series,
) -> pd.Series:
    """Flag hours where output flat-lines well below the physics estimate
    under conditions that should support more.

    All inputs must share the same (time-ordered) index. Returns an int8
    series of 0/1, same index as `actual_mw`.
    """
    actual = pd.Series(actual_mw).astype(float)
    physics = pd.Series(physics_mw).astype(float).reindex(actual.index)
    kt = pd.Series(kt).astype(float).reindex(actual.index)
    is_day = pd.Series(is_day).reindex(actual.index)

    good = (kt > KT_GOOD_THRESHOLD) & (is_day == 1)
    flatline = actual.rolling(ROLLING_WINDOW).std() < capacity * FLATLINE_STD_FRAC
    ratio = actual / physics.clip(lower=capacity * PHYSICS_FLOOR_FRAC)
    underperforming = ratio < RATIO_THRESHOLD

    flagged = (good & flatline & underperforming).fillna(False)
    return flagged.astype("int8").rename("curtailed")


def sample_weights(df: pd.DataFrame) -> pd.Series:
    """0 for rows that should not teach the model anything, 1 otherwise.

    Zeroed when: `curtailed` is truthy, `qc_flag` marks the row as
    MISSING/FROZEN/OUT_OF_RANGE/CURTAILED, or `availability_pct` < 90%.
    Any column that is absent is treated as "not a problem" for that check.
    """
    n = len(df)
    weight = pd.Series(np.ones(n), index=df.index, dtype=float)

    if "curtailed" in df.columns:
        weight = weight.where(df["curtailed"].fillna(0).astype(int) == 0, 0.0)
    if "qc_flag" in df.columns:
        weight = weight.where(~df["qc_flag"].isin(BAD_QC_FLAGS), 0.0)
    if "availability_pct" in df.columns:
        low_avail = df["availability_pct"] < LOW_AVAILABILITY_PCT
        weight = weight.where(~low_avail.fillna(False), 0.0)

    return weight.rename("sample_weight")
