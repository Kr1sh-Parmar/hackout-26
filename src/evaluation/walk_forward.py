"""Temporal splitting.

Adjacent hours are near-identical, so a random split puts almost the same row on
both sides of the fence. Every number that comes out of it is inflated, and the
error only shows up in production. There is no `train_test_split` anywhere in this
repository and there should never be one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TS = "valid_ts_utc"


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold. `calibrate` is empty for the two-way split."""

    train: pd.DataFrame
    calibrate: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        def span(d: pd.DataFrame) -> str:
            if d.empty:
                return "empty"
            return f"{d[TS].min():%Y-%m-%d}..{d[TS].max():%Y-%m-%d} n={len(d):,}"

        return f"Fold(train={span(self.train)}, cal={span(self.calibrate)}, test={span(self.test)})"


def walk_forward(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
    gap_days: int = 1,
    calibrate_days: int = 0,
    ts_col: str = TS,
) -> list[Fold]:
    """Expanding-window folds in time order, separated by a gap.

    The gap matters because a 72-hour forecast issued just before the boundary is
    still describing hours that fall inside the test window; without it the model
    has effectively seen the start of its own test set.

    Set `calibrate_days > 0` for the three-way split conformal prediction needs:
    train -> calibrate -> gap -> test, strictly in time order. Calibrating on data
    the model was fitted on produces an interval that is confidently wrong.
    """
    if ts_col not in df.columns:
        raise KeyError(f"{ts_col!r} not in frame; got {list(df.columns)[:8]}...")

    d = df.sort_values(ts_col)
    ts = pd.to_datetime(d[ts_col], utc=True)
    start, end = ts.min().normalize(), ts.max()

    folds: list[Fold] = []
    train_end = start + pd.DateOffset(months=train_months)

    while True:
        cal_end = train_end + pd.Timedelta(days=calibrate_days)
        test_start = cal_end + pd.Timedelta(days=gap_days)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        folds.append(
            Fold(
                train=d[ts < train_end],
                calibrate=d[(ts >= train_end) & (ts < cal_end)],
                test=d[(ts >= test_start) & (ts < test_end)],
            )
        )
        train_end += pd.DateOffset(months=test_months)

    return folds


def assert_no_overlap(folds: list[Fold], ts_col: str = TS) -> None:
    """Fail loudly if any fold leaks. Cheap to run, catches a whole error class."""
    for i, f in enumerate(folds):
        for name, part in (("train", f.train), ("calibrate", f.calibrate)):
            if part.empty or f.test.empty:
                continue
            latest = pd.to_datetime(part[ts_col], utc=True).max()
            earliest = pd.to_datetime(f.test[ts_col], utc=True).min()
            if latest >= earliest:
                raise AssertionError(
                    f"fold {i}: {name} ends {latest} but test starts {earliest} -- overlap"
                )
