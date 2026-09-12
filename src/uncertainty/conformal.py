"""Split conformal calibration of the quantile band (dev-03 SS7).

Banded by lead hour, because the error at hour 24 and the error at hour 72 are
not the same distribution and a single global width is wrong at both ends.

BAND WIDTH IS A BIAS/VARIANCE TRADE, AND IT WAS MEASURED.
One q_hat per individual lead hour sounds the most faithful, but a 45-day
calibration window leaves only ~30 usable points per hour, so each q_hat is
itself noisy and 13 of 49 hours fell below the minimum and got no correction at
all. Pooling into 12-hour bands quadruples the evidence behind each estimate.
Measured over 22 walk-forward folds (solar, daylight-only):

    per lead hour   PICP 0.768 +/- 0.106   width 12.5%   36% of folds near nominal
    12-hour bands   PICP 0.793 +/- 0.091   width 11.7%   50%
    24-hour bands   PICP 0.795 +/- 0.094   width 11.7%   41%
    single pooled   PICP 0.796 +/- 0.089   width 11.7%   41%

12-hour bands are better on every axis at once -- closer to nominal, LESS wide,
and less variable fold to fold. Wider pooling buys nothing further and starts
flattening the horizon dependence the band exists to express.
"""

from __future__ import annotations

import math

import numpy as np

MIN_CALIBRATION_POINTS = 30
DEFAULT_BAND_HOURS = 12


class SplitConformal:
    """Finite-sample coverage for an interval that was only ever asymptotic.

    Calibrate on a held-out RECENT window that the model did not train on and
    that sits before the test window. Calibrating on training data produces an
    interval that is confidently wrong -- the residuals there are already fitted.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        band_hours: int = DEFAULT_BAND_HOURS,
        scaled: bool = True,
    ) -> None:
        """`scaled=True` normalises the nonconformity score by the band width.

        AGGREGATE COVERAGE IS NOT CONDITIONAL COVERAGE, AND ONLY ONE OF THEM
        SIZES A RESERVE. An additive q_hat adds the SAME megawatts at 05:00 and
        at 13:00, but solar error is heteroscedastic within the day: near zero at
        dawn, largest at midday. The result is a band that is far too generous at
        dawn and too narrow exactly when output is largest -- the aggregate looks
        calibrated while almost every individual hour is not.

        Measured, solar, 22 walk-forward folds, daylight-only:

            additive   PICP 0.794   width 11.5%    2/34 lead hours in 0.78-0.82
            scaled     PICP 0.788   width 12.8%   10/34 lead hours in 0.78-0.82

        Same aggregate, five times the conditional coverage, and the per-hour
        spread tightens from 0.63-1.00 to 0.72-0.96. The extra 1.3pp of width is
        the honest cost of covering the hard hours instead of padding the easy
        ones.
        """
        self.alpha = float(alpha)
        self.band_hours = max(int(band_hours), 1)
        self.scaled = bool(scaled)
        self.q_hat: dict[int, float] = {}

    def __setstate__(self, state: dict) -> None:
        """Unpickle artifacts written before `band_hours`/`scaled` existed.

        A model artifact outlives the code that produced it. Without this, an
        older pickle loads with the attributes simply missing and the first
        `apply()` dies with AttributeError inside the serving path -- long after
        the deploy that caused it. Defaults reproduce the historical behaviour
        (per-lead-hour, additive), so an old artifact keeps predicting exactly
        what it predicted before; only a retrain opts into the new scheme.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("band_hours", 1)
        self.__dict__.setdefault("scaled", False)

    def _band(self, lead_hours) -> np.ndarray:
        lead = np.asarray(lead_hours, dtype=float)
        return ((lead // self.band_hours) * self.band_hours).astype(int)

    def calibrate(self, lo, hi, y, lead_hours) -> SplitConformal:
        lo, hi, y = (np.asarray(v, dtype=float) for v in (lo, hi, y))
        lead = self._band(lead_hours)
        s = np.maximum(lo - y, y - hi)
        if self.scaled:
            # a RATIO of the band, so the correction inherits the quantile
            # model's own condition-dependent width instead of flattening it
            s = s / np.maximum(hi - lo, 1e-6)

        for lead_h in np.unique(lead):
            sel = s[(lead == lead_h) & np.isfinite(s)]
            n = sel.size
            if n < MIN_CALIBRATION_POINTS:
                # Too few points to make a finite-sample claim. Widening by a
                # number we cannot justify is worse than not widening.
                self.q_hat[int(lead_h)] = 0.0
                continue
            k = min(math.ceil((n + 1) * (1 - self.alpha)), n)
            self.q_hat[int(lead_h)] = float(np.sort(sel)[k - 1])
        return self

    def apply(self, lo, hi, lead_hours):
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        q = np.array([self.q_hat.get(int(h), 0.0) for h in self._band(lead_hours)], dtype=float)
        if self.scaled:
            q = q * np.maximum(hi - lo, 1e-6)
        return lo - q, hi + q


def calibration_mask(frame, tech: str):
    """Rows a conformal calibrator should use for `tech`.

    The calibration set must match the set the model is SCORED on. Solar is
    evaluated daylight-only, because roughly half a solar series is night zeros
    that any model covers trivially. Calibrating on those rows too dilutes the
    nonconformity scores, shrinks q_hat and produces a band that is too narrow
    exactly where it matters -- measured PICP 0.767 against a 0.80 nominal,
    versus 0.798 when calibrated on daylight alone (and at a tighter width).
    """
    import numpy as _np

    if tech == "solar" and "is_day" in getattr(frame, "columns", []):
        return (frame["is_day"] == 1).to_numpy()
    return _np.ones(len(frame), dtype=bool)
