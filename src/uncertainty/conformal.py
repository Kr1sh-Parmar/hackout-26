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

# Group keys stay plain ints -- `band * COND_STRIDE + bucket` -- so `q_hat` is
# still an int-keyed dict and an artifact pickled before conditioning existed
# unpickles and behaves exactly as it did.
COND_STRIDE = 1000

# The scaled score divides by the band width, so a near-zero band makes the
# ratio explode. Measured: bucketing solar by elevation with a <10 degree bucket
# produced a q_hat off dawn bands whose widths are a few MW, and applying it to
# a midday band served an interval 425x installed capacity. Floor the divisor at
# a fraction of the calibration set's own median width -- below that the band is
# too small to normalise by and the additive behaviour is the safe one.
SCALE_FLOOR_FRAC = 0.05


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
        self.scale_floor = 1e-6
        self.conditioned = False

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
        self.__dict__.setdefault("scale_floor", 1e-6)
        self.__dict__.setdefault("conditioned", False)

    def _band(self, lead_hours) -> np.ndarray:
        lead = np.asarray(lead_hours, dtype=float)
        return ((lead // self.band_hours) * self.band_hours).astype(int)

    def _keys(self, lead_hours, cond=None) -> np.ndarray:
        """Calibration group per row: the lead band, optionally crossed with a
        caller-supplied conditioning bucket (see `calibrate`)."""
        band = self._band(lead_hours)
        if cond is None:
            return band
        c = np.asarray(cond, dtype=int)
        if c.size and (c.min() < 0 or c.max() >= COND_STRIDE):
            raise ValueError(
                f"cond buckets must be in [0, {COND_STRIDE}); got {c.min()}..{c.max()}"
            )
        return band * COND_STRIDE + c

    def calibrate(self, lo, hi, y, lead_hours, cond=None) -> SplitConformal:
        """Fit one q_hat per calibration group.

        `cond` is an optional per-row bucket crossed with the lead band -- for
        solar, a solar-elevation bucket. Lead band alone answers "how far ahead
        is this", which for a 00Z run is also "what time of day is this", but
        only modulo 24: hours 25, 49 and 73 are all 01:00 and land in three
        different bands. Crossing in an elevation bucket lets the correction
        differ between dawn and midday WITHIN a band, which is where solar's
        remaining conditional miscoverage lives.
        """
        lo, hi, y = (np.asarray(v, dtype=float) for v in (lo, hi, y))
        self.conditioned = cond is not None
        lead = self._keys(lead_hours, cond)
        s = np.maximum(lo - y, y - hi)
        if self.scaled:
            width = hi - lo
            finite = width[np.isfinite(width)]
            self.scale_floor = max(
                float(SCALE_FLOOR_FRAC * np.median(finite)) if finite.size else 0.0, 1e-6
            )
            # a RATIO of the band, so the correction inherits the quantile
            # model's own condition-dependent width instead of flattening it
            s = s / np.maximum(width, self.scale_floor)

        for key in np.unique(lead):
            sel = s[(lead == key) & np.isfinite(s)]
            n = sel.size
            if n < MIN_CALIBRATION_POINTS:
                # Too few points to make a finite-sample claim. Widening by a
                # number we cannot justify is worse than not widening.
                self.q_hat[int(key)] = 0.0
                continue
            k = min(math.ceil((n + 1) * (1 - self.alpha)), n)
            self.q_hat[int(key)] = float(np.sort(sel)[k - 1])
        return self

    def apply(self, lo, hi, lead_hours, cond=None):
        """Widen the band. `cond` must match what `calibrate` was given.

        Mismatching it is a SILENT failure, not a loud one: the group keys simply
        miss, every `q_hat` lookup falls back to 0.0, and the caller is handed an
        uncalibrated band that still says `calibrated=True`. So a conditioned
        calibrator refuses to apply without buckets, and an unconditioned one
        ignores buckets it never fitted rather than looking up keys it lacks.
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if self.conditioned and cond is None:
            raise ValueError(
                "this calibrator was fitted with conditioning buckets; applying it without "
                "them would silently return an uncalibrated band"
            )
        if not self.conditioned:
            cond = None
        keys = self._keys(lead_hours, cond)
        q = np.array([self.q_hat.get(int(k), 0.0) for k in keys], dtype=float)
        if self.scaled:
            q = q * np.maximum(hi - lo, self.scale_floor)
        return lo - q, hi + q


# Solar error is heteroscedastic WITHIN the day, and the lead band only knows
# about it modulo 24: hours 25, 49 and 73 are all 01:00 and sit in three
# different bands. These edges cross an elevation bucket into the band so the
# correction can differ between dawn and midday. Measured, 22 walk-forward
# folds, solar daylight-only, against the 12h-band incumbent:
#
#     no conditioning        PICP 0.787  width 12.73%   9/34 hours in 0.78-0.82
#     2 buckets (<20)        PICP 0.787  width 13.38%  11/34
#     3 buckets (<15,35)     PICP 0.782  width 13.14%  12/34   <- adopted
#     4 buckets (<10,25,40)  PICP 0.776  width 13.11%   9/34
#
# Mean |PICP_h - 0.80| across lead hours falls 0.0526 -> 0.0424, so both the
# coarse count and the continuous measure agree. Four buckets over-splits: the
# dawn bucket stops having enough points to estimate from. The cost is 0.41pp of
# width, and the aggregate stays inside the 0.78-0.82 target.
SOLAR_ELEVATION_EDGES: tuple[float, ...] = (15.0, 35.0)


def conditioning_buckets(frame, tech: str):
    """Per-row conformal conditioning bucket for `tech`, or None.

    Pairs with `calibration_mask`: same module, same shape of decision, and the
    caller passes the result of both into `calibrate`/`apply`. Wind gets None --
    its coverage is already 37/49 lead hours in band and it has no within-day
    structure of this kind to condition on.
    """
    import numpy as _np

    if tech != "solar" or "solar_elevation_deg" not in getattr(frame, "columns", []):
        return None
    elev = frame["solar_elevation_deg"].to_numpy(dtype=float)
    return _np.digitize(_np.nan_to_num(elev, nan=-90.0), SOLAR_ELEVATION_EDGES)


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
