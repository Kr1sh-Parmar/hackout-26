"""Split conformal calibration of the quantile band (dev-03 SS7).

Per lead hour, because the error at hour 24 and the error at hour 72 are not the
same distribution and a single global width is wrong at both ends.
"""

from __future__ import annotations

import math

import numpy as np

MIN_CALIBRATION_POINTS = 30


class SplitConformal:
    """Finite-sample coverage for an interval that was only ever asymptotic.

    Calibrate on a held-out RECENT window that the model did not train on and
    that sits before the test window. Calibrating on training data produces an
    interval that is confidently wrong -- the residuals there are already fitted.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self.q_hat: dict[int, float] = {}

    def calibrate(self, lo, hi, y, lead_hours) -> SplitConformal:
        lo, hi, y = (np.asarray(v, dtype=float) for v in (lo, hi, y))
        lead = np.asarray(lead_hours).astype(int)
        s = np.maximum(lo - y, y - hi)

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
        q = np.array(
            [self.q_hat.get(int(h), 0.0) for h in np.asarray(lead_hours).astype(int)], dtype=float
        )
        return np.asarray(lo, dtype=float) - q, np.asarray(hi, dtype=float) + q
