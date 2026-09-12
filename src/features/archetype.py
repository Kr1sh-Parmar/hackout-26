"""Archetype aggregation and share fitting (dev-03 SS5).

Spatial and NWP-model aggregation already happened in ETL. This layer folds the
ARCHETYPES of one region into a single regional row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..core.config import RegionConfig, SiteMaster

# Solar geometry is identical for every archetype in a region -- same centroid,
# same instant. Averaging it would be arithmetically harmless and conceptually
# wrong, so take it straight from the first part.
#
# NOTE: wind cut-in/cut-out flags are NOT in this list. Onshore cuts out at
# 25 m/s and offshore at 27 m/s from a different hub height, so archetype 0's
# flag is not the fleet's flag. Weighted, they become the fraction of fleet
# capacity at risk, which is what the decision layer actually wants to know.
GEOMETRIC_COLUMNS = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "solar_elevation_deg",
    "airmass",
    "is_day",
)


def aggregate_archetypes(parts: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    """Capacity-weighted mean of archetype feature frames.

    Anything in MW is SUMMED, never weighted. `SiteMaster.capacity_mw` is
    already `total * share`, so every absolute-power column arrives pre-scaled
    by its own share; weighting it again returns `total * sum(share^2)` -- a 65%
    deflation for Belgian solar that still looks like a plausible number.
    Intensive columns (capacity factors, temperatures, flags) are weighted.
    """
    if not parts:
        raise ValueError("aggregate_archetypes needs at least one part")
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("archetype weights must sum to > 0")
    w = w / w.sum()

    numeric = parts[0].select_dtypes(include="number").columns
    out = sum(p[numeric].to_numpy(dtype=float) * wi for p, wi in zip(parts, w, strict=True))
    out = pd.DataFrame(out, columns=numeric, index=parts[0].index)

    for col in parts[0].columns:
        if col in GEOMETRIC_COLUMNS or col not in numeric:
            out[col] = parts[0][col]
    for col in numeric:
        if col.endswith("_mw"):
            out[col] = sum(p[col].to_numpy(dtype=float) for p in parts)
    return out[parts[0].columns]


def fit_shares(
    wx: pd.DataFrame,
    actual_mw: pd.Series,
    cfg: RegionConfig,
    tech: str = "solar",
) -> dict[str, float]:
    """Re-estimate archetype shares from clear-sky days only.

    Cloudy hours are dominated by NWP irradiance error; clear hours are
    dominated by orientation. Fitting on everything estimates the cloud error,
    not the fleet.
    """
    from ..core.config import site_master
    from .build import _as_indexed
    from .solar import solar_features
    from .wind import wind_features

    if tech != "solar":
        # There is no clear-sky analogue for wind, so the fit has nothing to
        # separate ORIENTATION from NWP wind-speed bias and simply votes for the
        # archetype whose power curve best absorbs that bias -- it returns
        # onshore 1.0 / offshore 0.0, contradicting the 0.6224/0.3776 measured
        # directly from Elia's own offshore/onshore capacity split. Measured
        # beats fitted; refuse rather than quietly return a worse number.
        raise ValueError(
            "fit_shares is solar-only: wind shares come from measured capacity in "
            "silver/generation_actuals_wind_segment, which is strictly better evidence"
        )

    sites: list[SiteMaster] = site_master(cfg, tech)
    kernel = solar_features if tech == "solar" else wind_features
    col = "physics_pac_mw" if tech == "solar" else "physics_power_mw"

    # the physics kernels need a tz-aware DatetimeIndex; build_features does this
    # for its own callers, and calling a kernel directly must do the same
    wx = _as_indexed(wx)
    parts = [kernel(wx, s) for s in sites]
    # Unit-share basis: each column is the region at 100% of that archetype.
    basis = np.column_stack(
        [p[col].to_numpy(dtype=float) / s.capacity_share for p, s in zip(parts, sites, strict=True)]
    )
    # Align POSITIONALLY, not by label: `wx` has just been re-indexed to a
    # DatetimeIndex while `actual_mw` still carries the caller's index, so a
    # reindex silently yields all-NaN, trips the `mask.sum() < 50` guard and
    # returns the prior unchanged -- a fit that looks like it ran and did nothing.
    if len(actual_mw) != len(wx):
        raise ValueError(
            f"actual_mw has {len(actual_mw)} rows but weather has {len(wx)}; "
            "they must be row-aligned"
        )
    y = np.asarray(actual_mw, dtype=float)

    mask = np.isfinite(y) & np.isfinite(basis).all(axis=1)
    if tech == "solar":
        clear = parts[0]
        mask &= (clear["clearsky_index_kt"].to_numpy() > 0.85) & (clear["is_day"].to_numpy() == 1)
    if mask.sum() < 50:
        return {s.archetype.id: s.capacity_share for s in sites}

    a, b = basis[mask], y[mask]
    scale = float((b**2).mean()) or 1.0

    # A free gain alongside the shares. Without it the fit is degenerate: the
    # physics chain has no fleet loss stack (soiling, snow, shading,
    # availability) so it runs ~60% hot, and a simplex-constrained fit absorbs
    # that level error by voting for whichever orientation is dimmest. The gain
    # takes the level, which is what leaves SHAPE -- i.e. orientation -- to the
    # shares. Measured for Belgium: gain ~0.64, SSE 0.60 -> 0.03.
    def sse(p):
        return float(((p[0] * (a @ p[1:]) - b) ** 2).mean()) / scale

    x0 = np.r_[1.0, [s.capacity_share for s in sites]]
    res = minimize(
        sse,
        x0,
        method="SLSQP",
        bounds=[(0.2, 1.5)] + [(0.0, 1.0)] * len(sites),
        constraints=[{"type": "eq", "fun": lambda p: p[1:].sum() - 1.0}],
    )
    w = res.x[1:] if res.success else x0[1:]
    w = w / w.sum()
    return {s.archetype.id: float(v) for s, v in zip(sites, w, strict=True)}
