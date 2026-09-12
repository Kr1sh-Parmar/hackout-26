"""Per-lead-hour scoring -- integration point I1.

A single 72-hour-average nRMSE hides everything that matters. Skill decays with
lead time, and the shape of that decay is the most convincing evidence that a
forecasting system understands where its skill comes from. Always report the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ace, interval_width, mbe, nmae, nrmse, picp, pinball, skill_score

# The frame `per_lead_hour_report` expects. Track A produces it; Track C scores it.
REQUIRED = ["y_true", "p50", "lead_hours"]
OPTIONAL = ["p10", "p90", "persistence", "physics", "tso", "tso_wa", "is_day", "climatology"]

REPORT_COLUMNS = [
    "lead_hours",
    "n_rows",
    "nrmse_model",
    "nrmse_persistence",
    "nrmse_physics",
    "nrmse_tso",
    "nrmse_tso_wa",
    "skill_vs_persistence",
    "skill_vs_physics",
    "nmae_model",
    "mbe_model",
    "picp_80",
    "ace_80",
    "mean_width_frac",
    "pinball_mean",
]


def per_lead_hour_report(
    preds: pd.DataFrame,
    capacity: float,
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Score a prediction frame, one row per lead hour.

    Args:
        preds: `y_true`, `p50`, `lead_hours` required; `p10`/`p90`/`persistence`/
            `physics`/`tso`/`is_day` optional and scored when present. All values
            in the SAME units -- capacity factor or MW -- matching `capacity`.
        capacity: installed capacity, the normaliser for every error metric.
        daylight_only: drop `is_day == 0` rows. Mandatory for solar: roughly half
            a solar series is zeros that any model predicts perfectly, so including
            them roughly halves the reported error and means nothing.

    Returns:
        DataFrame with REPORT_COLUMNS, one row per lead hour, ascending.
    """
    missing = [c for c in REQUIRED if c not in preds.columns]
    if missing:
        raise KeyError(f"per_lead_hour_report needs {missing}; got {list(preds.columns)}")

    d = preds
    if daylight_only:
        if "is_day" not in d.columns:
            raise KeyError("daylight_only=True requires an `is_day` column")
        d = d[d["is_day"] == 1]

    rows = []
    for lh, g in d.groupby("lead_hours", sort=True):
        has = lambda c: c in g.columns and g[c].notna().any()  # noqa: E731

        row: dict[str, float | int] = {
            "lead_hours": int(lh),
            "n_rows": int(len(g)),
            "nrmse_model": nrmse(g.y_true, g.p50, capacity),
            "nmae_model": nmae(g.y_true, g.p50, capacity),
            "mbe_model": mbe(g.y_true, g.p50, capacity),
        }
        row["nrmse_persistence"] = (
            nrmse(g.y_true, g.persistence, capacity) if has("persistence") else np.nan
        )
        row["nrmse_physics"] = nrmse(g.y_true, g.physics, capacity) if has("physics") else np.nan
        # Elia day-ahead (~6-30 h lead) and week-ahead (~144 h+) bracket our
        # 24-72 h band. Reporting both is the honest framing: neither alone is
        # a like-for-like comparator at every lead hour.
        row["nrmse_tso"] = nrmse(g.y_true, g.tso, capacity) if has("tso") else np.nan
        row["nrmse_tso_wa"] = nrmse(g.y_true, g.tso_wa, capacity) if has("tso_wa") else np.nan
        row["skill_vs_persistence"] = (
            skill_score(g.y_true, g.p50, g.persistence, capacity) if has("persistence") else np.nan
        )
        row["skill_vs_physics"] = (
            skill_score(g.y_true, g.p50, g.physics, capacity) if has("physics") else np.nan
        )

        if has("p10") and has("p90"):
            row["picp_80"] = picp(g.y_true, g.p10, g.p90)
            row["ace_80"] = ace(g.y_true, g.p10, g.p90, nominal=0.80)
            row["mean_width_frac"] = interval_width(g.p10, g.p90, capacity)
            row["pinball_mean"] = float(
                np.nanmean(
                    [
                        pinball(g.y_true, g.p10, 0.1),
                        pinball(g.y_true, g.p50, 0.5),
                        pinball(g.y_true, g.p90, 0.9),
                    ]
                )
            )
        else:
            row["picp_80"] = row["ace_80"] = row["mean_width_frac"] = np.nan
            row["pinball_mean"] = pinball(g.y_true, g.p50, 0.5)

        rows.append(row)

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def summarise(report: pd.DataFrame) -> dict[str, float]:
    """Headline numbers, weighted by row count so sparse lead hours cannot dominate.

    `lead_hours == 72` carries far fewer rows than the rest of the band, so a plain
    mean over lead hours would let the noisiest slice move the headline.
    """
    w = report["n_rows"].to_numpy(dtype=float)
    if w.sum() == 0:
        return {}

    def wmean(col: str) -> float:
        # An absent optional column is not an error: a report built without a
        # benchmark should still summarise the columns it does have.
        if col not in report.columns:
            return float("nan")
        v = report[col].to_numpy(dtype=float)
        m = np.isfinite(v)
        return float(np.average(v[m], weights=w[m])) if m.any() else float("nan")

    model = wmean("nrmse_model")
    persist = wmean("nrmse_persistence")
    physics = wmean("nrmse_physics")

    def agg_skill(ref: float) -> float:
        """Skill from the AGGREGATED errors, never a mean of per-lead ratios.

        Averaging ratios is not the ratio of averages: one lead hour where the
        reference happens to be nearly perfect makes `1 - a/b` explode, and a
        single such row drags the headline to nonsense (observed: -5.35 when the
        true aggregate skill was +0.57).
        """
        if not np.isfinite(model) or not np.isfinite(ref) or ref < 1e-12:
            return float("nan")
        return float(1.0 - model / ref)

    return {
        "nrmse_mean": model,
        "nrmse_persistence": persist,
        "nrmse_physics": physics,
        "nrmse_tso": wmean("nrmse_tso"),
        "nrmse_tso_wa": wmean("nrmse_tso_wa"),
        "skill_mean": agg_skill(persist),
        "skill_vs_physics": agg_skill(physics),
        "mbe_mean": wmean("mbe_model"),
        "picp_mean": wmean("picp_80"),
        "n_rows": float(w.sum()),
    }
