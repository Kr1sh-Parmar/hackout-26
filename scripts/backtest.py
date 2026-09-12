"""Walk-forward backtest: the numbers every accuracy claim routes through.

    python scripts/backtest.py --region BE --tech solar --tech wind
    python scripts/backtest.py --region BE --quick        # single fold

Reports four lines per lead hour -- persistence, physics, model, Elia -- because
the GAPS between them are the contribution, and a single 72-hour average hides
where the skill actually is.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.evaluation.baselines import climatology, hourly_actual_cf, persistence  # noqa: E402
from src.evaluation.report import per_lead_hour_report, summarise  # noqa: E402
from src.evaluation.walk_forward import assert_no_overlap, walk_forward  # noqa: E402
from src.features.build import build_all_features  # noqa: E402
from src.models.physics import physics_forecast  # noqa: E402
from src.models.residual_gbdt import predict_residual, train_residual  # noqa: E402
from src.uncertainty.conformal import SplitConformal  # noqa: E402

GOLD = "data/gold/training_base_24_72h/part-0.parquet"
OUT = pathlib.Path("data/gold/backtest")


def fold_predictions(cfg, tech, fold, actuals, cap) -> pd.DataFrame:
    """Fit on this fold's train window, calibrate on its middle, score its test."""

    def prep(part):
        x = build_all_features(cfg.region_id, part, actuals=actuals, tech=tech)
        phys = physics_forecast(cfg, part, tech).to_numpy(dtype=float)
        return x, phys, part[f"y_{tech}_cf"].to_numpy(dtype=float)

    x_tr, phys_tr, y_tr = prep(fold.train)
    models = train_residual(x_tr, pd.Series(y_tr), pd.Series(phys_tr), pd.Series([1.0] * len(x_tr)))

    conformal = None
    if len(fold.calibrate) > 200:
        x_c, phys_c, y_c = prep(fold.calibrate)
        b_c = predict_residual(models, x_c, phys_c, cap)
        conformal = SplitConformal(alpha=0.2).calibrate(
            b_c["p10_mw"].to_numpy(dtype=float),
            b_c["p90_mw"].to_numpy(dtype=float),
            y_c * cap,
            fold.calibrate["lead_hours"].to_numpy(dtype=float),
        )

    te = fold.test
    x_te, phys_te, y_te = prep(te)
    b = predict_residual(models, x_te, phys_te, cap)
    lo = b["p10_mw"].to_numpy(dtype=float)
    hi = b["p90_mw"].to_numpy(dtype=float)
    if conformal is not None:
        lo, hi = conformal.apply(lo, hi, te["lead_hours"].to_numpy(dtype=float))

    cf = hourly_actual_cf(actuals)
    return pd.DataFrame(
        {
            "lead_hours": te["lead_hours"].to_numpy(),
            "y_true": y_te * cap,
            "p50": b["p50_mw"].to_numpy(dtype=float),
            "p10": lo,
            "p90": hi,
            "physics": phys_te * cap,
            "persistence": persistence(te, cf).to_numpy(dtype=float) * cap,
            "climatology": climatology(te, cf).to_numpy(dtype=float) * cap,
            "tso": te[f"tso_{tech}_p50_mw"].to_numpy(dtype=float),
            "is_day": te["is_day"].to_numpy() if tech == "solar" else 1,
        }
    )


def run(region: str, tech: str, quick: bool) -> dict:
    cfg = load_region(region)
    cap = float(cfg.capacity_mw[tech])
    gold = pd.read_parquet(GOLD)
    gold = gold[(gold["region_id"] == region) & (gold[f"sample_weight_{tech}"] == 1)]
    actuals = pd.read_parquet(f"data/silver/generation_actuals_{tech}/part-0.parquet")
    actuals = actuals[actuals["qc_flag"] == "OK"]

    folds = walk_forward(gold, train_months=6, test_months=1, gap_days=1, calibrate_days=45)
    assert_no_overlap(folds)
    if not folds:
        raise SystemExit("no folds -- not enough history")
    if quick:
        folds = folds[-1:]
    print(f"  {len(folds)} fold(s)")

    preds = pd.concat(
        [fold_predictions(cfg, tech, f, actuals, cap) for f in folds], ignore_index=True
    )
    daylight = tech == "solar"
    report = per_lead_hour_report(preds, capacity=cap, daylight_only=daylight)
    metrics = summarise(report)

    print(
        f"  {'lead':>5} {'n':>6} {'model':>7} {'persist':>8} {'physics':>8} {'Elia':>7} {'PICP':>6}"
    )
    for _, r in report[report.lead_hours.isin([24, 36, 48, 60, 72])].iterrows():
        print(
            f"  {int(r.lead_hours):5d} {int(r.n_rows):6,} {r.nrmse_model * 100:6.2f}% "
            f"{r.nrmse_persistence * 100:7.2f}% {r.nrmse_physics * 100:7.2f}% "
            f"{r.nrmse_tso * 100:6.2f}% {r.picp_80:6.3f}"
        )
    print(
        f"  OVERALL nRMSE {metrics['nrmse_mean'] * 100:.2f}%  "
        f"skill {metrics['skill_mean']:.3f}  PICP {metrics['picp_mean']:.3f}  "
        f"(daylight_only={daylight})"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    report.assign(region_id=region, tech=tech).to_parquet(
        OUT / f"region_id={region}_tech={tech}.parquet", index=False
    )
    return {"region": region, "tech": tech, "folds": len(folds), **metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--tech", action="append", choices=["solar", "wind"])
    ap.add_argument("--quick", action="store_true", help="last fold only")
    args = ap.parse_args()

    results = []
    for tech in args.tech or ["solar", "wind"]:
        print(f"== {args.region}/{tech} ==")
        results.append(run(args.region, tech, args.quick))

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/backtest.json").write_text(json.dumps(results, indent=2))
    print("\nwrote artifacts/backtest.json")


if __name__ == "__main__":
    main()
