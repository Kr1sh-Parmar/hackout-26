"""Train the residual quantile models and fit the conformal calibrator.

    python scripts/train.py --region BE --tech solar --tech wind

Three-way CHRONOLOGICAL split: train -> calibrate -> test. The calibration window
must sit after training and before test. Calibrating on rows the model was fitted
on produces an interval that is confidently wrong, which is worse than no
interval at all -- an operator sizes reserves from it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.evaluation.baselines import hourly_actual_cf, persistence  # noqa: E402
from src.evaluation.report import per_lead_hour_report, summarise  # noqa: E402
from src.features.build import build_all_features  # noqa: E402
from src.models.physics import physics_forecast  # noqa: E402
from src.models.registry import load_model, promote, save_model  # noqa: E402
from src.models.residual_gbdt import predict_residual, train_residual  # noqa: E402
from src.uncertainty.conformal import SplitConformal, calibration_mask  # noqa: E402

GOLD = "data/gold/training_base_24_72h/part-0.parquet"
CALIBRATE_DAYS = 45
TEST_DAYS = 90


def load_actuals(tech: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/silver/generation_actuals_{tech}/part-0.parquet")
    return df[df["qc_flag"] == "OK"]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["valid_ts_utc"], utc=True)
    test_start = ts.max() - pd.Timedelta(days=TEST_DAYS)
    cal_start = test_start - pd.Timedelta(days=CALIBRATE_DAYS)
    return (
        df[ts < cal_start],
        df[(ts >= cal_start) & (ts < test_start)],
        df[ts >= test_start],
    )


def run(region: str, tech: str) -> dict:
    cfg = load_region(region)
    cap = float(cfg.capacity_mw[tech])
    gold = pd.read_parquet(GOLD)
    gold = gold[(gold["region_id"] == region) & (gold[f"sample_weight_{tech}"] == 1)]
    actuals = load_actuals(tech)

    tr, cal, te = split(gold)
    print(f"  train {len(tr):,}  calibrate {len(cal):,}  test {len(te):,}")
    if min(len(tr), len(cal), len(te)) < 200:
        raise SystemExit(f"not enough rows for a three-way split on {region}/{tech}")

    def prep(part: pd.DataFrame):
        x = build_all_features(region, part, actuals=actuals, tech=tech)
        phys = physics_forecast(cfg, part, tech).to_numpy(dtype=float)
        y_cf = part[f"y_{tech}_cf"].to_numpy(dtype=float)
        return x, phys, y_cf

    x_tr, phys_tr, y_tr = prep(tr)
    models = train_residual(x_tr, pd.Series(y_tr), pd.Series(phys_tr), pd.Series([1.0] * len(x_tr)))

    # calibrate on the held-out middle window, per lead hour
    x_cal, phys_cal, y_cal = prep(cal)
    b_cal = predict_residual(models, x_cal, phys_cal, cap)
    cmask = calibration_mask(cal, tech)
    conformal = SplitConformal(alpha=0.2).calibrate(
        b_cal["p10_mw"].to_numpy(dtype=float)[cmask],
        b_cal["p90_mw"].to_numpy(dtype=float)[cmask],
        (y_cal * cap)[cmask],
        cal["lead_hours"].to_numpy(dtype=float)[cmask],
    )

    # score on the final window, which neither fitting step has seen
    x_te, phys_te, y_te = prep(te)
    b_te = predict_residual(models, x_te, phys_te, cap)
    lo, hi = conformal.apply(
        b_te["p10_mw"].to_numpy(dtype=float),
        b_te["p90_mw"].to_numpy(dtype=float),
        te["lead_hours"].to_numpy(dtype=float),
    )
    cf = hourly_actual_cf(actuals)
    preds = pd.DataFrame(
        {
            "lead_hours": te["lead_hours"].to_numpy(),
            "y_true": y_te * cap,
            "p50": b_te["p50_mw"].to_numpy(dtype=float),
            "p10": lo,
            "p90": hi,
            "physics": phys_te * cap,
            "persistence": persistence(te, cf).to_numpy(dtype=float) * cap,
            "tso": te[f"tso_{tech}_p50_mw"].to_numpy(dtype=float),
            "is_day": te["is_day"].to_numpy() if tech == "solar" else 1,
        }
    )
    daylight = tech == "solar"
    report = per_lead_hour_report(preds, capacity=cap, daylight_only=daylight)
    metrics = summarise(report)

    print(f"  nRMSE      {metrics['nrmse_mean'] * 100:5.2f}%  (daylight_only={daylight})")
    print(
        f"  persistence{metrics['nrmse_persistence'] * 100:5.2f}%   physics {metrics['nrmse_physics'] * 100:5.2f}%"
    )
    print(f"  Elia       {metrics['nrmse_tso'] * 100:5.2f}%   skill {metrics['skill_mean']:.3f}")
    print(f"  PICP       {metrics['picp_mean']:.3f}  (target 0.78-0.82)")

    version = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "region": region,
        "tech": tech,
        "model_version": version,
        "calibration_date": str(pd.Timestamp.utcnow().date()),
        "train_end": str(tr["valid_ts_utc"].max()),
        "calibrate_end": str(cal["valid_ts_utc"].max()),
        "test_start": str(te["valid_ts_utc"].min()),
        "daylight_only": daylight,
        "features": list(x_tr.columns),
        **{k: float(v) for k, v in metrics.items()},
    }

    try:
        incumbent = load_model(region, tech).get("meta", {})
    except (FileNotFoundError, KeyError):
        incumbent = {}
    if incumbent and not promote(meta, incumbent):
        print(
            f"  PROMOTION REFUSED -- incumbent nRMSE {incumbent.get('nrmse_mean', float('nan')) * 100:.2f}%"
            f" / PICP {incumbent.get('picp_mean', float('nan')):.3f}. Keeping it."
        )
        return meta

    save_model(models, conformal, meta, f"artifacts/models/{region}_{tech}")
    print(f"  promoted {version}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--tech", action="append", choices=["solar", "wind"])
    args = ap.parse_args()
    for tech in args.tech or ["solar", "wind"]:
        print(f"== {args.region}/{tech} ==")
        run(args.region, tech)


if __name__ == "__main__":
    main()
