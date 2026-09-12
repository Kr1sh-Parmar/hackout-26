"""Train the residual quantile models and fit the conformal calibrator.

    python scripts/train.py --region BE --tech solar --tech wind

Three-way CHRONOLOGICAL split: train -> calibrate -> test. The calibration window
must sit after training and before test. Calibrating on rows the model was fitted
on produces an interval that is confidently wrong, which is worse than no
interval at all -- an operator sizes reserves from it.

Progress is visible on purpose: a tqdm bar over the stages, LightGBM's own
per-iteration validation loss, and a wall-clock number per stage. A training run
that prints nothing for four minutes is indistinguishable from a hung one.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.evaluation.baselines import hourly_actual_cf, persistence  # noqa: E402
from src.evaluation.report import per_lead_hour_report, summarise  # noqa: E402
from src.features.build import build_all_features  # noqa: E402
from src.models.physics import physics_forecast  # noqa: E402
from src.models.registry import load_model, promote, save_model  # noqa: E402
from src.models.residual_gbdt import (  # noqa: E402
    QUANTILES,
    fitted_features,
    load_params,
    predict_residual,
    train_residual,
)
from src.uncertainty.conformal import (  # noqa: E402
    SplitConformal,
    calibration_mask,
    conditioning_buckets,
)

GOLD = "data/gold/training_base_24_72h/part-0.parquet"
CALIBRATE_DAYS = 45
TEST_DAYS = 90
# features + one unit per quantile head + calibrate + score + save
STAGES_PER_TECH = 3 + len(QUANTILES)


def _route_lightgbm_through_tqdm() -> None:
    """LightGBM prints straight to stdout and would shred the progress bar."""

    class _TqdmLogger:
        def info(self, msg):
            tqdm.write(f"      {msg}")

        def warning(self, msg):
            tqdm.write(f"      ! {msg}")

        def debug(self, msg):
            pass

        def error(self, msg):
            tqdm.write(f"      !! {msg}")

    lgb.register_logger(_TqdmLogger())


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


def run(region: str, tech: str, bar: tqdm, valid_frac: float, log_period: int) -> dict:
    cfg = load_region(region)
    cap = float(cfg.capacity_mw[tech])
    gold = pd.read_parquet(GOLD)
    gold = gold[(gold["region_id"] == region) & (gold[f"sample_weight_{tech}"] == 1)]
    actuals = load_actuals(tech)

    tr, cal, te = split(gold)
    tqdm.write(f"  train {len(tr):,}  calibrate {len(cal):,}  test {len(te):,}")
    if min(len(tr), len(cal), len(te)) < 200:
        raise SystemExit(f"not enough rows for a three-way split on {region}/{tech}")

    params, source = load_params(region, tech)
    tqdm.write(f"  hyperparameters from {source}")

    clock = time.perf_counter()
    done_stages = 0

    def stage(label: str) -> None:
        """One unit of the bar, with the wall clock for the unit just finished."""
        nonlocal clock, done_stages
        now = time.perf_counter()
        done_stages += 1
        bar.set_postfix_str(f"last {now - clock:.1f}s", refresh=False)
        if bar.disable:
            # redirected to a file: a live bar is unreadable there, so the same
            # information goes out as one plain line per stage (a disabled tqdm
            # does not advance `bar.n`, hence the local counter)
            tqdm.write(f"  [{done_stages}/{STAGES_PER_TECH}] {bar.desc} {now - clock:.1f}s")
        clock = now
        bar.set_description(f"{region}/{tech} {label}")
        bar.update(1)

    def prep(part: pd.DataFrame):
        x = build_all_features(region, part, actuals=actuals, tech=tech)
        phys = physics_forecast(cfg, part, tech).to_numpy(dtype=float)
        y_cf = part[f"y_{tech}_cf"].to_numpy(dtype=float)
        return x, phys, y_cf

    bar.set_description(f"{region}/{tech} features")
    x_tr, phys_tr, y_tr = prep(tr)
    stage("fitting p10")

    heads = iter(["fitting p50", "fitting p90", "calibrating"])
    models = train_residual(
        x_tr,
        pd.Series(y_tr),
        pd.Series(phys_tr),
        pd.Series([1.0] * len(x_tr)),
        params=params,
        valid_frac=valid_frac,
        log_period=log_period,
        on_head=lambda q: stage(next(heads)),
    )

    # calibrate on the held-out middle window, per lead-hour band
    x_cal, phys_cal, y_cal = prep(cal)
    b_cal = predict_residual(models, x_cal, phys_cal, cap)
    cmask = calibration_mask(cal, tech)
    cond_cal = conditioning_buckets(x_cal, tech)
    conformal = SplitConformal(alpha=0.2).calibrate(
        b_cal["p10_mw"].to_numpy(dtype=float)[cmask],
        b_cal["p90_mw"].to_numpy(dtype=float)[cmask],
        (y_cal * cap)[cmask],
        cal["lead_hours"].to_numpy(dtype=float)[cmask],
        None if cond_cal is None else cond_cal[cmask],
    )
    stage("scoring")

    # score on the final window, which neither fitting step has seen
    x_te, phys_te, y_te = prep(te)
    b_te = predict_residual(models, x_te, phys_te, cap)
    lo, hi = conformal.apply(
        b_te["p10_mw"].to_numpy(dtype=float),
        b_te["p90_mw"].to_numpy(dtype=float),
        te["lead_hours"].to_numpy(dtype=float),
        conditioning_buckets(x_te, tech),
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
    w = report["n_rows"].to_numpy(dtype=float)
    metrics["width_mean"] = float(
        np.average(report["mean_width_frac"].to_numpy(dtype=float), weights=w)
    )
    stage("saving")

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
        # ORDER matters: LightGBM scores by position, so the artifact records the
        # exact sequence the heads were fitted on, not merely the set.
        "features": fitted_features(models) or list(x_tr.columns),
        "params": {k: v for k, v in params.items() if k not in ("verbose", "n_jobs")},
        "params_source": source,
        "tuned": source != "defaults",
        "valid_frac": valid_frac,
        "n_estimators_used": {
            f"p{int(q * 100)}": int(m.best_iteration_ or m.n_estimators_) for q, m in models.items()
        },
        **{k: float(v) for k, v in metrics.items()},
    }

    try:
        incumbent = load_model(region, tech).get("meta", {})
    except (FileNotFoundError, KeyError):
        incumbent = {}
    meta["promoted"] = not (incumbent and not promote(meta, incumbent))
    if incumbent and not set(incumbent.get("features", [])) <= set(meta["features"]):
        tqdm.write(
            "  incumbent needs features the builder no longer produces "
            f"({sorted(set(incumbent.get('features', [])) - set(meta['features']))}) "
            "-- it cannot score live data, so it is not a valid incumbent"
        )
    if not meta["promoted"]:
        tqdm.write(
            f"  PROMOTION REFUSED -- incumbent nRMSE "
            f"{incumbent.get('nrmse_mean', float('nan')) * 100:.2f}%"
            f" / PICP {incumbent.get('picp_mean', float('nan')):.3f}. Keeping it."
        )
        return meta

    save_model(models, conformal, meta, f"artifacts/models/{region}_{tech}")
    tqdm.write(f"  promoted {version}")
    return meta


def summary_table(results: list[dict]) -> str:
    head = (
        f"{'tech':<7}{'nRMSE':>8}{'persist':>9}{'physics':>9}{'Elia':>8}"
        f"{'skill':>8}{'PICP':>7}{'width':>8}  status"
    )
    lines = [head, "-" * len(head)]
    for m in results:
        lines.append(
            f"{m['tech']:<7}{m['nrmse_mean'] * 100:7.2f}%{m['nrmse_persistence'] * 100:8.2f}%"
            f"{m['nrmse_physics'] * 100:8.2f}%{m['nrmse_tso'] * 100:7.2f}%"
            f"{m['skill_mean']:8.3f}{m['picp_mean']:7.3f}{m['width_mean'] * 100:7.2f}%"
            f"  {'promoted' if m.get('promoted') else 'REFUSED'}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--tech", action="append", choices=["solar", "wind"])
    ap.add_argument(
        "--valid-frac",
        type=float,
        default=0.1,
        help="chronological tail of the train window held out for early stopping; 0 disables",
    )
    ap.add_argument("--log-period", type=int, default=50, help="LightGBM iterations between logs")
    args = ap.parse_args()

    _route_lightgbm_through_tqdm()
    techs = args.tech or ["solar", "wind"]
    started = time.perf_counter()
    results = []
    with tqdm(
        total=len(techs) * STAGES_PER_TECH,
        unit="stage",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=None,  # bar on a terminal, plain lines when redirected to a file
    ) as bar:
        for tech in techs:
            tqdm.write(f"== {args.region}/{tech} ==")
            results.append(run(args.region, tech, bar, args.valid_frac, args.log_period))

    print(f"\n{summary_table(results)}", flush=True)
    print(f"\ntotal {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
