"""Hyperparameter search for the residual GBDT, against a WALK-FORWARD objective.

    python scripts/tune.py --region BE --tech solar --trials 30
    python scripts/tune.py --region BE --timeout 900

OBJECTIVE: mean pinball loss over the three quantile heads, on held-out
walk-forward folds, scored on the same rows the model is reported on (solar
daylight-only). Not nRMSE on p50: nRMSE scores one of the three heads and is
blind to the band, so tuning on it quietly buys a sharper median with a rotten
p10/p90 -- and the band is the half of the output an operator sizes reserve
from. Pinball prices all three at once and is the loss the model is fitted on.

A random split would be meaningless here: adjacent hours are near-identical, so
it puts almost the same row on both sides of the fence. Every fold is the same
expanding-window split `scripts/backtest.py` reports through.

Trial 0 is always the current defaults, so "tuned vs default" is a comparison on
identical folds rather than against a remembered number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.evaluation.report import per_lead_hour_report  # noqa: E402
from src.evaluation.walk_forward import assert_no_overlap, walk_forward  # noqa: E402
from src.features.build import build_all_features  # noqa: E402
from src.models.physics import physics_forecast  # noqa: E402
from src.models.residual_gbdt import PARAMS, TUNING_DIR, predict_residual, train_residual  # noqa: E402

GOLD = "data/gold/training_base_24_72h/part-0.parquet"

# The knobs that actually move a GBDT. `objective`, `metric`, `n_jobs` are not
# hyperparameters, they are the problem statement.
TUNED_KEYS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
)


def suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1600, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def prepared_folds(region: str, tech: str, n_folds: int) -> tuple[list[dict], float]:
    """Features, physics and target for the last `n_folds` folds, built ONCE.

    Feature construction is ~1.2 s per fold and is identical across trials;
    rebuilding it inside the objective would spend most of the search budget
    recomputing the same numbers.
    """
    cfg = load_region(region)
    cap = float(cfg.capacity_mw[tech])
    gold = pd.read_parquet(GOLD)
    gold = gold[(gold["region_id"] == region) & (gold[f"sample_weight_{tech}"] == 1)]
    actuals = pd.read_parquet(f"data/silver/generation_actuals_{tech}/part-0.parquet")
    actuals = actuals[actuals["qc_flag"] == "OK"]

    folds = walk_forward(gold, train_months=6, test_months=1, gap_days=1, calibrate_days=45)
    assert_no_overlap(folds)
    folds = folds[-n_folds:]

    def prep(part):
        x = build_all_features(region, part, actuals=actuals, tech=tech)
        phys = physics_forecast(cfg, part, tech).to_numpy(dtype=float)
        return x, phys, part[f"y_{tech}_cf"].to_numpy(dtype=float)

    out = []
    for f in tqdm(
        folds,
        desc=f"{region}/{tech} building folds",
        unit="fold",
        file=sys.stdout,
        disable=None,
    ):
        x_tr, phys_tr, y_tr = prep(f.train)
        x_te, phys_te, y_te = prep(f.test)
        out.append(
            {
                "x_tr": x_tr,
                "phys_tr": phys_tr,
                "y_tr": y_tr,
                "x_te": x_te,
                "phys_te": phys_te,
                "y_te": y_te,
                "lead": f.test["lead_hours"].to_numpy(),
                "is_day": f.test["is_day"].to_numpy() if tech == "solar" else 1,
            }
        )
    return out, cap


def score(folds: list[dict], cap: float, params: dict, tech: str) -> tuple[float, float]:
    """(mean pinball, nRMSE) over the held-out folds, in MW-normalised units."""
    frames = []
    for f in folds:
        models = train_residual(
            f["x_tr"],
            pd.Series(f["y_tr"]),
            pd.Series(f["phys_tr"]),
            pd.Series(np.ones(len(f["x_tr"]))),
            params={**PARAMS, **params},
        )
        b = predict_residual(models, f["x_te"], f["phys_te"], cap)
        frames.append(
            pd.DataFrame(
                {
                    "lead_hours": f["lead"],
                    "y_true": f["y_te"] * cap,
                    "p10": b["p10_mw"].to_numpy(dtype=float),
                    "p50": b["p50_mw"].to_numpy(dtype=float),
                    "p90": b["p90_mw"].to_numpy(dtype=float),
                    "is_day": f["is_day"],
                }
            )
        )
    report = per_lead_hour_report(
        pd.concat(frames, ignore_index=True), capacity=cap, daylight_only=tech == "solar"
    )
    w = report["n_rows"].to_numpy(dtype=float)
    wmean = lambda c: float(np.average(report[c].to_numpy(dtype=float), weights=w))  # noqa: E731
    return wmean("pinball_mean"), wmean("nrmse_model")


def tune(region: str, tech: str, trials: int, timeout: float | None, n_folds: int) -> dict:
    folds, cap = prepared_folds(region, tech, n_folds)
    history: list[tuple[float, float]] = []

    def objective(trial: optuna.Trial) -> float:
        pinball, nrmse = score(folds, cap, suggest(trial), tech)
        trial.set_user_attr("nrmse", nrmse)
        history.append((pinball, nrmse))
        return pinball

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    # trial 0 = today's defaults, so the search has a real incumbent to beat
    study.enqueue_trial({k: PARAMS[k] for k in TUNED_KEYS})

    bar = tqdm(
        total=trials,
        desc=f"{region}/{tech} trials",
        unit="trial",
        file=sys.stdout,
        disable=None,  # bar on a terminal, plain lines when redirected to a file
    )

    def tick(study_, trial_):
        best = study_.best_trial
        note = (
            f"best pinball {best.value:.4f} "
            f"nRMSE {best.user_attrs.get('nrmse', float('nan')) * 100:.2f}%"
        )
        bar.set_postfix_str(note)
        if bar.disable:
            tqdm.write(f"  trial {trial_.number:3d}/{trials} pinball {trial_.value:.4f} | {note}")
        bar.update(1)

    started = time.perf_counter()
    study.optimize(objective, n_trials=trials, timeout=timeout, callbacks=[tick])
    bar.close()

    base_pinball, base_nrmse = history[0]
    best = study.best_trial
    won = best.number != 0 and best.value < base_pinball
    result = {
        "region": region,
        "tech": tech,
        "folds": n_folds,
        "trials": len(study.trials),
        "objective": "mean pinball over p10/p50/p90, walk-forward, "
        + ("daylight-only" if tech == "solar" else "all hours"),
        "default_pinball": base_pinball,
        "default_nrmse": base_nrmse,
        "best_pinball": best.value,
        "best_nrmse": best.user_attrs.get("nrmse"),
        "improvement_pct": 100.0 * (base_pinball - best.value) / base_pinball,
        "beat_defaults": bool(won),
        # Only write params the search actually won with. A "tuned" file that is
        # really the defaults dressed up would make every later artifact lie
        # about where its hyperparameters came from.
        "params": {k: best.params[k] for k in TUNED_KEYS} if won else {},
        "elapsed_s": time.perf_counter() - started,
    }

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    path = TUNING_DIR / f"{region}_{tech}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n== {region}/{tech} ==")
    print(f"  defaults  pinball {base_pinball:.5f}  nRMSE {base_nrmse * 100:.2f}%")
    print(
        f"  best      pinball {best.value:.5f}  nRMSE "
        f"{(best.user_attrs.get('nrmse') or float('nan')) * 100:.2f}%  (trial {best.number})"
    )
    if won:
        print(f"  TUNED WINS by {result['improvement_pct']:.2f}% pinball -> {path}")
        print(f"  {json.dumps(result['params'])}")
    else:
        print("  tuning did NOT beat the defaults on held-out folds. Keeping the defaults.")
    print(f"  {result['elapsed_s']:.0f}s over {n_folds} folds, {len(study.trials)} trials")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--tech", action="append", choices=["solar", "wind"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=None, help="seconds, per tech")
    ap.add_argument("--folds", type=int, default=4, help="how many trailing folds to score on")
    args = ap.parse_args()

    for tech in args.tech or ["solar", "wind"]:
        tune(args.region, tech, args.trials, args.timeout, args.folds)


if __name__ == "__main__":
    main()
