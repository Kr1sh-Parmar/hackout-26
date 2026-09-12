"""Rungs 6/7 benchmark: deep sequence models against the served LightGBM rung.

    python scripts/train_deep.py --region BE              # all 22 folds
    python scripts/train_deep.py --region BE --folds 2    # smoke
    python scripts/train_deep.py --region BE --graph-ablate

NOTHING HERE IS EVER SERVED. Production stays on rung 5 (`scripts/train.py` ->
`registry.py`); this script writes `artifacts/deep_benchmark.json` and nothing
else. Losing to LightGBM is a legitimate published result -- it is reported as
measured, not tuned until it wins.

All three rungs get the SAME `Fold` objects, the SAME feature build, and the
SAME scoring call, and the rung-5 headline is asserted against
`artifacts/backtest.json`. If that cross-check fails the harness has drifted and
every other number in the file is suspect.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.config import load_region  # noqa: E402
from src.evaluation.baselines import climatology, hourly_actual_cf, persistence  # noqa: E402
from src.evaluation.report import per_lead_hour_report, summarise  # noqa: E402
from src.evaluation.walk_forward import assert_no_overlap, walk_forward  # noqa: E402
from src.features.build import build_all_features  # noqa: E402
from src.models import deep_seq  # noqa: E402
from src.models.physics import physics_forecast  # noqa: E402
from src.models.residual_gbdt import design_matrix, predict_residual, train_residual  # noqa: E402
from src.uncertainty.conformal import (  # noqa: E402
    SplitConformal,
    calibration_mask,
    conditioning_buckets,
)

GOLD = "data/gold/training_base_24_72h/part-0.parquet"
WEATHER = "data/silver/weather_nwp"
OUT = pathlib.Path("artifacts/deep_benchmark.json")
INCUMBENT = pathlib.Path("artifacts/backtest.json")

SEEDS = (0, 1, 2)
FOLD_KW = dict(train_months=6, test_months=1, gap_days=1, calibrate_days=45)
ALPHA = 0.2
THREADS = 4
LEAD_MARKS = [24, 36, 48, 60, 72]


# --------------------------------------------------------------------------- #
# per-fold plumbing
# --------------------------------------------------------------------------- #
def prep(cfg, tech, part, actuals):
    """Exactly `backtest.fold_predictions.prep`, plus the tensors the deep rungs need."""
    x = build_all_features(cfg.region_id, part, actuals=actuals, tech=tech)
    phys = physics_forecast(cfg, part, tech).to_numpy(dtype=float)
    y = part[f"y_{tech}_cf"].to_numpy(dtype=float)
    return x, phys, y


def conformalise(fold, tech, cal_band, y_c, cap, te, lo, hi, x_c=None, x_te=None):
    """Byte-identical to `scripts/backtest.py`'s calibration block, called three
    times. `x_c`/`x_te` carry the conditioning buckets; without them the rung-5
    cross-check would calibrate differently from the harness it must reproduce."""
    if len(fold.calibrate) <= 200:
        return lo, hi
    cmask = calibration_mask(fold.calibrate, tech)
    cond_c = conditioning_buckets(x_c, tech) if x_c is not None else None
    conformal = SplitConformal(alpha=ALPHA).calibrate(
        cal_band["p10_mw"].to_numpy(dtype=float)[cmask],
        cal_band["p90_mw"].to_numpy(dtype=float)[cmask],
        (y_c * cap)[cmask],
        fold.calibrate["lead_hours"].to_numpy(dtype=float)[cmask],
        None if cond_c is None else cond_c[cmask],
    )
    cond_t = conditioning_buckets(x_te, tech) if x_te is not None else None
    return conformal.apply(lo, hi, te["lead_hours"].to_numpy(dtype=float), cond_t)


def preds_frame(te, tech, y_te, cap, p50, lo, hi, phys_te, actual_cf):
    return pd.DataFrame(
        {
            "lead_hours": te["lead_hours"].to_numpy(),
            "y_true": y_te * cap,
            "p50": p50,
            "p10": lo,
            "p90": hi,
            "physics": phys_te * cap,
            "persistence": persistence(te, actual_cf).to_numpy(dtype=float) * cap,
            "climatology": climatology(te, actual_cf).to_numpy(dtype=float) * cap,
            "tso": te[f"tso_{tech}_da_p50_mw"].to_numpy(dtype=float),
            "tso_wa": te[f"tso_{tech}_wa_p50_mw"].to_numpy(dtype=float),
            "is_day": te["is_day"].to_numpy() if tech == "solar" else 1,
        }
    )


def fold_rungs(cfg, tech, fold, actuals, cap, table, meta, ablate):
    """One fold, three rungs, same inputs. Returns `{rung: preds}` and fit seconds."""
    parts = {k: getattr(fold, k) for k in ("train", "calibrate", "test")}
    built = {k: prep(cfg, tech, p, actuals) for k, p in parts.items()}
    x_tr, phys_tr, y_tr = built["train"]
    te = parts["test"]
    _, phys_te, y_te = built["test"]
    _, _, y_c = built["calibrate"]
    actual_cf = hourly_actual_cf(actuals)

    # ---- rung 5: refit in-process, bare defaults, matching backtest.py:44 ----
    t0 = time.perf_counter()
    gbdt = train_residual(x_tr, pd.Series(y_tr), pd.Series(phys_tr), pd.Series([1.0] * len(x_tr)))
    secs = {"rung5": time.perf_counter() - t0, "rung6": 0.0, "rung7": 0.0, "rung7_ablate": 0.0}

    b_c = predict_residual(gbdt, built["calibrate"][0], built["calibrate"][1], cap)
    b_te = predict_residual(gbdt, built["test"][0], phys_te, cap)
    lo, hi = conformalise(
        fold,
        tech,
        b_c,
        y_c,
        cap,
        te,
        b_te["p10_mw"].to_numpy(float),
        b_te["p90_mw"].to_numpy(float),
        built["calibrate"][0],
        built["test"][0],
    )
    out = {
        "rung5": preds_frame(
            te, tech, y_te, cap, b_te["p50_mw"].to_numpy(float), lo, hi, phys_te, actual_cf
        )
    }
    seeded: dict[str, list[pd.DataFrame]] = {}
    meta_out: dict[str, int] = {}

    # ---- shared deep tensors: standardiser fitted on TRAIN ROWS ONLY ----
    cols = list(design_matrix(x_tr).columns)
    std = deep_seq.Standardiser().fit(design_matrix(x_tr).to_numpy(dtype=float), cols)
    seq = {}
    for k, p in parts.items():
        flat = std.transform(design_matrix(built[k][0]).to_numpy(dtype=float))
        seq[k] = deep_seq.to_sequences(p, flat, p[f"sample_weight_{tech}"].to_numpy(dtype=float))
    run_ts = deep_seq.run_timestamps(parts["train"])
    resid_tr = np.asarray(y_tr) - np.asarray(phys_tr)
    y_seq, _, _ = deep_seq.to_sequences(parts["train"], resid_tr)
    y_seq = y_seq[:, :, 0]

    node_std = None
    nseq = {}
    if table is not None:
        flat_tr = deep_seq.node_matrix(parts["train"], table, meta)
        ncols = [f"n{i}" for i in range(flat_tr.shape[-1])]
        node_std = deep_seq.Standardiser().fit(flat_tr, ncols)
        for k, p in parts.items():
            nf = node_std.transform(deep_seq.node_matrix(p, table, meta))
            nseq[k] = deep_seq.to_sequences(p, nf.reshape(len(p), -1))[0].reshape(
                seq[k][0].shape[0], len(deep_seq.LEADS), flat_tr.shape[1], -1
            )

    rungs = [("rung6", False, False)]
    if table is not None:
        rungs.append(("rung7", True, False))
        if ablate:
            rungs.append(("rung7_ablate", True, True))

    for name, use_graph, abl in rungs:
        models = []
        t0 = time.perf_counter()
        for s in SEEDS:
            models.append(
                deep_seq.fit(
                    seq["train"][0],
                    y_seq,
                    seq["train"][1],
                    nseq["train"] if use_graph else None,
                    meta=meta if use_graph else None,
                    ablate=abl,
                    seed=s,
                    run_ts=run_ts,
                    threads=THREADS,
                )
            )
        secs[name] = time.perf_counter() - t0
        meta_out[name] = deep_seq.n_params(models[0])

        def band(key, models=models, use_graph=use_graph):
            return deep_seq.predict_deep(
                models,
                seq[key][0],
                nseq[key] if use_graph else None,
                seq[key][2],
                built[key][1],
                cap,
            )

        d_c, d_te = band("calibrate"), band("test")
        lo, hi = conformalise(
            fold,
            tech,
            d_c,
            y_c,
            cap,
            te,
            d_te["p10_mw"].to_numpy(float),
            d_te["p90_mw"].to_numpy(float),
            built["calibrate"][0],
            built["test"][0],
        )
        out[name] = preds_frame(
            te, tech, y_te, cap, d_te["p50_mw"].to_numpy(float), lo, hi, phys_te, actual_cf
        )
        # Per-seed, unconformalised: the spread of the headline across seeds is
        # the significance bar every rung-6/7 gap has to clear.
        seeded[name] = [
            pd.DataFrame(
                {
                    "lead_hours": te["lead_hours"].to_numpy(),
                    "y_true": y_te * cap,
                    "p50": deep_seq.predict_deep(
                        [m],
                        seq["test"][0],
                        nseq["test"] if use_graph else None,
                        seq["test"][2],
                        phys_te,
                        cap,
                    )["p50_mw"].to_numpy(float),
                    "is_day": te["is_day"].to_numpy() if tech == "solar" else 1,
                }
            )
            for m in models
        ]

    # One line, kills the whole misalignment class.
    for name, p in out.items():
        assert len(p) == len(out["rung5"]), f"{name} row count differs from rung5"
        assert (p["y_true"].values == out["rung5"]["y_true"].values).all(), f"{name} y_true moved"
        assert (p["lead_hours"].values == out["rung5"]["lead_hours"].values).all(), (
            f"{name} lead_hours moved"
        )

    return out, seeded, secs, meta_out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def wmean(report: pd.DataFrame, col: str) -> float:
    w = report["n_rows"].to_numpy(dtype=float)
    v = report[col].to_numpy(dtype=float)
    m = np.isfinite(v)
    return float(np.average(v[m], weights=w[m])) if m.any() else float("nan")


def score(preds: pd.DataFrame, cap: float, tech: str) -> tuple[pd.DataFrame, dict]:
    report = per_lead_hour_report(preds, capacity=cap, daylight_only=(tech == "solar"))
    m = summarise(report)
    m["width_mean"] = wmean(report, "mean_width_frac")
    m["pinball_mean"] = wmean(report, "pinball_mean")
    m["nrmse_by_lead"] = {
        int(r.lead_hours): float(r.nrmse_model)
        for _, r in report[report.lead_hours.isin(LEAD_MARKS)].iterrows()
    }
    return report, m


def verdict(r6: dict, r7: dict, r5: dict, spread: float) -> str:
    best = min(r6["nrmse_mean"], r7["nrmse_mean"]) if r7 else r6["nrmse_mean"]
    gap = best - r5["nrmse_mean"]
    if abs(gap) < spread:
        return (
            f"no measurable difference: best deep rung is {gap * 100:+.3f}pp vs rung 5, "
            f"inside the {spread * 100:.3f}pp seed spread"
        )
    return (
        f"rung 5 (LightGBM) wins by {gap * 100:.3f}pp nRMSE"
        if gap > 0
        else f"deep rung wins by {-gap * 100:.3f}pp nRMSE"
    )


# --------------------------------------------------------------------------- #
def run(region: str, tech: str, n_folds: int | None, ablate: bool) -> dict:
    cfg = load_region(region)
    cap = float(cfg.capacity_mw[tech])
    gold = pd.read_parquet(GOLD)
    gold = gold[(gold["region_id"] == region) & (gold[f"sample_weight_{tech}"] == 1)]
    actuals = pd.read_parquet(f"data/silver/generation_actuals_{tech}/part-0.parquet")
    actuals = actuals[actuals["qc_flag"] == "OK"]

    wx = pd.read_parquet(WEATHER)
    wx = wx[(wx["region_id"] == region) & wx["lead_hours"].between(24, 72)]
    meta = deep_seq.node_meta(wx)
    table = deep_seq.node_table(wx)
    del wx

    folds = walk_forward(gold, **FOLD_KW)
    assert_no_overlap(folds)
    if not folds:
        raise SystemExit("no folds -- not enough history")
    if n_folds:
        folds = folds[:n_folds]
    print(f"== {region}/{tech} == {len(folds)} fold(s), nodes {list(meta.index)}")

    acc: dict[str, list[pd.DataFrame]] = {}
    seed_acc: dict[str, list[list[pd.DataFrame]]] = {}
    secs: dict[str, float] = {}
    params: dict[str, int] = {}
    for i, f in enumerate(folds):
        out, seeded, s, p = fold_rungs(cfg, tech, f, actuals, cap, table, meta, ablate)
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
        for k, v in seeded.items():
            seed_acc.setdefault(k, []).append(v)
        for k, v in s.items():
            secs[k] = secs.get(k, 0.0) + v
        params.update(p)
        print(
            f"  fold {i + 1}/{len(folds)} done  ({', '.join(f'{k} {v:.0f}s' for k, v in s.items())})"
        )

    results, spreads = {}, {}
    for rung, frames in acc.items():
        preds = pd.concat(frames, ignore_index=True)
        _, m = score(preds, cap, tech)
        m["n_params"] = params.get(rung, 0)
        m["folds"] = len(folds)
        m["fit_seconds"] = round(secs.get(rung, 0.0), 1)
        if rung in seed_acc:
            per_seed = [
                score(pd.concat([f[j] for f in seed_acc[rung]], ignore_index=True), cap, tech)[1][
                    "nrmse_mean"
                ]
                for j in range(len(SEEDS))
            ]
            m["seed_spread_nrmse"] = float(np.std(per_seed))
            m["seed_nrmse"] = [float(v) for v in per_seed]
            spreads[rung] = m["seed_spread_nrmse"]
        results[rung] = m
        print(
            f"  {rung:13s} nRMSE {m['nrmse_mean'] * 100:6.3f}%  skill {m['skill_mean']:6.3f}  "
            f"PICP {m['picp_mean']:.3f}  width {m['width_mean'] * 100:5.2f}%  "
            f"params {m['n_params']:,}"
        )

    spread = max(spreads.values()) if spreads else 0.0
    return {
        "region": region,
        "tech": tech,
        "nodes": list(meta.index),
        "results": results,
        "verdict": verdict(results["rung6"], results.get("rung7"), results["rung5"], spread),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="BE")
    ap.add_argument("--tech", action="append", choices=["solar", "wind"])
    ap.add_argument("--folds", type=int, default=None, help="first N folds only, for iteration")
    ap.add_argument(
        "--graph-ablate",
        action="store_true",
        help="also run rung 7 with identity adjacency and pooling frozen at capacity weights",
    )
    args = ap.parse_args()

    incumbent = {
        f"{r['region']}_{r['tech']}": r["nrmse_mean"] for r in json.loads(INCUMBENT.read_text())
    }
    blocks, match = [], {}
    for tech in args.tech or ["solar", "wind"]:
        blob = run(args.region, tech, args.folds, args.graph_ablate)
        key = f"{args.region}_{tech}"
        delta = blob["results"]["rung5"]["nrmse_mean"] - incumbent.get(key, float("nan"))
        match[key] = {
            "artifact_nrmse": incumbent.get(key),
            "in_process_nrmse": blob["results"]["rung5"]["nrmse_mean"],
            "delta": delta,
            "ok": bool(abs(delta) < 1e-3),
        }
        # The gate on everything: if rung 5 does not reproduce, the harness has
        # drifted and no other number in this file means anything.
        if args.folds is None:
            assert abs(delta) < 1e-3, (
                f"in-process rung 5 for {key} is {delta:+.5f} off artifacts/backtest.json -- "
                "the harness has drifted; fix that before reporting anything"
            )
        blocks.append(blob)

    ablations = {
        b["tech"]: {
            "rung7_ablate_nrmse": b["results"]["rung7_ablate"]["nrmse_mean"],
            "rung7_nrmse": b["results"]["rung7"]["nrmse_mean"],
        }
        for b in blocks
        if "rung7_ablate" in b["results"]
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_utc": pd.Timestamp.utcnow().isoformat(),
                "served": "rung5",
                "protocol": {
                    "folds": FOLD_KW,
                    "folds_run": args.folds or "all",
                    "alpha": ALPHA,
                    "conformal": {"band_hours": 12, "scaled": True},
                    "seeds": list(SEEDS),
                    "torch": torch.__version__,
                    "threads": THREADS,
                    "leads": [deep_seq.LEADS[0], deep_seq.LEADS[-1]],
                    "nodes": blocks[0]["nodes"],
                    "architecture": "N-HiTS-style multi-rate stack (forecast-only, no backcast)",
                },
                "incumbent_match": match,
                "results": {b["tech"]: b["results"] for b in blocks},
                "ablations": ablations,
                "verdict": {b["tech"]: b["verdict"] for b in blocks},
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {OUT}")
    for b in blocks:
        print(f"  {b['tech']}: {b['verdict']}")


if __name__ == "__main__":
    main()
