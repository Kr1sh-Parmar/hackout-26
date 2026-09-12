# ZERO BIAS — AI-Powered Renewable Generation Forecasting Platform

A regional, probabilistic 24–72 hour forecasting and grid-decision platform for solar and wind generation, built on open data with a physics-first modelling approach and a full operational serving path.

**Validation region:** Belgium (Elia) — the only open portal publishing generation, the grid operator's own forecast, and demand on one 15-minute timebase.

---

## 1. What This Platform Does

Given weather forecasts and site parameters, the system produces:

1. **Calibrated probabilistic forecasts** (P10 / P50 / P90) of solar and wind output, 24–72 hours ahead
2. **Grid event detection** — over-generation windows, steep ramps, deficit risk, low-confidence periods, storm shutdown risk
3. **Sized, priced grid-action recommendations** — curtail, charge storage, discharge storage, or commit backup, each with a confidence rating
4. **Storage sizing analysis** — curtailment avoided vs. battery capacity, for infrastructure planning
5. **A live, operational forecast cycle** — fetches real-time weather and demand, runs the full pipeline, and serves results over a REST API

The output is designed for grid operators, utilities, and energy traders — not a chart, but a decision with a number attached.

---

## 2. Data Pipeline

### 2.1 Sources

| Source | What it provides | Resolution | Volume |
|---|---|---|---|
| **Elia Open Data** (`ods032`, `ods031`, `ods001`) | Solar + wind generation (measured), demand, and Elia's own day-ahead/week-ahead forecasts with P10/P90 | 15-minute | 344 MB |
| **Open-Meteo Historical Forecast API** | Archived weather forecasts (not reanalysis) — ECMWF, ICON, GFS models | Hourly | 15 files |
| **Open-Meteo Previous Runs API** | Lead-time-stratified weather (24–95h lead bands), the training signal for the 24–72h horizon | Hourly | 15 files |
| **Open-Meteo live Forecast API** | Real-time weather for live inference | Hourly | On-demand |
| **Elia live load forecast** | Forward demand (day-ahead / week-ahead vintages with P10/P90) | 15-minute | On-demand |

**Why forecast archives, not reanalysis:** training on reanalysis (the retrospective "best guess" weather) and serving on live forecasts is a textbook train/serve mismatch. The pipeline trains exclusively on archived *forecasts* — the same imperfect information a live system will actually have.

### 2.2 Medallion Architecture

```
raw/  → bronze/  → silver/  → gold/
(as       (typed,    (validated,  (regional matrix,
downloaded) UTC)      QC-flagged) model-ready)
```

- **raw/** — untouched source files, fully reproducible via `scripts/dl_*.py`
- **bronze/** — parsed, typed, timezone-normalized (UTC everywhere)
- **silver/** — quality-flagged (`OK` / `MISSING` / `FROZEN` / `OUT_OF_RANGE` / `CURTAILED`), deduplicated
- **gold/** — capacity-weighted regional aggregation across the available grid points × 3 NWP models, joined with generation truth, demand, and TSO benchmark forecasts. Five points are fetched, but only four survive into the 24–72 h band (`namur` is published at lead 0–23 h only, and `liege` carries two of the three NWP models there); weights are renormalised over the points actually present

### 2.3 Training Dataset

**`gold/training_base_24_72h`** — 45,080 rows × 140 columns, spanning **2024-03-05 to 2026-09-10**.

| Property | Detail |
|---|---|
| Grid points | 5 fetched, 4 present across the whole 24–72 h band; capacity-weighted, renormalised over those present |
| NWP models | 3 (ECMWF IFS, ICON, GFS) — blended, with model disagreement as a feature |
| Lead-time bands | 24–47h, 48–71h, 72h — genuinely time-stratified, not day-of forecasts |
| Target | Capacity factor residual (`y_cf − physics_cf`), never raw MW |
| Quality control | Curtailed / missing / frozen intervals carry zero training weight |
| Train / calibrate / test split | Chronological three-way split — no random shuffling, no leakage |

Full provenance, licensing, and reproduction steps: [`SOURCES.md`](SOURCES.md) and [`data/README.md`](data/README.md).

---

## 3. Technical Specification

### 3.1 Modelling Approach — Physics-First, Residual-Learned

Rather than asking a neural network to learn orbital mechanics and turbine aerodynamics from scratch, the pipeline separates the deterministic physics from the learned correction:

```
Weather → [Physics Model] → Physics-only forecast (pvlib solar geometry,
                              IEC 61400 wind power curves)
              ↓
         [LightGBM]      → learns the RESIDUAL: what physics missed
              ↓
    [Quantile Regression] → P10 / P50 / P90 bands
              ↓
  [Conformal Calibration] → statistically guaranteed coverage
```

### 3.2 Model Ladder (per the technical design)

| Rung | Method | Status |
|---|---|---|
| 0 | Persistence & climatology baselines | ✅ Built |
| 1 | Pure physics (pvlib solar geometry, IEC wind power curves) | ✅ Built |
| 2 | Gradient boosting (LightGBM) | ✅ Built |
| 3 | Residual learning (`y_cf − physics_cf`) | ✅ Built |
| 4 | Quantile regression (α = 0.1 / 0.5 / 0.9) | ✅ Built |
| 5 | Split-conformal calibration | ✅ Built |
| 6 | Deep sequence models (N-HiTS / TFT) | In training |
| 7 | Spatio-temporal GNN | In training |

The core modelling ladder (rungs 0–5) is fully implemented and serving live traffic.

### 3.2.1 Rungs 6–7: the deep benchmark, and why it is not served

The ladder's rule is that each rung must beat the rung below on a held-out window
*or be skipped with a stated reason*. Rungs 6 and 7 were built and measured rather than
assumed, on the **same 22 folds, same features, same conformal calibration and same scoring
call** as the incumbent — the harness asserts that its in-process rung-5 refit reproduces
`artifacts/backtest.json` exactly before any other number is written.

| Rung | Solar nRMSE | Wind nRMSE | PICP (solar / wind) | Params |
|---|---|---|---|---|
| 5 — LightGBM residual (**served**) | **5.363 %** | **8.568 %** | 0.789 / 0.792 | — |
| 6 — sequence stack | 5.555 % | 9.517 % | 0.797 / 0.798 | 382 k |
| 7 — + spatial graph | 5.690 % | 9.272 % | 0.792 / 0.798 | 521 k |

LightGBM wins by 0.192 pp on solar and 0.704 pp on wind — both far outside the ~0.05 pp
seed spread across three seeds, so these are real losses, not noise. The deep rungs are
well *calibrated* (PICP 0.79–0.80 for about 1 pp more width); they lose on **sharpness**,
which is what ~450 k parameters over ~900 training runs should be expected to do.

The **ablation is the informative part**: rung 7 with identity adjacency and its pooling
frozen at the fixed capacity weights scores 5.675 % / 9.250 % — equal to or very slightly
*better* than the real graph. The learned spatial aggregation buys nothing over the
capacity weights it was initialised from. Rung 7's edge over rung 6 on wind comes from the
extra node-level features, not from spatial structure. Both numbers are published.

Nothing here is served: `registry.py`, `scripts/train.py` and `scripts/backtest.py` are
untouched by this track, torch is an optional `[deep]` extra so the serving path never
imports it, and no tuning was attempted before reporting. Full detail in
`artifacts/deep_benchmark.json`; reproduce with `python scripts/train_deep.py --region BE
--graph-ablate`.

### 3.3 Feature Engineering

44 features per forecast row, spanning five families:

| Family | Examples | Count |
|---|---|---|
| Solar physics | Clear-sky index, POA irradiance, cell temperature, clipping | 12 |
| Wind physics | Hub-height wind shear, air density correction, power curve position | 13 |
| NWP quality | Multi-model disagreement, forecast ramp rate | 4 |
| Temporal | Cyclic hour/day-of-year encoding | 6 |
| Autoregressive lags | 24h / 168h lags, smart persistence (24h minimum — no leakage) | 6 |

**Feature-order enforcement**: the serving path validates that live features match training features exactly (name and order), raising loudly rather than silently mis-scoring if the two ever drift.

### 3.4 Uncertainty Quantification

Split-conformal calibration, tuned via measurement:

- **Banded by 12-hour lead groups** (not per-hour) — pools calibration data for statistically stable width estimates
- **Scaled nonconformity scores** — the correction adapts to each hour's own predicted uncertainty rather than adding a flat margin
- **Calibrated on the same distribution the model is scored on** (daylight-only for solar)

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                           │
│  Open-Meteo (weather) · Elia (generation, demand, benchmark)      │
└───────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              MEDALLION PIPELINE (raw → bronze → silver → gold)   │
│         Shared regional aggregation (src/ingest/regional.py)     │
│         used identically by historical ETL and live serving      │
└───────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                         │
│    Physics (pvlib, IEC wind) · NWP quality · Temporal · Lags      │
└───────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                                │
│   LightGBM residual quantile models · Conformal calibration      │
│   Hyperparameter tuning (Optuna) · Promotion gate                │
│   SHAP explainability · Model registry                           │
└───────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION ENGINE                              │
│  Net-load calculation · Event detection · Storage simulation     │
│  LP dispatch optimiser · Grid action recommendations              │
└───────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                │
│         FastAPI REST endpoints over DuckDB / Parquet               │
│              Replay mode for offline demonstration                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Module Inventory

| Module | Path | Responsibility |
|---|---|---|
| **Configuration** | `src/core/config.py` | Region definitions, archetype fleet composition, storage/pricing parameters |
| **Ingestion** | `src/ingest/` | Historical + live weather adapters, Elia live demand adapter, regional aggregation, replay snapshots |
| **Feature engineering** | `src/features/` | Solar physics, wind physics, temporal encoding, NWP quality signals, autoregressive lags |
| **Quality** | `src/quality/` | Pandera schema validation, PV curtailment detection |
| **Models** | `src/models/` | Physics baseline, LightGBM residual GBDT, prediction orchestration, model registry, SHAP explainability |
| **Uncertainty** | `src/uncertainty/` | Split-conformal calibration, coverage reporting |
| **Evaluation** | `src/evaluation/` | Metrics (nRMSE, skill score, PICP, pinball loss), walk-forward backtesting, baselines, drift monitoring |
| **Decisions** | `src/decisions/` | Net-load calculation, event scanning, storage simulation, LP dispatch optimiser, action recommendation |
| **API** | `src/api/` | FastAPI application, 9 REST endpoints, response schemas |

### 4.2 CLI Scripts

| Script | Purpose |
|---|---|
| `scripts/train.py` | Trains residual models with visible progress (tqdm + per-iteration validation loss), three-way chronological split, automatic promotion gating |
| `scripts/tune.py` | Optuna hyperparameter search against walk-forward pinball loss |
| `scripts/backtest.py` | Full 22-fold walk-forward evaluation against persistence, physics, and Elia benchmarks |
| `scripts/train_deep.py` | Rung 6/7 benchmark — deep sequence + graph models against the served LightGBM rung, never promoted |
| `scripts/run_cycle.py` | End-to-end forecast cycle — supports `--live` for real-time operation |
| `scripts/fetch_forecast.py` | Fetches and caches live weather |
| `scripts/snapshot_replay.py` | Freezes a forecast cycle for offline demonstration |

---

## 5. API Surface

FastAPI application exposing 12 operations, auto-documented via OpenAPI (`/docs`):

| Endpoint | Returns |
|---|---|
| `GET /health` | System status, model freshness, warnings |
| `GET /sites` | Configured regions, fleet parameters, and whether a region is physics-only |
| `GET /runs` | The forecast runs that can be served, newest first — each a valid `run_ts` |
| `GET /forecast` | P10/P50/P90 generation forecast, per hour, per technology |
| `GET /outlook` | Net-load balance with propagated uncertainty bands |
| `GET /events` | Detected grid events with severity and lead time |
| `GET /actions` | Ranked, sized, priced grid-action recommendations, each with an `action_id` and its acknowledgement state |
| `POST` / `DELETE /actions/{action_id}/ack` | Record or withdraw an operator acknowledgement (persisted under `data/ops/`, scoped to the run) |
| `GET /storage/sweep` | Curtailment-avoided vs. battery-capacity sizing curve |
| `GET /backtest` | Accuracy per lead hour, plus the headline `summary` — computed by the same `summarise()` that writes `artifacts/backtest.json` |
| `GET /explain` | Ranked SHAP drivers of the forecast correction, per hour |

Every response carries **provenance** (`model_version`, `calibration_date`, `replay_mode`) so any served number is traceable back to the model and data that produced it. Operational endpoints accept `run_ts` to pin a historical run, in live and replay mode alike; an unknown run is a 404 that names the runs that exist.

---

## 6. Model Performance

Measured on a **22-fold walk-forward backtest** — chronological, no data leakage, retrained per fold:

| Technology | nRMSE | Skill Score* | Coverage (PICP)** |
|---|---|---|---|
| Solar (daylight hours) | 5.36% of capacity | 0.50 | 0.79 |
| Wind | 8.57% of capacity | 0.69 | 0.79 |

\* *Skill Score* = improvement over a naive persistence baseline (0 = no better than naive, 1 = perfect). Both technologies clear 0.40, the threshold the design set for "meaningfully better than naive."

\*\* *PICP* (Prediction Interval Coverage Probability) — the fraction of actual outcomes that fall inside the model's stated 80% confidence interval. Target range 0.78–0.82; both technologies land inside it, meaning the uncertainty bands are honest, not decorative.

Every published number is reproducible via `python scripts/backtest.py --region BE`, and is also benchmarked per-lead-hour against Elia's own published day-ahead and week-ahead operational forecasts as an external, professional reference point.

---

## 7. Expected Output

A single live forecast cycle (`python scripts/run_cycle.py --region BE --live`) produces, end to end:

```
== LIVE cycle BE @ 2026-09-12T09:00:00+00:00 ==
  weather   72 rows  (5 grid points × 3 NWP models, live-fetched)
  demand    Elia forward forecast (day-ahead vintage)
  forecast  144 rows (72h × 2 technologies, P10/P50/P90)
  events    STEEP_RAMP ×3, OVER_GENERATION ×1, DEFICIT_RISK ×1
  actions   DISCHARGE_BESS ×3, COMMIT_BACKUP ×1, CHARGE_BESS ×1
```

A resulting recommendation, as served by `/actions`:

```json
{
  "action": "DISCHARGE_BESS",
  "valid_from": "2026-09-12T15:00:00Z",
  "mwh": 160,
  "value_inr": 1040000,
  "decisive": true,
  "rationale": "Net load is moving 1450 MW/h -- discharge storage to
                flatten the ramp rather than lean on backup fuel."
}
```

Every action carries a size, a value, and a stated confidence — the platform's core design principle: an operator receives a decision, not a curve to interpret.

---

## 8. Reproducibility & Testing

- **210 automated tests** covering physics correctness, feature-parity between training and serving, leakage guards, API contracts, run selection, operator acknowledgements, and calibration behaviour
- **Zero data leakage by construction**: no random train/test splits anywhere in the codebase; every split is chronological with an explicit gap
- **Deterministic aggregation**: the same regional weather-aggregation function is called by both the historical training pipeline and the live serving path, verified identical to floating-point precision
- **Replay mode**: a full forecast cycle can be snapshotted and replayed with zero network access, for reliable offline demonstration — every operational endpoint reads through the same switch, verified against an empty data root so the frozen copy is genuinely what gets served

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 210 tests
python scripts/backtest.py --region BE
python scripts/run_cycle.py --region BE --live
uvicorn src.api.main:app --reload   # API at localhost:8000/docs
```

**Operations console** (`frontend/`, React + Vite). It calls the API through a same-origin `/api` proxy, so it never needs CORS or a hardcoded host:

```bash
npm --prefix frontend ci
REPLAY_MODE=true uvicorn src.api.main:app --port 8000   # or: make demo  (offline, frozen runs)
npm --prefix frontend run dev                           # or: make ui    -> http://localhost:5173
npm --prefix frontend run gen:api                       # after changing a response model; CI fails on drift
docker compose up --build                               # api on :8000, console on http://localhost:8080
```

Every panel reads the live API — there is no mock data in the frontend. With the API down, each panel says so and shows the hint for starting it.

---

## 9. Technology Stack

| Layer | Choice |
|---|---|
| Storage | Parquet + DuckDB (zero-ops, columnar) |
| Solar physics | `pvlib` — peer-reviewed reference implementation |
| Wind physics | IEC 61400-12 power-curve normalization |
| Modelling | LightGBM (quantile objective) |
| Calibration | Custom split-conformal implementation |
| Tuning | Optuna |
| Explainability | SHAP |
| Optimisation | PuLP (LP dispatch) |
| API | FastAPI + Pydantic |
| Validation | Pandera schema contracts |

Every component is open-source with no licensing cost and no vendor dependency in the critical path.

---

*For dataset provenance and licensing, see [`SOURCES.md`](SOURCES.md). For the full technical design rationale, see [`context/01-technical-approach.md`](context/01-technical-approach.md) and [`context/06-system-architecture.md`](context/06-system-architecture.md).*
