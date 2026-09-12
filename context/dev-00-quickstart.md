# dev-00 — Quickstart and Team Playbook

**Team ZERO BIAS** · Read this before writing any code. It takes fifteen minutes and prevents the two
failures that kill projects of this shape: everyone blocked on one person, and integration left to the end.

---

## 1. The three contracts — fix these first

Before anyone opens an editor, agree these three function signatures. Everything else is implementation
detail behind them, and agreeing them is what lets four people work at once without collisions.

```python
# src/features/build.py
def build_features(
    weather: pd.DataFrame,          # Layer-0 weather_nwp rows
    site: SiteMaster,               # one archetype
    actuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pure function. Same code path for training and inference.
    Index: (run_ts_utc, valid_ts_utc). No I/O, no globals, no config reads."""


# src/models/predict.py
def predict(
    region_id: str,
    run_ts: pd.Timestamp,
    horizons: range = range(1, 73),
    quantiles: tuple = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """One row per (valid_ts, tech). Columns:
    p10_mw, p50_mw, p90_mw, model_version, calibrated: bool"""


# src/decisions/recommend.py
def recommend(
    outlook: pd.DataFrame,          # quantiles + demand
    grid_state: GridState,          # must_run, storage SoC, ramp limits
    config: RegionConfig,
) -> pd.DataFrame:
    """Ranked actions. Columns:
    action, mwh, value_inr, confidence, rationale, valid_from, valid_to"""
```

**Write these as stubs returning empty DataFrames with the right columns on day one, and commit them.**
Everyone can then build against a real signature instead of waiting.

> `build_features` being a **pure function with no I/O** is not stylistic. The most common production
> failure in forecasting systems is training and serving computing features differently — a different fill
> rule, a different timezone, a lag from a different origin. One pure function removes that class of bug
> structurally rather than by discipline.

---

## 2. Repository layout

```
renewable-forecast-platform/
├── config/
│   ├── regions/            belgium.yaml · india_rajasthan.yaml
│   └── storage/            default.yaml
├── data/
│   ├── raw/                see data.md — gitignored
│   ├── interim/            gitignored
│   └── processed/          gitignored
├── artifacts/              cached replay data + model files — gitignored except .gitkeep
├── src/
│   ├── ingest/             adapters/{openmeteo,elia,zenodo_india}.py · scheduler.py
│   ├── quality/            schemas.py · validators.py · curtailment.py
│   ├── features/           build.py · solar.py · wind.py · temporal.py · nwp_quality.py · archetype.py
│   ├── models/             baselines.py · physics.py · residual_gbdt.py · quantile.py · registry.py · predict.py
│   ├── uncertainty/        conformal.py · coverage.py
│   ├── decisions/          events.py · net_load.py · storage_sim.py · recommend.py · optimiser.py
│   ├── evaluation/         metrics.py · walk_forward.py · skill.py · drift.py
│   └── api/                main.py · routes/ · schemas.py · deps.py
├── ui/                     React app — see dev-02
├── notebooks/              exploration ONLY — never imported by src/
├── tests/                  unit/ · integration/ · fixtures/
├── scripts/                dl_*.py from data.md · train.py · backtest.py
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

`notebooks/` is exploration-only and nothing in `src/` may import from it. Notebook code drifting into
production is a classic and avoidable source of train/serve inconsistency.

---

## 3. Environment setup

```bash
git clone <repo> && cd renewable-forecast-platform
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
pre-commit install
```

`pyproject.toml`:

```toml
[project]
name = "renewable-forecast-platform"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115", "uvicorn[standard]>=0.32", "pydantic>=2.9", "pydantic-settings>=2.6",
  "pandas>=2.2", "numpy>=1.26", "pyarrow>=17", "duckdb>=1.1",
  "pvlib>=0.11", "windpowerlib>=0.2",
  "lightgbm>=4.5", "scikit-learn>=1.5", "mapie>=0.9", "shap>=0.46",
  "pandera>=0.20", "requests>=2.32", "httpx>=0.27",
  "apscheduler>=3.10", "pyyaml>=6.0", "structlog>=24.4", "typer>=0.13",
  "xarray>=2024.9", "netCDF4>=1.7", "redis>=5.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov", "pytest-asyncio", "hypothesis>=6.112",
       "ruff>=0.7", "mypy>=1.13", "mlflow>=2.17", "matplotlib>=3.9", "jupyterlab"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
```

`.env.example`:

```bash
ENV=dev
DATA_ROOT=./data
ARTIFACT_ROOT=./artifacts
REPLAY_MODE=false            # true serves cached forecast runs — demo insurance
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=file:./artifacts/mlruns
LOG_LEVEL=INFO
# optional
NREL_API_KEY=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

---

## 4. Day-1 order of work

Do these in sequence. Each is a real checkpoint, not a formality.

| # | Task | Owner | Done when |
|---|---|---|---|
| 1 | Repo skeleton + `pyproject.toml` + pre-commit | Lead | `pip install -e ".[dev]"` succeeds |
| 2 | The three contract stubs, committed | Lead | Everyone can import them |
| 3 | `config/regions/belgium.yaml` | Lead | Loads into a `RegionConfig` |
| 4 | Download Tier-1 data per `data.md` | Data | §15 verification passes |
| 5 | **Evaluation harness** — `evaluation/metrics.py` + `walk_forward.py` | ML | Scores a persistence baseline end to end |
| 6 | Persistence + smart-persistence baselines | ML | Error-vs-lead-hour plot exists |
| 7 | FastAPI skeleton with `/health` and stubbed routes | Backend | `curl localhost:8000/health` returns |
| 8 | React shell + API client + design tokens | Frontend | Renders a chart from stubbed API data |

> **Build the evaluation harness before the second model exists.** Teams that add evaluation late end up
> with numbers they cannot reproduce and quietly stop quoting them. It is one function, one held-out
> window, one metrics table — and every later claim routes through it.

---

## 5. Parallel work split (four people)

The dependency graph allows genuine parallelism after day 1 because of the contracts.

```
Day 1  ── everyone ── contracts, config, data download, harness
          │
          ├── ML/Physics ──── features (M3) ──► models (M4) ──► uncertainty (M5)
          ├── Backend ─────── ingest (M1,M2) ──► API (M9) ──► decisions (M7)
          ├── Frontend ────── design system ──► components ──► screens (M10)
          └── Data/Eval ───── quality (M2) ──► backtest (M8) ──► events (M6)
```

| Role | Owns | Reads | Never touches |
|---|---|---|---|
| **ML / Physics** | `features/`, `models/`, `uncertainty/` | `config/`, `data/processed/` | `api/`, `ui/` |
| **Backend** | `ingest/`, `api/`, `decisions/` | contracts, `config/` | `features/`, `ui/` |
| **Frontend** | `ui/` | the OpenAPI schema only | all of `src/` |
| **Data / Eval** | `quality/`, `evaluation/`, `scripts/` | everything read-only | `api/`, `ui/` |

**The frontend works against the OpenAPI schema, never against the Python.** FastAPI generates it at
`/openapi.json`; generate a typed client from it (§dev-02 §7). That decouples the two entirely — the UI
can be finished before a single real forecast exists.

---

## 6. Git workflow

Keep it light. Trunk-based with short branches.

```bash
git switch -c feat/solar-physics-features
# ... work ...
git add -A && git commit -m "feat(features): pvlib clear-sky index + POA transposition"
git push -u origin feat/solar-physics-features
gh pr create --fill
```

**Commit prefixes:** `feat` · `fix` · `refactor` · `test` · `docs` · `chore` · `data`
**Scope** is the module: `feat(models):`, `fix(api):`, `test(features):`

**Branch naming:** `feat/<short-slug>`, `fix/<short-slug>`, `spike/<short-slug>`

Merge to `main` early and often. A branch older than a day is a merge conflict waiting to happen at 2 a.m.

---

## 7. Definition of done

A task is done when **all** of these hold. Not four of five.

- [ ] Code merged to `main`
- [ ] At least one test that would fail if the change were reverted
- [ ] Type hints on every public function
- [ ] `ruff check` and `ruff format` clean
- [ ] No hardcoded paths, thresholds or credentials — they live in `config/` or `.env`
- [ ] If it changes a data schema, the `pandera` schema is updated too
- [ ] If it changes model behaviour, the backtest was re-run and the number recorded

---

## 8. Coding conventions that matter here

**Time.** Every timestamp in `src/` is UTC and timezone-aware. Convert to local only at the presentation
layer. Store `run_ts_utc` and `valid_ts_utc` separately, always — never collapse to one.

**Units in names.** `power_mw`, `ghi_wm2`, `ws_hub_ms`, `energy_mwh`. A variable called `wind_speed` with
no unit suffix is a bug waiting to happen; Open-Meteo defaults to km/h and power scales as v³.

**Config over code.** Regions, archetypes, thresholds, prices and storage parameters live in YAML. If you
are about to type a number into a function body, it probably belongs in `RegionConfig`.

**Fail loudly at boundaries.** Every layer transition validates with `pandera`. Bad data should raise at
the boundary, not silently propagate into a model.

**Logging.** `structlog`, JSON in production. Log `run_ts`, `region_id` and `model_version` on every
forecast so a bad number can be traced back afterwards.

```python
import structlog
log = structlog.get_logger()
log.info("forecast.generated", region_id=region, run_ts=str(run_ts),
         model_version=mv, n_rows=len(df), calibrated=True)
```

---

## 9. Daily rhythm

**Standup — 10 minutes, three questions.** What merged yesterday? What is blocking? What number moved?
The third question keeps the team pointed at the backtest rather than at line count.

**End of day — run the harness.** `python scripts/backtest.py --region BE --quick` and paste the skill
score in the channel. A model that got worse should be visible within a day, not a week.

---

## 10. The document set

| File | What it covers |
|---|---|
| `dev-00-quickstart.md` | This file — contracts, setup, team split, conventions |
| `dev-01-backend.md` | FastAPI structure, full endpoint spec, services, scheduler, errors |
| `dev-02-frontend.md` | Design system, component library, screens, state, API client |
| `dev-03-ml-pipeline.md` | Features, models, conformal calibration, evaluation — with code |
| `dev-04-data-layer.md` | Storage layout, schemas, pandera contracts, DuckDB queries |
| `dev-05-devops-testing.md` | Docker, CI, testing strategy, replay mode, demo insurance |
| `data.md` | Where every dataset comes from and how to download it |

Architecture and rationale live in `01-technical-approach.md` and `06-system-architecture.md`. These dev
files say *how to build it*; those say *why it is built that way*. When they disagree, the architecture
docs win and this set should be corrected.
