# dev-05 — DevOps, Testing and Demo Insurance

**Principle:** the demo must work when the Wi-Fi does not. Build the live path, present from cache, and
say which you are showing.

---

## 1. Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./config:/app/config:ro
    depends_on: [redis]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s

  worker:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    command: python -m src.ingest.scheduler
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./config:/app/config:ro
    depends_on: [redis]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  ui:
    build: { context: ./ui, dockerfile: ../docker/Dockerfile.ui }
    ports: ["5173:80"]
    depends_on: [api]
```

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`libgomp1` is required by LightGBM at runtime. Leaving it out produces an import error that only appears
inside the container — a classic hour lost to a one-line fix.

```dockerfile
# docker/Dockerfile.ui
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

```bash
docker compose up --build          # everything
docker compose up api redis        # backend only, UI on the Vite dev server
```

---

## 2. Replay mode — the most valuable 30 lines in the repo

Live APIs fail during demonstrations far more often than models do. Conference Wi-Fi, an expired token, a
cold container — all more likely than a bad forecast.

```python
# src/ingest/replay.py
import pathlib, pandas as pd
from ..core.config import get_settings

REPLAY_ROOT = pathlib.Path("artifacts/replay")

def snapshot(region_id: str, run_ts) -> None:
    """Freeze one good cycle: weather, forecast, outlook, events, actions."""
    dest = REPLAY_ROOT / region_id / pd.Timestamp(run_ts).strftime("%Y%m%dT%H")
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in collect_cycle(region_id, run_ts).items():
        df.to_parquet(dest / f"{name}.parquet", index=False)

def load_replay(region_id: str, table: str) -> pd.DataFrame:
    runs = sorted((REPLAY_ROOT / region_id).iterdir())
    if not runs:
        raise FileNotFoundError(f"no replay snapshot for {region_id}")
    return pd.read_parquet(runs[-1] / f"{table}.parquet")

def read_table(region_id: str, table: str, store) -> pd.DataFrame:
    if get_settings().replay_mode:
        return load_replay(region_id, table)
    return store.read(region_id, table)
```

**Rules:**

1. Snapshot a good week during development: `python scripts/snapshot_replay.py --region BE --days 7`.
2. Commit model artifacts to `artifacts/models/` — or at minimum have them on the demo machine.
3. `REPLAY_MODE=true` must be visible in the UI status bar. This is an integrity feature, not a debug flag.
4. Test the replay path in CI, so it cannot rot.

```bash
REPLAY_MODE=true docker compose up      # the demo command
```

---

## 3. Testing strategy

Ordered by how often each catches a real bug in this specific project.

### 3.1 Physics unit tests — exactly checkable, so check them

```python
# tests/unit/test_physics.py
import pvlib, pandas as pd, numpy as np
from src.features.wind import power_curve

def test_clearsky_ghi_known_value():
    """Solar noon in June at the equator should give near-maximum clear-sky GHI."""
    loc = pvlib.location.Location(0, 0, tz="UTC")
    cs = loc.get_clearsky(pd.DatetimeIndex(["2026-06-21 12:00"], tz="UTC"))
    assert 900 < cs["ghi"].iloc[0] < 1100

def test_clearsky_zero_at_night():
    loc = pvlib.location.Location(50.85, 4.35, tz="UTC")
    cs = loc.get_clearsky(pd.DatetimeIndex(["2026-01-15 00:00"], tz="UTC"))
    assert cs["ghi"].iloc[0] == 0

def test_power_curve_breakpoints():
    assert power_curve(2.0)  == 0.0        # below cut-in
    assert power_curve(12.0) == 1.0        # at rated
    assert power_curve(20.0) == 1.0        # plateau
    assert power_curve(26.0) == 0.0        # above cut-out — a cliff, not a slope

def test_power_curve_monotone_on_ramp():
    v = np.linspace(3, 12, 50)
    assert np.all(np.diff(power_curve(v)) >= 0)

def test_density_correction_direction():
    """Hot thin air ⇒ lower effective wind speed ⇒ less power. Relevant for Indian sites."""
    from src.features.wind import wind_features
    hot  = wind_features(_wx(temp_c=45), _site())
    cool = wind_features(_wx(temp_c=15), _site())
    assert hot.ws_density_corrected_ms.iloc[0] < cool.ws_density_corrected_ms.iloc[0]
```

### 3.2 Property tests

```python
# tests/unit/test_properties.py
from hypothesis import given, strategies as st

@given(st.lists(st.floats(0, 1), min_size=3, max_size=3))
def test_quantiles_never_cross(vals):
    out = sort_quantiles(pd.DataFrame([dict(zip(["p10_mw","p50_mw","p90_mw"], vals))]))
    assert out.p10_mw.iat[0] <= out.p50_mw.iat[0] <= out.p90_mw.iat[0]

@given(st.floats(0, 2000), st.floats(1, 10000))
def test_forecast_bounded_by_capacity(poa, cap):
    assert 0 <= physics_power(poa, cap) <= cap
```

### 3.3 The train/serve parity test — the most important test in the repository

```python
# tests/integration/test_feature_parity.py
def test_training_and_serving_features_identical(fixture_week):
    """The one test that guards the failure mode most likely to destroy a deployed
    forecasting system — and the one almost always missing."""
    train_feats = build_features(fixture_week.weather, fixture_week.site,
                                 actuals=fixture_week.actuals)
    serve_feats = build_features(fixture_week.weather, fixture_week.site,
                                 actuals=None)          # serving has no future actuals

    shared = [c for c in train_feats.columns if not c.startswith("power_lag")]
    pd.testing.assert_frame_equal(
        train_feats[shared], serve_feats[shared], check_exact=False, rtol=1e-9,
    )
```

If this passes, features computed at 02:00 during nightly training are byte-identical to those computed at
09:00 during inference. If it fails, the model is being served inputs it was never trained on.

### 3.4 Golden-file integration test

```python
def test_pipeline_end_to_end(fixture_week, tmp_path):
    out = run_pipeline(fixture_week, out_dir=tmp_path)
    golden = pd.read_parquet("tests/fixtures/golden_forecast.parquet")
    pd.testing.assert_frame_equal(
        out[["valid_ts_utc","tech","p50_mw"]], golden, rtol=1e-6,
    )
```

Regenerate the golden file deliberately, never casually — and say so in the commit message when you do.

### 3.5 API contract tests

```python
def test_every_response_carries_provenance(client):
    for path in ["/forecast", "/outlook", "/events", "/actions"]:
        r = client.get(path, params={"region_id": "BE"})
        assert r.status_code == 200
        body = r.json()
        for k in ("model_version", "issued_at", "replay_mode", "region_id"):
            assert k in body, f"{path} missing {k}"

def test_unknown_region_returns_404_with_hint(client):
    r = client.get("/forecast", params={"region_id": "ZZ"})
    assert r.status_code == 404
    assert r.json()["hint"]

def test_api_serves_without_redis(client_no_redis):
    """A cache is an optimisation, never a dependency."""
    assert client_no_redis.get("/forecast", params={"region_id": "BE"}).status_code == 200
```

### Coverage targets

| Area | Target | Why |
|---|---|---|
| `features/` | 90 % | Physics is exactly checkable; a regression here corrupts every forecast |
| `decisions/` | 85 % | Arithmetic, easy to test, high consequence |
| `evaluation/` | 90 % | If the harness is wrong, every reported number is wrong |
| `api/` | 70 % | Contract tests matter more than line coverage |
| `ingest/` | 60 % | Mostly I/O; test the transforms, mock the network |

---

## 4. CI

```yaml
# .github/workflows/ci.yml
name: ci
on: { push: { branches: [main] }, pull_request: }

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src --ignore-missing-imports
      - run: pytest --cov=src --cov-report=term-missing --cov-fail-under=70
      - name: replay path must work offline
        run: REPLAY_MODE=true pytest tests/integration/test_replay.py

  frontend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: ui } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20", cache: npm, cache-dependency-path: ui/package-lock.json }
      - run: npm ci
      - run: npm run lint
      - run: npx tsc --noEmit
      - run: npm run test -- --run
      - run: npm run build
```

CI must never hit a live external API. Network calls are mocked; the fixture week is committed.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks: [{ id: ruff, args: [--fix] }, { id: ruff-format }]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=2000]        # stops a 600 MB Parquet reaching the repo
```

---

## 5. Makefile

```makefile
.PHONY: setup data train backtest api ui test lint demo clean

setup:      ; pip install -e ".[dev]" && cd ui && npm ci
data:       ; python scripts/dl_openmeteo_hist_forecast.py && python scripts/dl_elia.py
train:      ; python scripts/train.py --region BE --tech solar --tech wind
backtest:   ; python scripts/backtest.py --region BE --report artifacts/backtest.html
api:        ; uvicorn src.api.main:app --reload --port 8000
ui:         ; cd ui && npm run dev
test:       ; pytest -q && cd ui && npm run test -- --run
lint:       ; ruff check src tests && ruff format --check src tests && mypy src
snapshot:   ; python scripts/snapshot_replay.py --region BE --days 7
demo:       ; REPLAY_MODE=true docker compose up --build
clean:      ; rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache
```

---

## 6. Observability

```python
# src/core/logging.py
import structlog, logging, sys

def setup_logging(level: str = "INFO"):
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
    )
```

Every forecast logs `region_id`, `run_ts`, `model_version`, `n_rows`, `calibrated`. Without those, "which
model produced this number?" is unanswerable — and that is exactly the question asked after a bad forecast.

**`/health` surfaces the things that actually go wrong:**

```python
@router.get("/health", response_model=Health)
def health(store=Depends(get_store)):
    w = []
    if (age := calibration_age_days()) and age > 7:
        w.append(f"calibration is {age} days old")
    if (lag := minutes_since_last_ingest()) and lag > 400:
        w.append(f"no weather ingest for {lag} minutes")
    if get_settings().replay_mode:
        w.append("serving cached replay data")
    return Health(status="degraded" if w else "healthy", warnings=w, ...)
```

---

## 7. Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| Weather API unreachable | HTTP error / timeout | Serve last successful run, mark `stale`, warn on `/health` |
| Weather schema changed | pandera failure | Reject the batch, alert, keep serving the previous forecast |
| Generation feed gap | `qc_flag = MISSING` | Impute for features; `sample_weight = 0` for training |
| Frozen sensor | rolling variance ≈ 0 | `qc_flag = FROZEN`, exclude from training |
| Curtailment unflagged | output flat below physics under good conditions | Heuristic flag, `sample_weight = 0` |
| Model drift | feature and error distribution monitors | Trigger retrain; alert if the candidate fails the gate |
| Calibration stale | `calibration_date` age > 7 days | Warn in every response; recalibrate |
| Retrained model worse | candidate fails the held-out gate | Keep incumbent, log the attempt, alert |
| Demo-time network failure | — | `REPLAY_MODE=true` |

---

## 8. Pre-demo checklist

Run this the night before, not on the morning.

- [ ] `make demo` works with Wi-Fi **switched off**
- [ ] Replay snapshot covers a day with a real over-generation event worth showing
- [ ] Model artifacts present in `artifacts/models/`
- [ ] `REPLAY` chip visible in the UI status bar
- [ ] Backtest report regenerated and the numbers match the slides
- [ ] Coverage plot renders — it is the differentiating fifteen seconds
- [ ] All four skill-curve lines present: persistence, physics, model, TSO
- [ ] `/health` returns `healthy` (or the warnings are ones you can explain)
- [ ] Laptop display scaling tested at the projector resolution
- [ ] One specific date chosen to narrate — "on 14 March we flagged a 340 MW ramp six hours early" beats
      any aggregate statistic

---

## 9. DevOps checklist

- [ ] `docker compose up` starts everything from a clean clone
- [ ] `libgomp1` installed in the API image (LightGBM runtime dependency)
- [ ] CI runs lint, types, tests and the frontend build
- [ ] CI never calls a live external API
- [ ] Replay path tested in CI
- [ ] Large files blocked by pre-commit
- [ ] Structured JSON logs with `region_id`, `run_ts`, `model_version`
- [ ] `/health` reports ingest lag, calibration age and replay state
- [ ] Secrets only in `.env`; `.env` in `.gitignore`; `.env.example` committed
