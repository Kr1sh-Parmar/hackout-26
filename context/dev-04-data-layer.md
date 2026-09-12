# dev-04 — Data Layer

**Stack:** Parquet on disk · DuckDB for query · pandera for contracts
**Principle:** bad data fails loudly at a layer boundary, never silently downstream.

---

## 1. Medallion layout

```
data/
├── raw/        exactly as downloaded — never modified, never written to by code
├── bronze/     parsed to Parquet, typed, UTC. No business logic.
├── silver/     validated, deduplicated, quality-flagged
└── gold/       model matrix + forecasts + outlook + actions — what the API reads
```

Partition gold by region and run date so DuckDB prunes files instead of scanning:

```
data/gold/forecast/region_id=BE/run_date=2026-09-12/part-0.parquet
data/gold/outlook/region_id=BE/run_date=2026-09-12/part-0.parquet
data/gold/actions/region_id=BE/run_date=2026-09-12/part-0.parquet
data/silver/weather_nwp/region_id=BE/run_date=2026-09-12/part-0.parquet
data/silver/generation/region_id=BE/year=2025/part-0.parquet
```

`raw/` is append-only. If a transform is wrong, fix the transform and rebuild bronze — never edit raw.
That property is what makes the pipeline reproducible from scratch.

---

## 2. Table reference

### `site_master` — static, hand-authored

| Column | Type | Unit | Note |
|---|---|---|---|
| `site_id` | str | — | PK, e.g. `BE-PV-ROOFTOP-S` |
| `region_id` | str | — | FK to region config |
| `tech` | enum | — | `solar` \| `wind` |
| `lat`, `lon` | float | ° | |
| `elevation_m` | float | m | |
| `capacity_mw` | float | MW | AC nameplate — the normaliser for everything |
| `capacity_share` | float | 0–1 | Archetype weight; must sum to 1 per region-tech |
| `tilt_deg`, `azimuth_deg` | float | ° | solar only |
| `tracking` | enum | — | `fixed` \| `single_axis` \| `dual_axis` |
| `dc_ac_ratio` | float | — | drives inverter clipping |
| `gamma_pdc` | float | /°C | temp coefficient, e.g. −0.0035 |
| `albedo`, `soiling_pct_per_day` | float | — | solar only |
| `n_turbines`, `rated_kw` | int | — | wind only |
| `hub_height_m`, `rotor_diameter_m` | float | m | wind only |
| `cut_in_ms`, `rated_ms`, `cut_out_ms` | float | m/s | typically 3 / 12 / 25 |
| `shear_alpha` | float | — | ~0.14 open, ~0.25 forested |
| `wake_loss_frac` | float | 0–1 | |

### `weather_nwp` — keyed by (run, valid)

| Column | Type | Note |
|---|---|---|
| `run_ts_utc` | timestamp | **PK part** — model initialisation |
| `valid_ts_utc` | timestamp | **PK part** — the hour being forecast |
| `lead_hours` | int | `valid − run`. A **feature**, not metadata. |
| `nwp_model` | str | `ecmwf_ifs025` \| `icon_seamless` \| `gfs_seamless` |
| `grid_point_id` | str | **PK part** |
| `weight` | float | capacity weight of this grid point |
| `ghi_wm2`, `dni_wm2`, `dhi_wm2` | float | W/m² |
| `temperature_2m_c`, `dew_point_2m_c` | float | °C |
| `relative_humidity_2m_pct` | float | % |
| `surface_pressure_hpa` | float | hPa |
| `cloud_cover_pct` + `_low/_mid/_high` | float | % — **keep separate**, see below |
| `wind_speed_10m_ms`, `wind_speed_100m_ms` | float | **m/s** — not km/h |
| `wind_direction_100m_deg`, `wind_gusts_10m_ms` | float | ° / m/s |
| `precipitation_mm`, `snowfall_cm`, `visibility_m` | float | |
| `is_day` | int8 | 0/1 |

**Primary key: `(run_ts_utc, valid_ts_utc, nwp_model, grid_point_id)`.**

> This compound key is the single most important schema decision in the project. Storing only `valid_ts`
> overwrites a 48-hour-old forecast with a 6-hour-old one, and every backtest afterwards silently cheats.
> The bug only surfaces in production, when live accuracy is far worse than the backtest promised.

**Keep the cloud layers separate.** Low cloud is optically thick and destroys PV output; high cirrus barely
dents it. Collapsing them into one `cloud_cover` discards most of the signal.

### `generation_actuals` — the label source

| Column | Type | Note |
|---|---|---|
| `region_id`, `ts_utc`, `tech` | — | PK |
| `power_mw` | float | metered output |
| `monitored_capacity_mw` | float | **changes as the fleet grows** — this is why the target is capacity factor |
| `load_factor` | float | |
| `curtailed_mw` | float | censored-label flag |
| `availability_pct` | float | |
| `qc_flag` | enum | `OK` \| `MISSING` \| `FROZEN` \| `OUT_OF_RANGE` \| `CURTAILED` |

### `training_matrix` (gold)

All Layer-1 features, union of solar and wind, NaN where not applicable, plus:

| Column | Note |
|---|---|
| `y_capacity_factor` | `power_mw / monitored_capacity_mw` — **the training target** |
| `y_power_mw` | reported, never trained on |
| `sample_weight` | 0 where curtailed, missing or unavailable |

---

## 3. Pandera contracts

```python
# src/quality/schemas.py
import pandera as pa
from pandera import Column, Check, DataFrameSchema

weather_nwp_schema = DataFrameSchema(
    {
        "run_ts_utc":              Column("datetime64[ns, UTC]", nullable=False),
        "valid_ts_utc":            Column("datetime64[ns, UTC]", nullable=False),
        "lead_hours":              Column(int,   Check.in_range(0, 168)),
        "nwp_model":               Column(str,   Check.isin(["ecmwf_ifs025","icon_seamless","gfs_seamless"])),
        "grid_point_id":           Column(str,   nullable=False),
        "weight":                  Column(float, Check.in_range(0, 1)),
        "ghi_wm2":                 Column(float, Check.in_range(0, 1400), nullable=True),
        "dni_wm2":                 Column(float, Check.in_range(0, 1100), nullable=True),
        "dhi_wm2":                 Column(float, Check.in_range(0, 700),  nullable=True),
        "temperature_2m_c":        Column(float, Check.in_range(-60, 60)),
        "surface_pressure_hpa":    Column(float, Check.in_range(800, 1100)),
        "relative_humidity_2m_pct":Column(float, Check.in_range(0, 100)),
        "cloud_cover_pct":         Column(float, Check.in_range(0, 100)),
        "wind_speed_10m_ms":       Column(float, Check.in_range(0, 90)),
        "wind_speed_100m_ms":      Column(float, Check.in_range(0, 110)),
        "wind_direction_100m_deg": Column(float, Check.in_range(0, 360)),
        "is_day":                  Column("int8", Check.isin([0, 1])),
    },
    unique=["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"],
    strict="filter",     # drop unexpected columns rather than failing on a new API field
    coerce=True,
)

generation_schema = DataFrameSchema(
    {
        "region_id":             Column(str),
        "ts_utc":                Column("datetime64[ns, UTC]"),
        "tech":                  Column(str,   Check.isin(["solar", "wind"])),
        "power_mw":              Column(float, Check.ge(0)),
        "monitored_capacity_mw": Column(float, Check.gt(0)),
        "curtailed_mw":          Column(float, Check.ge(0), nullable=True),
        "qc_flag":               Column(str,   Check.isin(
            ["OK","MISSING","FROZEN","OUT_OF_RANGE","CURTAILED"])),
    },
    checks=[
        pa.Check(lambda d: (d.power_mw <= d.monitored_capacity_mw * 1.05).all(),
                 error="power exceeds monitored capacity by more than 5%"),
    ],
    unique=["region_id", "ts_utc", "tech"],
    coerce=True,
)

training_matrix_schema = DataFrameSchema(
    {
        "y_capacity_factor": Column(float, Check.in_range(0, 1.05), nullable=True),
        "sample_weight":     Column(float, Check.in_range(0, 1)),
        "lead_hours":        Column(int,   Check.in_range(1, 168)),
        "clearsky_index_kt": Column(float, Check.in_range(0, 1.35), nullable=True),
    },
    strict=False, coerce=True,
)
```

`strict="filter"` on the weather schema is deliberate: Open-Meteo occasionally adds a variable, and the
pipeline should drop it rather than crash the whole ingest cycle.

`clearsky_index_kt` is allowed above 1.0 up to 1.35 — cloud-edge enhancement is physically real, and a
`Check.le(1.0)` here would reject valid high-output hours.

### Applying contracts

```python
# src/quality/validators.py
import structlog
log = structlog.get_logger()

def validate(df, schema, name: str, strict: bool = True):
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        log.error("schema.failed", table=name, n_failures=len(e.failure_cases),
                  cases=e.failure_cases.head(20).to_dict("records"))
        if strict:
            raise
        good = df.drop(index=e.failure_cases["index"].dropna().unique())
        log.warning("schema.partial", table=name, kept=len(good), dropped=len(df) - len(good))
        return good
```

Use `strict=True` on ingest (a bad batch should not enter the lake) and `strict=False` on historical
backfill, where dropping a handful of bad rows beats losing the run.

---

## 4. DuckDB queries

```python
import duckdb
con = duckdb.connect(":memory:")

# latest forecast for a region
con.execute("""
SELECT valid_ts_utc, tech, p10_mw, p50_mw, p90_mw, lead_hours
FROM read_parquet('data/gold/forecast/**/*.parquet', hive_partitioning=1)
WHERE region_id = ?
  AND run_ts_utc = (SELECT max(run_ts_utc)
                    FROM read_parquet('data/gold/forecast/**/*.parquet', hive_partitioning=1)
                    WHERE region_id = ?)
ORDER BY valid_ts_utc, tech
""", ["BE", "BE"]).df()

# capacity-weighted regional weather from per-grid-point rows
con.execute("""
SELECT valid_ts_utc, nwp_model,
       sum(ghi_wm2            * weight) AS ghi_wm2,
       sum(wind_speed_100m_ms * weight) AS ws_100m_ms,
       sum(temperature_2m_c   * weight) AS temp_c
FROM read_parquet('data/silver/weather_nwp/**/*.parquet', hive_partitioning=1)
WHERE region_id = ? AND run_ts_utc = ?
GROUP BY 1, 2 ORDER BY 1, 2
""", ["BE", run_ts]).df()

# join forecast to actuals for the backtest — the join that must be exactly right
con.execute("""
SELECT f.valid_ts_utc, f.lead_hours, f.tech,
       f.p10_mw, f.p50_mw, f.p90_mw,
       a.power_mw AS y_true, a.monitored_capacity_mw, a.qc_flag
FROM read_parquet('data/gold/forecast/**/*.parquet', hive_partitioning=1) f
JOIN read_parquet('data/silver/generation/**/*.parquet', hive_partitioning=1) a
  ON  f.valid_ts_utc = a.ts_utc
  AND f.tech         = a.tech
  AND f.region_id    = a.region_id
WHERE f.region_id = ? AND a.qc_flag = 'OK'
""", ["BE"]).df()
```

The backtest join filters `qc_flag = 'OK'`, which is what excludes curtailed and missing intervals from
evaluation. Forgetting that filter is the easiest way to publish a wrong accuracy number.

---

## 5. Ingest adapters

Each source implements the same interface, so adding a region or swapping a weather provider is one class.

```python
# src/ingest/adapters/base.py
from abc import ABC, abstractmethod
import pandas as pd

class SourceAdapter(ABC):
    name: str
    @abstractmethod
    def fetch(self, **kw) -> pd.DataFrame: ...
    @abstractmethod
    def to_bronze(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise units, rename to canonical columns, set UTC index."""
```

```python
# src/ingest/adapters/openmeteo.py
class OpenMeteoAdapter(SourceAdapter):
    name = "open-meteo"
    FORECAST = "https://api.open-meteo.com/v1/forecast"
    HIST_FC  = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    RENAME = {
        "shortwave_radiation": "ghi_wm2",
        "direct_normal_irradiance": "dni_wm2",
        "diffuse_radiation": "dhi_wm2",
        "wind_speed_10m": "wind_speed_10m_ms",
        "wind_speed_100m": "wind_speed_100m_ms",
        "wind_direction_100m": "wind_direction_100m_deg",
        "wind_gusts_10m": "wind_gusts_10m_ms",
        "temperature_2m": "temperature_2m_c",
        "surface_pressure": "surface_pressure_hpa",
        "relative_humidity_2m": "relative_humidity_2m_pct",
        "cloud_cover": "cloud_cover_pct",
    }

    def fetch(self, lat, lon, model, run_ts=None, days=4):
        params = {
            "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS, "models": model,
            "wind_speed_unit": "ms",     # ⚠️ defaults to km/h — power goes as v³
            "timezone": "UTC", "forecast_days": days, "past_days": 2,
        }
        r = httpx.get(self.FORECAST, params=params, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json()["hourly"])
```

> **`wind_speed_unit=ms` is the most consequential line in the adapter.** Open-Meteo returns km/h by
> default. Power scales as v³, so a missed conversion is roughly a 47× error in megawatts — and it fails
> silently, because the numbers still look like numbers. Assert it in a test.

---

## 6. Integration checks

Run before building any features. Each has caught a real bug in this class of project.

```python
# tests/integration/test_data_sanity.py
def test_units_sane(weather):
    assert weather.wind_speed_100m_ms.max() < 60,  "wind still in km/h"
    assert weather.ghi_wm2.max() < 1400,           "irradiance out of physical range"

def test_no_solar_at_night(generation, solar_position):
    night = generation[(solar_position.elevation < -5) & (generation.tech == "solar")]
    assert night.power_mw.max() < generation.monitored_capacity_mw.iloc[0] * 0.001, \
        "solar output at night — timezone misalignment between weather and generation"

def test_forecast_key_unique(weather):
    key = ["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"]
    assert not weather.duplicated(key).any(), "duplicate forecast rows — check the write path"

def test_all_models_present(weather, cfg):
    got = set(weather.nwp_model.unique())
    assert got == set(cfg.nwp_models), f"missing NWP models: {set(cfg.nwp_models) - got}"

def test_archetype_shares(cfg):
    for tech, arcs in cfg.archetypes.items():
        assert abs(sum(a.share for a in arcs) - 1.0) < 1e-6
```

The night-solar test is worth keeping permanently. Timezone misalignment between the weather source and
the generation source produces a model that looks fine on aggregate metrics and is wrong every single day.

---

## 7. Gap handling

```python
def gap_census(df, freq="15min"):
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    missing = full.difference(df.index)
    runs = []
    if len(missing):
        blocks = np.split(missing, np.where(np.diff(missing) > pd.Timedelta(freq))[0] + 1)
        runs = [(b[0], b[-1], len(b)) for b in blocks]
    return {"n_missing": len(missing), "longest_run": max((r[2] for r in runs), default=0),
            "blocks": runs[:20]}
```

| Gap length | Treatment |
|---|---|
| ≤ 1 hour | Linear interpolation; `qc_flag = OK` |
| 1–6 hours | Interpolate for features, `sample_weight = 0` for training |
| > 6 hours | Leave as NaN, `qc_flag = MISSING`, exclude from training and evaluation |

**Census before you choose a rule.** Filling a three-day outage by interpolation produces a plausible-looking
series that teaches the model nothing true.

---

## 8. Storage budget

| Table | Rows/year (1 region) | Size |
|---|---|---|
| `weather_nwp` (5 points × 3 models × 4 runs/day × 96 h) | ~2.1 M | ~180 MB |
| `generation_actuals` (15-min, 2 techs) | ~70 k | ~4 MB |
| `training_matrix` (hourly, ~60 features) | ~8.8 k | ~6 MB |
| `forecast` (4 runs/day × 72 h × 2 techs) | ~210 k | ~14 MB |
| `outlook` + `actions` | ~110 k | ~8 MB |

Roughly **210 MB per region-year**. Parquet with snappy compression; no need for anything cleverer at this
scale.

---

## 9. Data layer checklist

- [ ] `raw/` is never written to by pipeline code
- [ ] Every layer boundary validates with pandera
- [ ] Weather PK is `(run_ts, valid_ts, nwp_model, grid_point_id)` and uniqueness is enforced
- [ ] All timestamps UTC and timezone-aware in `src/`
- [ ] Every column name carries its unit
- [ ] Cloud layers stored separately, never collapsed
- [ ] `qc_flag` and `sample_weight` computed in silver, not in the model code
- [ ] Gold partitioned by `region_id` and `run_date`
- [ ] Gap census run before choosing an imputation rule
- [ ] `wind_speed_unit=ms` asserted in a test, not just in a comment
