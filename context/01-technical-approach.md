# Technical Approach

> **Scope.** Regional (system-level) 24–72 hour probabilistic forecasting of solar and wind generation, with a decision layer that converts forecasts into sized grid actions. Belgium (Elia) as the validation region; India as the transfer region.

---

## 1. Problem decomposition

"Forecast renewable output" is five distinct problems with different inputs, failure modes and metrics. Treating them separately is what prevents the common failure of building one monolithic model and calling it a platform.

| # | Sub-problem | The actual question | Why it is hard | Metric |
|---|---|---|---|---|
| 1 | **Weather → resource** | Turn a numerical weather prediction into expected irradiance / hub-height wind at this location | NWP grids are 9–25 km; the error is systematic bias, not noise | MBE, nRMSE on GHI / wind speed |
| 2 | **Resource → power** | Turn irradiance or wind speed into MW | Strongly non-linear and asset-specific: inverter clipping, soiling, wake losses, cut-out shutdown | nRMSE, nMAE on MW |
| 3 | **Uncertainty** | How wrong might this be, at each hour? | Error is heteroscedastic — near zero at 03:00, very large on a broken-cloud afternoon | Pinball loss, CRPS, PICP |
| 4 | **Event detection** | Where are the ramps and the over-generation windows? | A 2 % RMSE improvement is worth less than catching one 40 % ramp | Ramp hit rate, lead time |
| 5 | **Action** | Curtail, charge, discharge or commit backup — and why | Requires demand, must-run capacity, storage state and ramp limits, not just the forecast | MWh recovered, deviation cost avoided |

Sub-problems 4 and 5 are where the stated users operate, and where most submissions stop short.

### 1.1 Why the horizon determines the architecture

Forecast skill is governed by which information remains useful at a given lead time.

| Horizon | Dominant signal | Autoregressive lags | Standard approach |
|---|---|---|---|
| 0–30 min | Sky camera, satellite cloud motion | Strong | Persistence + cloud advection |
| 30 min – 6 h | Satellite, recent measured output | Useful | Smart persistence, GBDT on lags |
| **6–72 h** | **NWP only** | **Near-useless** | **NWP post-processing** |
| 3–14 d | Ensemble NWP, climatology | None | Probabilistic ensemble |

The brief specifies **24–72 hours**. This has a hard consequence that drives the entire feature design:

> At lead hour 48, `power_lag_1h` does not exist. Any model trained with short lags will score brilliantly in a naive backtest and fail completely in deployment, because the backtest silently leaked information the operational system will never have.

We therefore build the feature set around **weather forecasts and site physics**, and admit autoregressive features only at lags ≥ 24 h (where they legitimately encode slow state such as soiling accumulation or partial outage).

This also reframes the project: it is **NWP post-processing**, not time-series forecasting. That reframing is the single most consequential technical decision in the document.

---

## 2. Data strategy

### 2.1 Source selection

Three things are required and rarely ship together: plant/region generation history, matching weather, and site parameters.

| Role | Source | Resolution | Why this one |
|---|---|---|---|
| Solar truth + benchmark | Elia `ods032` / `ods087` | 15 min, 2013→ | Measured PV MW, monitored capacity, load factor, **and Elia's own day-ahead / week-ahead / P10 / P90 forecasts** |
| Wind truth + benchmark | Elia `ods031` / `ods086` | 15 min | Measured wind MW split offshore/onshore, same forecast horizons |
| **Demand** | Elia `ods001` / `ods002` / `ods003` | 15 min | Measured **and** forecast total load. Without this the decision layer runs on synthetic demand. |
| Weather | Open-Meteo Historical + Forecast API | Hourly | Identical variable names on both sides; ECMWF/ICON/GFS selectable; ensemble endpoint; no API key |
| Fleet parameters | `site_master.yaml` (authored) | static | Archetype rows per region and technology |
| Transfer region | Zenodo `10.5281/zenodo.7824872` + CEA/MNRE | Hourly, 1979–2022 | Hourly Indian solar/wind capacity factors, state-level and 1°×1° gridded, plus installed capacity and plant locations |
| Cross-check | Open Power System Data | 15–60 min | `DE_solar_generation_actual`, `DE_wind_capacity`, `DE_load_actual_entsoe_transparency` + four German TSO zones |

**The decisive property of Elia** is that generation, the operator's own forecast, and load all sit on one portal at one resolution with one join key. That removes the two most common project-killers at once: cross-source time alignment, and a decision layer running on invented demand.

### 2.2 Multi-model NWP, deliberately

International practice is unambiguous: combination forecasts beat single-model forecasts, with error minimised around **five to six NWP models**, and a recommended minimum of **at least one global plus one regional model** ([GET.transform, 2024](https://www.get-transform.eu/wp-content/uploads/2024/01/GET.transform-Brief_VRE-Forecasting-Solar-Wind.pdf)).

Open-Meteo exposes ECMWF IFS (global), DWD ICON (regional, high resolution over Europe), GFS/HRRR, and GEM through one interface. We ingest **three** and use their disagreement two ways:

- as a **point-forecast input** (the models' weighted blend), and
- as an **uncertainty feature** — `nwp_model_disagreement` is one of the strongest available predictors of our own error.

This is a case where the correct engineering choice and the impressive-sounding choice coincide, and it costs three API calls instead of one.

### 2.3 Regional scale changes the physics layer

A region has no single tilt or hub height. Two consequences:

**Fleet archetypes.** `site_master` becomes a weighted set of representative configurations rather than a plant record. For Belgian PV, approximately:

| Archetype | Tilt | Azimuth | Share of installed MW | Rationale |
|---|---|---|---|---|
| `rooftop_south` | 35° | 180° | ~45 % **[ASSUMED]** | Dominant residential retrofit |
| `rooftop_east_west` | 20° | 90° / 270° | ~30 % **[ASSUMED]** | Common on flat commercial roofs |
| `ground_mount` | 30° | 180° | ~25 % **[ASSUMED]** | Utility scale |

The physics chain runs per archetype and outputs a capacity-weighted sum. This is what lets the model reproduce the **morning shoulder** that east-west rooftop arrays produce and that a single-tilt model underestimates every day of the year. Shares are assumptions to be refined against observed generation shape — the fitting procedure is described in §4.4.

**Capacity-weighted weather grid.** Weather is sampled at 5–10 coordinates spread across the region, each weighted by installed capacity in its vicinity, rather than at one centroid. A single centroid systematically misses frontal systems crossing the region. Note that spatial aggregation also makes regional forecasting *easier* than plant forecasting: independent cloud noise partially cancels, which is why regional day-ahead nRMSE is typically better than single-plant nRMSE.

---

## 3. Four-layer data model

Fixing these schemas on day one is what allows parallel work without collisions.

```
Layer 0  raw ingest       →  Layer 1  derived features
                          →  Layer 2  model matrix
                          →  Layer 3  decisions
```

### Layer 0 — raw ingest

```sql
-- site_master : static, one row per archetype per region
site_id              TEXT PRIMARY KEY   -- 'BE-FL-PV-ROOFTOP-S'
region_id            TEXT
tech                 ENUM               -- 'solar' | 'wind'
lat, lon, elevation_m FLOAT
timezone             TEXT
capacity_mw          FLOAT              -- the normaliser for everything downstream
capacity_share       FLOAT              -- archetype weight within region, sums to 1.0

-- solar only
tilt_deg, azimuth_deg      FLOAT
tracking                   ENUM         -- 'fixed' | 'single_axis' | 'dual_axis'
dc_ac_ratio                FLOAT        -- drives inverter clipping
gamma_pdc                  FLOAT        -- temperature coefficient, e.g. -0.0035 /°C
albedo, soiling_pct_per_day FLOAT

-- wind only
n_turbines, rated_kw             INT
hub_height_m, rotor_diameter_m   FLOAT
cut_in_ms, rated_ms, cut_out_ms  FLOAT  -- 3 / 12 / 25 typical
shear_alpha                      FLOAT  -- ~0.14 open terrain, ~0.25 forested
wake_loss_frac                   FLOAT
```

```sql
-- generation_actuals : the label source
site_id, region_id, ts_utc, ts_local
power_mw            FLOAT   -- metered output
monitored_capacity_mw FLOAT -- Elia publishes this; it changes as the fleet grows
load_factor         FLOAT
curtailed_mw        FLOAT   -- CRITICAL: see §4.5, censored labels
availability_pct    FLOAT
qc_flag             ENUM    -- OK | MISSING | FROZEN | OUT_OF_RANGE | CURTAILED
```

```sql
-- weather_nwp : keyed by (run, valid), never by valid alone
run_ts_utc     TIMESTAMP   -- model initialisation time
valid_ts_utc   TIMESTAMP   -- the hour being forecast
lead_hours     INT         -- valid − run. A FEATURE, not metadata.
nwp_model      TEXT        -- 'ecmwf_ifs' | 'icon' | 'gfs'
grid_point_id  TEXT        -- which of the region's sample coordinates
weight         FLOAT       -- capacity weight of this grid point

temperature_2m_c, dew_point_2m_c, relative_humidity_2m_pct   FLOAT
surface_pressure_hpa                                          FLOAT
cloud_cover_pct, cloud_cover_low_pct,
cloud_cover_mid_pct, cloud_cover_high_pct                     FLOAT
ghi_wm2, dni_wm2, dhi_wm2                                     FLOAT
wind_speed_10m_ms, wind_speed_100m_ms                         FLOAT
wind_direction_100m_deg, wind_gusts_10m_ms                    FLOAT
precipitation_mm, snowfall_cm, visibility_m                   FLOAT
```

> **The `(run_ts_utc, valid_ts_utc)` compound key is the detail most projects get wrong.** Storing only `valid_ts` silently overwrites a 48-hour-old forecast with a 6-hour-old one. Every subsequent backtest is then cheating, and the error only surfaces in production. This single design choice is worth stating explicitly in any technical review.

### Layer 1 — derived features

**Solar branch** (computed with `pvlib`, per archetype):

| Feature | Definition | Why |
|---|---|---|
| `solar_zenith`, `solar_azimuth`, `airmass` | Solar position algorithm | Exact, free, deterministic |
| `clearsky_ghi/dni/dhi` | Ineichen–Perez clear-sky model | The physical ceiling for this location and moment |
| **`clearsky_index_kt`** | `ghi_nwp / clearsky_ghi` | **The single most important solar feature.** Normalises away season and latitude so January and June become comparable. |
| `poa_global` | Plane-of-array transposition using tilt/azimuth/tracking | What the panel actually receives, not what the ground receives |
| `cell_temperature` | SAPM: f(POA, air temp, wind speed) | Panels lose ~0.35 %/°C above 25 °C |
| `thermal_derate` | `1 + gamma_pdc × (T_cell − 25)` | Makes the temperature loss explicit and interpretable |
| **`physics_pac_mw`** | `min(dc_cap × POA/1000 × derate × η_inv, capacity_mw)` | The deterministic baseline — used as a **feature**, not the answer |
| `clipping_headroom` | `p_dc / capacity_mw` | Values > 1 indicate inverter clipping; a hard non-linearity |
| `mins_since_sunrise`, `mins_to_sunset` | From solar position | Captures morning fog burn-off and evening shading |

**Wind branch** (IEC 61400-12 normalisation):

| Feature | Definition | Why |
|---|---|---|
| `ws_hub` | `ws_100m × (hub_h/100) ** shear_alpha` | Power-law shear extrapolation to hub height |
| `air_density` | `P / (287.05 × T_kelvin)` | Power scales linearly with density |
| **`ws_density_corrected`** | `ws_hub × (ρ/1.225) ** (1/3)` | IEC standard normalisation. Recovers several points of nRMSE in hot or high-altitude conditions — directly relevant to Indian sites. |
| **`power_curve_cf`** | Piecewise manufacturer curve | Encodes cut-in, cubic ramp, rated plateau, cut-out |
| `physics_power_mw` | `power_curve_cf × capacity_mw × (1 − wake_loss)` | Deterministic baseline |
| **`dP_dv`** | Local gradient of the power curve | Sensitivity → directly predicts uncertainty. On the cubic ramp a small wind error is a large power error; on the rated plateau it is zero. |
| `turbulence_proxy` | `gust_10m / ws_10m − 1` | Proxy for turbulence intensity |
| `wind_dir_sin`, `wind_dir_cos` | Cyclic encoding | Never feed degrees raw — 359° and 1° are adjacent |
| `below_cutin_flag`, `above_cutout_flag` | Threshold indicators | Storm shutdown is a cliff, not a slope |

**Shared:**

| Family | Features |
|---|---|
| Cyclic | `hour_sin/cos`, `doy_sin/cos`, `is_weekend`, `is_holiday` |
| NWP quality | `lead_hours`, `nwp_model_disagreement`, `ensemble_spread`, `nwp_bias_lag_7d`, `nwp_ramp` |
| Autoregressive (≥ 24 h only) | `power_lag_24h`, `power_lag_168h`, `roll_mean_24h`, `roll_std_24h`, `smart_persistence` |
| Spatial **[STRETCH]** | `neighbour_region_cf`, satellite cloud advection (u, v), turbine-graph adjacency |

The NWP-quality family deserves emphasis: **these features predict our own error**. They are what converts a point forecast into an honest interval, and they are routinely omitted.

Note also that `cloud_cover_low/mid/high` must stay separate. Low cloud is optically thick and destroys PV output; high cirrus barely dents it. Collapsing them into a single `cloud_cover` discards most of the signal.

### Layer 2 — model matrix

```sql
region_id, run_ts_utc, valid_ts_utc, lead_hours, tech, capacity_mw
[ ... all Layer-1 features, union of solar + wind, NaN where not applicable ... ]
y_capacity_factor  FLOAT   -- power_mw / capacity_mw   ← TRAIN ON THIS
y_power_mw         FLOAT   -- reported, not trained
sample_weight      FLOAT   -- 0 where curtailed or unavailable
```

**Training on capacity factor rather than raw MW is what makes the platform multi-site.** A 50 MW plant and a 300 MW region produce the same target range, so one model generalises across both — and a region whose fleet grows during the training period does not confuse the model, because `monitored_capacity_mw` absorbs the growth.

### Layer 3 — decisions

```sql
region_id, valid_ts_local, issued_at, model_version
solar_p10_mw, solar_p50_mw, solar_p90_mw
wind_p10_mw,  wind_p50_mw,  wind_p90_mw
demand_forecast_mw, must_run_mw, storage_soc_mwh, storage_power_mw
net_load_mw       = demand − solar_p50 − wind_p50
net_load_p10_mw, net_load_p90_mw          -- uncertainty propagated
headroom_mw       = net_load − must_run
ramp_mw_per_h     = Δ net_load
flag              ENUM  -- OVER_GENERATION | STEEP_RAMP | DEFICIT_RISK | LOW_CONFIDENCE | NORMAL
recommended_action ENUM -- CURTAIL | CHARGE_BESS | DISCHARGE_BESS | COMMIT_BACKUP | HOLD
action_mwh, action_value_inr, action_confidence, rationale_text
```

---

## 4. Modelling

### 4.1 The ladder

Each rung must beat the rung below on a held-out window, or be skipped with a stated reason.

| Rung | Method | Role |
|---|---|---|
| 0 | **Persistence & climatology** — `ŷ(t+h) = y(t)`; smart persistence `= kt(t) × clearsky(t+h)` | The floor. Every reported number becomes a *skill score* against this. |
| 1 | **Pure physics** — pvlib ModelChain; manufacturer power curve | Second baseline, later a feature. Works with **zero history** — the cold-start answer. |
| 2 | **Gradient boosting** — LightGBM, `lead_hours` as a feature | The workhorse. Seconds to train, handles missing values natively, SHAP-explainable. |
| 3 | **Residual learning** — GBDT predicts `y − physics_pac`, not `y` | Usually the largest single accuracy gain for the least code. |
| 4 | **Quantile regression** — `objective='quantile'`, `alpha ∈ {0.1, 0.5, 0.9}` | Produces the interval the decision layer consumes. |
| 5 | **Conformal calibration** — split-conformal on a rolling window | Makes the interval *honest*, with a finite-sample coverage guarantee. |
| 6 | **Deep sequence models** — N-HiTS, TFT **[STRETCH]** | Benchmarked as a documented experiment, not assumed to win. |
| 7 | **Spatio-temporal GNN** on turbine graph **[STRETCH]** | Wake interaction modelling. Research territory. |

### 4.2 Why residual learning, specifically

```
                 ┌──────────────────┐
   NWP  ─────────▶│  Physics engine  │───▶ P_phys  ──┐
   site_master ──▶│  pvlib / curve   │               │
                 └──────────────────┘               ├──▶ ( + ) ──▶ P10/P50/P90
                 ┌──────────────────┐               │
   features ────▶│    LightGBM      │───▶  r̂  ──────┘
                 │  learns residual │
                 └──────────────────┘
```

The physics engine converts weather to power deterministically. The boosted model never has to learn that job — it sees only what physics got wrong: soiling, wake losses, sensor drift, local microclimate, archetype mis-weighting. Remove the physics block and the ML model must re-derive orbital mechanics from a few months of data.

Concretely, this gives:

- **Far lower data requirement.** The residual is small and structured; the raw signal is large and seasonal.
- **Cold start.** A region with no history still gets rung-1 physics output on day one.
- **Interpretability.** A SHAP plot on the residual model answers "why is this forecast low?" with *soiling and low cloud*, not with an opaque 300-feature attribution.
- **Graceful degradation.** If the ML model fails or drifts, the system falls back to physics rather than to nothing.

### 4.3 Uncertainty quantification

Two stages, because quantile regression alone is not calibrated.

**Stage 1 — quantile heads.** Three LightGBM models with pinball-loss objectives at α = 0.1, 0.5, 0.9. This produces intervals whose *width varies with conditions*, which is the property that matters: narrow at 03:00, wide on a broken-cloud afternoon. The features that drive the width are `lead_hours`, `nwp_model_disagreement`, `ensemble_spread`, `cloud_cover_low`, and (for wind) `dP_dv`.

**Stage 2 — split-conformal calibration.** On a rolling recent calibration window, compute nonconformity scores and adjust the interval so empirical coverage matches nominal coverage. This gives a distribution-free, finite-sample guarantee that holds regardless of whether the underlying model is well-specified.

We report **PICP** (prediction interval coverage probability) and **ACE** (average coverage error) and publish the reliability diagram. An 80 % band that covers 55 % of outcomes is not a conservative forecast — it is a false statement, and it will size reserves wrong.

Adaptive conformal inference **[STRETCH]** extends this to track distribution shift online, which matters for a fleet whose composition changes.

### 4.4 Fitting the archetype weights

The archetype shares in §2.3 start as assumptions. They are refined by a small constrained optimisation:

> Minimise RMSE between the capacity-weighted physics sum and observed regional generation, over the share vector `w`, subject to `Σw = 1`, `w ≥ 0`, on a clear-sky-only subset of days.

Restricting the fit to clear-sky days is deliberate: on those days the only remaining degrees of freedom are geometric, so the fit identifies orientation rather than absorbing cloud error. This converts three guessed numbers into three estimated numbers with a stated procedure — a meaningful difference under questioning.

### 4.5 Three data traps, handled explicitly

**Curtailed hours are censored labels.** When the grid orders a plant down, measured output is not what the weather could have produced. Training on those rows teaches the model to under-forecast exactly during the events the platform exists to predict. Handling: `sample_weight = 0` where `curtailed_mw > 0`. Where curtailment is not flagged, detect it heuristically — output flat-lining substantially below the physics estimate under good conditions — and mark it `qc_flag = CURTAILED`.

**Night rows are free accuracy.** Roughly half of a solar series is zeros, which any model predicts perfectly. Including them makes RMSE look excellent and mean nothing. We report **daylight-only** metrics for solar, and state this alongside every number.

**Random train/test splits leak.** Adjacent hours are near-identical. All evaluation uses **walk-forward splits with a gap**, never random `train_test_split`.

---

## 5. Decision engine

Two derived quantities per hour drive everything:

```
net_load  = demand_forecast − solar_p50 − wind_p50
headroom  = net_load − must_run_mw
```

`headroom < 0` means the grid physically cannot absorb the renewable output.

| Condition | Flag | Action | Sized by |
|---|---|---|---|
| `headroom < 0` and `soc < soc_max` | OVER_GENERATION | **Charge storage** | `min(−headroom, p_charge_max)` |
| `headroom < 0` and `soc ≥ soc_max` | OVER_GENERATION | **Curtail** — cheapest marginal asset first | `−headroom` |
| `ramp > ramp_limit` | STEEP_RAMP | **Pre-start flexible plant** or schedule discharge across the ramp | `∫ ramp` over window |
| `net_load_p90 > available_capacity` | DEFICIT_RISK | **Commit backup** — size to P90, not P50 | `p90 − available` |
| `above_cutout_flag` across the wind fleet | STORM_SHUTDOWN | **Reserve alert** — a whole fleet goes to zero in minutes | fleet capacity at risk |
| `p90 − p10 > band_limit` | LOW_CONFIDENCE | **Hold reserve** — widen margin, defer trades | band width × hours |
| otherwise | NORMAL | Hold; publish schedule | — |

Thresholds are per-region configuration, not code.

### 5.1 The virtual battery

Neither region publishes a usable grid-scale storage dispatch series, so storage is **declared in configuration and simulated**:

```yaml
energy_capacity_mwh   : 400        # swept to produce the sizing curve
power_rating_mw       : 100
round_trip_efficiency : 0.88       # applied on the charge leg
soc_min_frac          : 0.10
soc_max_frac          : 0.95
degradation_per_cycle : 0.00003    # makes cycling non-free
response_time_min     : 5          # can it catch a ramp, or only a plateau?
```

The simulator steps an energy balance forward across the 72-hour horizon, charging when `headroom < 0` and state of charge permits, discharging into steep ramps and evening peaks, respecting power and SoC bounds at each step.

Because capacity is a parameter, we sweep it — `energy_capacity_mwh` from 0 to 1000 — and plot **curtailed MWh avoided against battery size**. That single output converts a forecasting tool into an infrastructure recommendation, and it directly addresses a question the sector is actively asking: Ember's FY2025-26 analysis concludes India needs roughly **10 GWh of storage** to prevent current curtailment levels ([pv magazine, June 2026](https://www.pv-magazine.com/2026/06/17/india-needs-10-gwh-of-battery-storage-to-prevent-renewable-energy-curtailment/)).

### 5.2 Making a recommendation credible

Two properties separate a recommendation from an `if` statement:

- **Every action carries a number.** "Charge 42 MWh between 11:00 and 14:00, avoiding approximately ₹1.8 lakh of curtailment at today's imbalance price." The price assumption is stated alongside.
- **Every action carries its confidence.** An action triggered by a P50 sitting inside a wide P10–P90 band is a *suggestion*; one where the entire band clears the threshold is a *decision*. The interface distinguishes them.

**[STRETCH]** The optimiser upgrade is a linear program over the 72-hour horizon: minimise curtailed MWh plus backup fuel cost, subject to storage energy balance, charge/discharge limits, SoC bounds and ramp constraints. Approximately 40 lines of PuLP, and it reframes the contribution from "rules" to "dispatch optimisation".

---

## 6. Evaluation protocol

Defined before any model is trained. This is what makes an accuracy claim believable.

| Metric | Definition | Reports on |
|---|---|---|
| **nRMSE** | RMSE ÷ installed capacity | Point accuracy. Daylight-only for solar. |
| **nMAE** | MAE ÷ installed capacity | Typical-day accuracy |
| **MBE** | Mean bias, signed | Systematic over/under-forecasting — what traders care about most |
| **Skill score** | `1 − RMSE_model / RMSE_persistence` | **The headline number.** Scale-free and comparable. |
| **Pinball loss** | Quantile loss averaged over quantiles and horizons | Probabilistic quality (the GEFCom2014 metric) |
| **CRPS** | Continuous ranked probability score | Whole-distribution sharpness and calibration |
| **PICP / ACE** | Fraction of actuals inside P10–P90; target ≈ 0.80 | **Honesty check** |
| **Ramp hit rate** | Share of ramps > X %/h detected with ≥ Y h lead | The operationally meaningful metric |

**Splitting rules**

- **Walk-forward, never random.** Train months 1–6, test month 7; roll forward. Report mean and spread across folds.
- **Report per lead hour.** A single 72-hour-average nRMSE hides everything. The error-vs-lead-hour curve is among the most convincing charts available, because it demonstrates understanding of where skill originates.
- **Always three lines:** persistence, physics-only, full model. The gaps are the contribution, stated visually.
- **Match the information cutoff.** Elia's day-ahead forecast is issued at a specific time. Scoring our model on forecasts issued at a different moment makes the comparison meaningless.

---

## 7. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Ingestion | Python, `httpx`, APScheduler | Simple, adequate; no orchestration overhead at this scale |
| Validation | `pandera` schemas | Declarative contracts at every layer boundary |
| Storage | Parquet on local/object store, DuckDB for query | Columnar, fast, zero-ops. Postgres/TimescaleDB **[PLANNED]** for production. |
| Solar physics | `pvlib-python` | The reference implementation; peer-reviewed models |
| Wind physics | `windpowerlib` + custom IEC normalisation | Standard power-curve and shear handling |
| Models | LightGBM (quantile objective), scikit-learn | Fast, accurate, explainable |
| Calibration | MAPIE / custom split-conformal | Distribution-free coverage guarantee |
| Explainability | SHAP | Per-forecast attribution |
| Tracking | MLflow | Experiment and model-version registry |
| API | FastAPI + Pydantic | Typed contracts, automatic OpenAPI docs |
| Dashboard | React + Recharts/D3 | Fan charts, net-load stacks, event timelines |
| Deployment | Docker Compose → container host | Reproducible; scales without rewrite |

Everything in this stack is open source and free. There is no licence cost and no vendor dependency anywhere in the critical path.

---

## 8. Build sequence

Each phase ends at a demonstrable checkpoint.

| Phase | Build | Ends when you can say | Modules |
|---|---|---|---|
| **0 — Spine** | Elia + Open-Meteo connectors, Layer-0 schema, joined parquet, persistence baseline, error-vs-lead-hour plot | "Here is the baseline, measured properly." | M1, M2 |
| **1 — Physics** | pvlib archetype chain, wind shear + density + power curve, capacity-weighted weather | "We forecast a region with zero training data." | M3 |
| **2 — Skill** | LightGBM residual model, walk-forward backtest vs persistence **and** Elia's forecast, SHAP | "We beat persistence by N % and sit within M % of the TSO." | M4, M8 |
| **3 — Honesty** | Quantile heads, conformal calibration, PICP plot, reliability diagram | "Our 80 % band covers 80 %. Here is the evidence." | M5 |
| **4 — Decisions** | Ramp/over-generation scan, net-load balance with band propagated, sized actions | "On 14 March we would have flagged a 340 MW ramp six hours early." | M6, M7 |
| **5 — Surface** | FastAPI, operator dashboard | "An operator can act on this without reading a notebook." | M9, M10 |
| **6 — Depth** | LP optimiser, deep-model benchmark, Indian transfer, drift monitoring | "Here is what we tried that did *not* beat LightGBM, and why." | M4, M7, M8 |

Two things are set up on day one regardless: a **frozen evaluation harness** (one function, one held-out window, wired before the second model exists), and a **`--replay` mode** caching a week of forecast runs and all model artifacts, because live APIs fail during demonstrations far more often than models do.

---

## 9. What makes this approach technically defensible

1. **The horizon dictates the method, and we say so.** Recognising 24–72 h as an NWP post-processing problem rather than a time-series problem is the correct and non-obvious framing.
2. **Physics is used where physics is exact.** Not as decoration — as the deterministic component of a hybrid estimator, which reduces the data requirement and provides cold-start capability.
3. **Uncertainty is calibrated and verified, not asserted.** Conformal prediction gives a guarantee; the coverage plot gives the evidence.
4. **The benchmark is external and professional.** Elia's published operational forecast is a far harder and more meaningful bar than any internally chosen baseline.
5. **Known failure modes are handled by name.** Censored curtailment labels, night-row inflation, temporal leakage, and train/serve skew are each addressed explicitly rather than discovered later.
6. **The output is a decision with a number and a confidence attached.** Which is what the stated users actually require.

---

*Continues in [`06-system-architecture.md`](06-system-architecture.md) for component design and interface contracts.*
