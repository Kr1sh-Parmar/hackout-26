# Research and References

> Every source below was accessed and checked during this work. Access notes reflect what was actually observed, including where a portal blocked automated retrieval.

---

## 1. Datasets — primary

### 1.1 Elia Open Data Portal (Belgium) — **primary validation region**

The decisive property: generation, the operator's own forecast, **and** demand, on one portal at one 15-minute resolution with one join key.

| Dataset | Contents | Resolution |
|---|---|---|
| [`ods032`](https://opendata.elia.be/explore/dataset/ods032/) | PV production estimation and forecast — **historical**. Measured production, most-recent forecast, P10/P90, day-ahead (11:00 and 18:00 issues), week-ahead, monitored capacity, load factor, regional breakdown | 15 min |
| [`ods087`](https://opendata.elia.be/explore/dataset/ods087/) | PV production — **near-real-time**. Intraday, day-ahead and week-ahead forecasts | 15 min, updated quarter-hourly |
| [`ods031`](https://opendata.elia.be/explore/dataset/ods031/) | Wind production — historical. Split offshore / onshore, by region and grid connection type | 15 min |
| [`ods086`](https://opendata.elia.be/explore/dataset/ods086/) | Wind production — near-real-time | 15 min, updated quarter-hourly |
| [`ods001`](https://opendata.elia.be/explore/dataset/ods001/) | Measured and forecast **total load** on the Belgian grid — historical | 15 min |
| [`ods002`](https://opendata.elia.be/explore/dataset/ods002/) | Measured and forecast total load — near-real-time | 15 min |
| [`ods003`](https://opendata.elia.be/explore/dataset/ods003/) | Load on the Elia grid specifically | 15 min |

**Access:** OpenDataSoft Explore API v2.1 at `https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{id}/records`, plus CSV/JSON export. Free, no API key.
**Coverage:** 2013 → present.
**Note:** the portal's `robots.txt` blocks automated fetchers; use the documented API endpoint or the export button rather than a scraper.
**Why it matters:** the bundled day-ahead forecast is a free, professional, operational benchmark. Almost no other open portal provides one.

### 1.2 Open-Meteo — **weather backbone**

[Documentation](https://open-meteo.com/en/docs) · [ECMWF endpoint](https://open-meteo.com/en/docs/ecmwf-api) · [DWD ICON endpoint](https://open-meteo.com/en/docs/dwd-api) · [GFS/HRRR endpoint](https://open-meteo.com/en/docs/gfs-api) · [Ensemble endpoint](https://open-meteo.com/en/docs/ensemble-api) · [Satellite radiation](https://open-meteo.com/en/docs/satellite-radiation-api)

**Hourly variables used:**

| Group | Parameters |
|---|---|
| Solar radiation | `shortwave_radiation` (GHI), `direct_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `global_tilted_irradiance`, `terrestrial_radiation` |
| Cloud | `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` |
| Wind | `wind_speed_10m/80m/120m/180m`, `wind_direction_10m/80m/120m/180m`, `wind_gusts_10m` |
| Thermodynamic | `temperature_2m`, `dew_point_2m`, `apparent_temperature`, `relative_humidity_2m`, `pressure_msl`, `surface_pressure` |
| Other | `precipitation`, `rain`, `snowfall`, `visibility`, `is_day`, soil temperature/moisture at depth |

**Response structure:** JSON with `latitude`, `longitude`, `elevation` (from a 90 m DEM), `hourly_units`, and `hourly` containing a `time` array (ISO 8601) plus one array per variable.
**Horizon:** 7 days default, up to 16 with `&forecast_days=16`. `&past_days` up to 92; separate Historical API covers 1940 → present.
**Authentication:** none required for non-commercial use.
**Why it matters:** identical variable names historically and forward, which eliminates train/serve skew. Multiple NWP models through one interface satisfies the "one global + one regional" best practice.

### 1.3 India — modelled renewable production

**[Historical and modelled renewable energy production for India](https://research-information.bris.ac.uk/en/datasets/historical-and-modelled-renewable-energy-production-for-india/)** — University of Bristol
**Zenodo DOI:** [`10.5281/zenodo.7824872`](https://zenodo.org/records/7824872)

| Property | Value |
|---|---|
| Contents | Hourly wind and solar **capacity factors**; installed capacity by type and state; georeferenced installation locations (hydro, wind, solar); gridded capacity estimates; reported daily production |
| Temporal | 1979–2022 (modelled), 2012–2023 (reported production) |
| Spatial | State-level, plus 1° × 1° gridded |
| Formats | CSV, NetCDF (`.nc`), GeoJSON |
| Licence | Open |

**Critical caveat:** capacity factors are reanalysis-modelled, not metered. Suitable for demonstrating pipeline transfer; **not** suitable as ground truth for an accuracy claim.

### 1.4 Open Power System Data — cross-check region

[Data platform](https://data.open-power-system-data.org/time_series/) · [2017 README with naming convention](https://data.open-power-system-data.org/time_series/2017-03-06/README.md) · [source definitions](https://github.com/Open-Power-System-Data/time_series/blob/master/input/sources.yml)

**Naming convention:** `[COUNTRY]_[METRIC]_[TYPE]` — e.g. `DE_solar_generation_actual`, `DE_wind_onshore_generation_actual`, `DE_solar_capacity`, `DE_solar_profile` (share of capacity producing), `DE_50hertz_solar_forecast`, `DE_load_actual_entsoe_transparency`.
**Coverage:** 35 European countries, 2005 → 2020. Germany resolved to four TSO zones (50Hertz, Amprion, TenneT, TransnetBW).
**Resolution:** 60-minute file (comprehensive) and 15-minute file (selected renewables).

---

## 2. Datasets — benchmark and case study

### 2.1 SDWPF — Spatial Dynamic Wind Power Forecasting (KDD Cup 2022)

[arXiv:2208.04360](https://arxiv.org/abs/2208.04360) · [Nature Scientific Data](https://www.nature.com/articles/s41597-024-03427-5) · Baidu / Longyuan Power

| Property | Value |
|---|---|
| Turbines | 134 |
| Records | 4,727,520 |
| Span | ~245 days |
| Interval | 10 minutes |
| Spatial | Relative x/y turbine coordinates in metres |

**Columns:** `TurbID`, `Day`, `Tmstamp`, `Wspd` (m/s, anemometer), `Wdir` (°, wind direction relative to nacelle), `Etmp` (°C, environment), `Itmp` (°C, nacelle interior), `Ndir` (°, nacelle yaw), `Pab1`/`Pab2`/`Pab3` (°, blade pitch), `Prtv` (kW, reactive), `Patv` (kW, active — **target**).

The turbine coordinates make wake modelling and graph neural networks tractable. Used here as an asset-level case study and the route into the spatio-temporal stretch goal.

### 2.2 GEFCom2014 — the academic benchmark

Hong, Pinson, Fan, Zareipour, Troccoli & Hyndman (2016), *Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond*, **International Journal of Forecasting**. [PDF](https://robjhyndman.com/papers/gefcom2014.pdf)

| Track | Setup | Inputs |
|---|---|---|
| **Wind** | 10 wind farms in Australia, rolling 24-h ahead, hourly | ECMWF u and v wind components at 10 m and 100 m |
| **Solar** | 3 solar plants, rolling 24-h ahead, hourly | 12 ECMWF variables: total column liquid water and ice water (kg m⁻²), surface pressure (Pa), relative humidity at 1000 mbar (%), total cloud cover (0–1), 10 m U/V wind (m s⁻¹), 2 m temperature (K), surface thermal radiation downward, surface solar radiation down, top net solar radiation (J m⁻²), total precipitation (m) |

**Output format:** 99 quantiles per target period.
**Evaluation:** pinball loss —

```
L(q_a, y) = (1 − a/100)(q_a − y)   if y < q_a
          = (a/100)(y − q_a)       if y ≥ q_a
```

averaged over quantiles, horizons and zones. Published leaderboards provide comparable numbers.

Also relevant: Nagy, Barta, Kazi, Borbély & Simon (2016), *GEFCom2014: Probabilistic solar and wind power forecasting using a generalized additive tree ensemble approach*, IJF — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001521). **Tree ensembles won this competition**, which is direct empirical support for the LightGBM-centred design rather than a deep-learning-first design.

### 2.3 NREL NSRDB — historical solar resource

[HPC dataset description](https://github.com/NREL/hsds-examples/blob/master/datasets/NSRDB.md) · [Developer API](https://developer.nrel.gov/docs/solar/nsrdb/) · [SUNY India endpoint](https://developer.nrel.gov/docs/solar/nsrdb/suny-india-data-download/)

**23 fields** including GHI, DNI, DHI, clear-sky GHI/DNI/DHI, temperature, dew point, pressure, relative humidity, wind speed and direction, precipitable water, cloud type, solar zenith angle, surface albedo, aerosol optical depth, single-scatter albedo, asymmetry parameter, Ångström exponent, cloud optical depth, cloud effective radius, ozone, fill flag.

| Product | Resolution | Coverage |
|---|---|---|
| US primary | 30 min, ~4 km (0.038°) | 1998–2022 |
| SUNY India | Hourly, 10 km | 2000–2014 |

**Rate limits (SUNY India):** 10,000 CSV requests/24 h, max 1/second; 2,000 requests/24 h for other formats at 1 per 2 seconds; 20 concurrent in-flight requests.
**Role here:** historical solar truth for training and validation only — it is reanalysis and cannot be obtained forward, so using it at serving time would create train/serve skew.

### 2.4 Kaggle — asset-level case studies

- [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (anikannal) — two Indian plants, 34 days, 15-minute. Generation file: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY` (inverter), `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`. Separate weather-sensor file with ambient temperature, module temperature and irradiation.
- [Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) (berkerisen) — single turbine in Turkey, 10-minute, one year. Columns: `Date/Time`, `LV ActivePower (kW)`, `Wind Speed (m/s)`, `Theoretical_Power_Curve (KWh)`, `Wind Direction (°)`. Useful specifically for power-curve fitting demonstrations.

### 2.5 Additional

- **China State Grid Renewable Energy Generation Forecasting Competition** — [Nature Scientific Data](https://www.nature.com/articles/s41597-022-01696-6). Solar and wind farm output with paired NWP, 15-minute. The closest public analogue to metered Asian plant data.
- **[OpenWindSCADA](https://github.com/sltzgs/OpenWindSCADA)** — curated index of open wind turbine datasets and accompanying code.
- **[awesome-industrial-datasets](https://github.com/jonathanwvd/awesome-industrial-datasets)** — broader industrial time-series index.

---

## 3. Methodology literature

### 3.1 Forecasting practice and benchmarks

**GET.transform (2024), *International Best Practices in Solar and Wind Power Forecasting*.** [PDF](https://www.get-transform.eu/wp-content/uploads/2024/01/GET.transform-Brief_VRE-Forecasting-Solar-Wind.pdf)

The most operationally useful single reference located. Key findings used throughout this documentation:

| Finding | Value |
|---|---|
| Day-ahead RMSE, regional aggregates | 10–20 % of installed capacity |
| Single wind farm MAE | 7–19 %, terrain-dependent |
| Chile, plant operators | ~13 % MAE (2017) → 9–10 % (2019) |
| Chile, professional providers | ~9 % MAE |
| Mexico, wind: operators vs professionals | 17 % vs 13 % MAE |
| Mexico, solar: operators vs professionals | 9 % vs 6 % MAE |
| Error growth with horizon | "Nearly linear" |
| Centralised forecasting adoption | ~80 % of world TSOs |
| Optimal NWP combination | 5–6 models; minimum one global + one regional |
| Operational horizons | 0–6 h (intraday), 6–48 h (day-ahead), 2–20 d (medium) |
| NREL, Western U.S. | 20 % forecast improvement at 24 % wind penetration → USD 195 M/yr operating savings |
| Berkeley Lab | NWP forecasts cost USD 1.00/MWh in errors vs USD 1.50/MWh for persistence |
| CENACE, Mexico | Upscaled forecasting would correct 622 MW/hr deviation by 2024 |

**Zhang, Florita, Hodge et al., *A suite of metrics for assessing the performance of solar power forecasting*, Solar Energy.** [ScienceDirect](https://www.sciencedirect.com/science/article/am/pii/S0038092X14005027) — the standard reference for solar forecast metric selection.

### 3.2 Probabilistic forecasting and calibration

- **[Probabilistic day-ahead forecasting of system-level renewable energy and electricity demand](https://www.nature.com/articles/s41467-026-69015-w)** — Nature Communications. Directly analogous problem framing at system level.
- **[Climate-Invariant Conformal Prediction Intervals for Multi-Horizon Solar and Wind Forecasting](https://arxiv.org/html/2607.11470v1)** — arXiv. Conformal prediction applied to exactly this multi-horizon setting.
- **[A non-parametric adaptive conformal inference based probabilistic hour-ahead solar PV power forecasting method](https://www.nature.com/articles/s41598-026-40911-x)** — Scientific Reports. Adaptive conformal inference for PV.
- **[Model-Agnostic, Probabilistic, Hour-Ahead Solar PV Forecasting Using Adaptive Conformal Inference](https://doi.org/10.3390/en19061495)** — Energies.
- **[Probabilistic Photovoltaic Power Forecasting with Reliable Uncertainty Quantification via Multi-Scale Temporal–Spatial Attention and Conformalized Quantile Regression](https://www.mdpi.com/2071-1050/18/2/739)** — Sustainability. Conformalised quantile regression, the exact Stage-1 + Stage-2 pattern used here.
- **Feng et al., *Probabilistic Short-term Wind Forecasting Based on Pinball Loss Optimization*, PMAPS 2018.** [PDF](https://fengcong1992.github.io/files/publication/Feng_2018_PMAPS.pdf)
- **[Essential Guide to CRPS for Forecasting](https://towardsdatascience.com/essential-guide-to-continuous-ranked-probability-score-crps-for-forecasting-ac0a55dcb30d/)** — accessible CRPS explanation.

### 3.3 Model architectures

- **[MATNet: Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting](https://arxiv.org/html/2306.10356v2)** — arXiv.
- **[DG-TFT-CQR: A Dynamic Graph–Temporal Fusion Transformer with Conformalized Quantile Regression for Wind Power Forecasting](https://www.mdpi.com/2571-9394/8/4/55)** — Forecasting.
- **[Solar Irradiance Forecasting Using Temporal Fusion Transformers](https://onlinelibrary.wiley.com/doi/full/10.1155/er/3534500)** — International Journal of Energy Research, 2025.
- **[Day-ahead regional solar power forecasting with hierarchical temporal convolutional neural networks](https://arxiv.org/html/2403.01653v1)** — arXiv. Directly relevant: *regional* day-ahead, the same framing as this work.
- **[Efficient Deterministic Renewable Energy Forecasting](https://arxiv.org/html/2404.17276v1)** — arXiv.
- **[Spatial-Temporal Graph Neural Network for Wind Power Forecasting](https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_9863.pdf)** — KDD Cup 2022 workshop, the stretch-goal reference for the SDWPF turbine graph.

### 3.4 India-specific forecasting

**[Development of a day-ahead solar power forecasting model chain for a 250 MW PV park in India](https://link.springer.com/article/10.1007/s40095-023-00560-6)** — International Journal of Energy and Environmental Engineering.

Closely parallel to our approach. Method chain: ECMWF and NCMRWF NWP → Lorenz polynomial bias correction → GHI-to-GTI transposition (Chandrasekaran and Klucher models) → AC power. Findings: interpolated ECMWF GHI showed +0.46 % bias and NCMRWF +1.71 % (both over-estimating); after Lorenz bias correction, ECMWF reduced to −0.001 % with 25 days of training and NCMRWF to −0.08 % with 20 days. A **linear combination of ECMWF- and NCMRWF-derived AC forecasts performed best**, outperforming a reference convex combination of climatology and persistence.

Two points transfer directly: NWP irradiance over India carries a systematic positive bias that a short training window corrects, and **multi-model combination wins** — supporting our three-model ingest.

---

## 4. Standards and physical models

| Standard / model | Application |
|---|---|
| **IEC 61400-12** | Wind turbine power performance measurement; air-density normalisation of wind speed |
| **Ineichen–Perez clear-sky model** | Clear-sky GHI/DNI/DHI — the denominator of the clear-sky index |
| **SAPM (Sandia Array Performance Model)** | Cell temperature from POA irradiance, ambient temperature and wind speed |
| **Perez / Hay-Davies transposition** | GHI → plane-of-array irradiance for tilted and tracking arrays |
| **NREL SPA (Solar Position Algorithm)** | Solar zenith and azimuth |
| **Wind profile power law** | Shear extrapolation from 100 m to hub height via `alpha` |
| **CEA CO₂ Baseline Database** | Indian grid emission factor for avoided-emissions calculations |

---

## 5. Regulatory sources (India)

| Source | Content |
|---|---|
| [CERC notifies phased X-factor reduction, tightens deviation bands from April 2026](https://www.energetica-india.net/news/cerc-notifies-phased-x-factor-reduction-for-wind-and-solar-tightens-deviation-bands-from-april-2026) — Energetica India | Deviation bands: solar/hybrid ± 10 % → **± 5 %**, wind ± 15 % → **± 10 %**, effective 1 April 2026. X factor phased 100 % → 0 % by FY 2031 (solar: 100/90/75/55/30/0; wind: 100/95/85/65/35/0). |
| [CERC Announces Phased DSM Reform For Wind And Solar](https://solarquarter.com/2026/04/01/cerc-announces-phased-dsm-reform-for-wind-and-solar-tightens-deviation-norms-by-2031/) — SolarQuarter | Corroborating coverage |
| [CERC (Deviation Settlement and Related Matters) Regulations, 2024 — Draft](https://cer.iitk.ac.in/odf_assets/upload_files/blog_cer_iitk_CERC_DSM_and_related_matters_Regulations_2024.pdf) — CER, IIT Kanpur | Primary regulatory text |
| [Karnataka High Court Stays CERC's Revised Deviation Settlement Mechanism](https://www.mercomindia.com/karnataka-high-court-stays-cercs-revised-deviation-settlement-mechanism) — Mercom | **Legal challenge — must be cited alongside the regulation** |
| [Rigid Deviation Settlement Limits Could Result in Higher Renewable Tariffs](https://www.mercomindia.com/deviation-settlement-higher-tariffs) — Mercom | Industry counter-argument |
| [CERC DSM To Hurt Renewable Players, Says NSEFI](https://www.saurenergy.com/solar-energy-news/cerc-deviation-settlement-mechanism-to-hurt-renewable-players-says-nsefi) — SaurEnergy | Industry association position |
| [KERC Forecasting, Scheduling, DSM Regulations, 2026](https://www.eqmagpro.com/kerc-forecasting-scheduling-deviation-settlement-mechanism-and-related-matters-for-sellers-of-wind-solar-and-ws-hybrid-generation-sources-regulations-2026-eq/) — EQ Magazine | State-level parallel regulation (Karnataka) |

---

## 6. Market and impact data (India)

| Source | Key figures |
|---|---|
| [India needs 10 GWh of battery storage to prevent renewable energy curtailment](https://www.pv-magazine.com/2026/06/17/india-needs-10-gwh-of-battery-storage-to-prevent-renewable-energy-curtailment/) — pv magazine, 17 June 2026, reporting Ember analysis by Neshwin Rodrigues | **2.1 TWh curtailed FY 2025-26 = 1.3 % of renewable generation**; peak-hour curtailment reached **4 %** by April 2026; **~10 GWh storage** needed; coal cycles to ~**55 %** of rated capacity at midday |
| [India lost 300 million units of renewable energy owing to transmission constraints in Q1 2026](https://ember-energy.org/latest-updates/india-lost-300-million-units-of-renewable-energy-owing-to-transmission-constraints-in-q1-2026/) — Ember | **300 GWh** from transmission constraints of **~470 GWh** total; Northern **178 GWh**, Western **122 GWh**, Southern **0 GWh**; **34 GWh lost on 30 March 2026**; India met only **80 %** of transmission targets over five years; **1 in 4** major schemes > 1 year behind |
| [Transmission gaps are beginning to constrain India's renewables integration](https://ember-energy.org/latest-insights/transmission-gaps-are-beginning-to-constrain-indias-rapid-renewables-integration/) — Ember | Structural analysis |
| [India curtails 2.3 TWh of solar power in 2025 as grid flexibility falls short](https://www.downtoearth.org.in/energy/india-curtails-23-twh-of-solar-power-in-2025-as-grid-flexibility-falls-short-ember-report) — Down To Earth | Calendar-year figure |
| [India Electricity Data Explorer](https://ember-energy.org/data/india-electricity-data-explorer/) — Ember | Interactive generation and capacity data |
| [Renewable Generation Report](https://cea.nic.in/renewable-generation-report/?lang=en) — Central Electricity Authority | Official Indian generation statistics |
| [National Power Portal](https://npp.gov.in/publishedReports) — Government of India | Published sector reports |
| [India Climate & Energy Dashboard](https://iced.niti.gov.in/) — NITI Aayog | Capacity and generation mix |
| [Open Government Data — Renewable Energy](https://www.data.gov.in/dataset-group-name/Renewable%20Energy) | Indian open data catalogue |

> **Note on the two curtailment figures.** 2.1 TWh is the fiscal-year FY 2025-26 number; 2.3 TWh is a calendar-2025 solar figure from a separate Ember publication. They are different periods and different scopes — cite the one that matches the claim being made, and do not conflate them.

---

## 7. Software and tooling

| Tool | Role | Reference |
|---|---|---|
| **pvlib-python** | Solar position, clear-sky models, transposition, cell temperature, ModelChain | [Docs](https://pvlib-python.readthedocs.io/en/stable/) · [irradiance module](https://pvlib-python.readthedocs.io/en/latest/_modules/pvlib/irradiance.html) · [ModelChain](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.html) |
| **windpowerlib** | Wind power curves, shear extrapolation, density correction | PyPI / GitHub |
| **LightGBM** | Gradient boosting; `objective='quantile'` for quantile heads | Microsoft |
| **MAPIE** | Conformal prediction intervals, scikit-learn compatible | GitHub |
| **SHAP** | Per-forecast feature attribution | GitHub |
| **MLflow** | Experiment tracking, model registry | Databricks |
| **pandera** | Declarative dataframe schema validation | GitHub |
| **DuckDB** | In-process columnar query over Parquet | duckdb.org |
| **FastAPI** | Typed API with automatic OpenAPI documentation | fastapi.tiangolo.com |
| **PuLP** | Linear programming for the dispatch optimiser **[STRETCH]** | GitHub |

**Worked references:**
- [Forecasting Wind Power Generation Using XGBoost](https://mo-saif.github.io/Wind-Power-Forecasting/Wind%20Power%20Forecasting%20Using%20XGBoost.html)
- [Predicting wind and solar generation from weather data using Machine Learning](https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e) — uses OPSD data
- [pvlib Forecasting module documentation](https://pvlib-python.readthedocs.io/en/v0.9.0/forecasts.html)
- [Open Climate Fix — Quartz Solar Forecast](https://huggingface.co/openclimatefix/open-source-quartz-solar-forecast) — production open-source solar forecasting reference

---

## 8. How the evidence maps onto our design decisions

| Design decision | Supporting evidence |
|---|---|
| Treat 24–72 h as NWP post-processing, not time-series | GET.transform: error grows nearly linearly with horizon; operational horizons are defined by information availability |
| Multi-model NWP ingest (ECMWF + ICON + GFS) | GET.transform: 5–6 models optimal, minimum one global + one regional. India PV park study: linear combination of ECMWF + NCMRWF performed best. |
| Gradient boosting as the core model | GEFCom2014 won by a generalized additive tree ensemble; tree methods remain competitive at this scale |
| Physics baseline + learned residual | pvlib/IEC models are peer-reviewed and exact; the India PV park study uses the same physics-chain-then-correct structure |
| Bias correction on recent forecasts | India PV park study: Lorenz polynomial reduced ECMWF bias from +0.46 % to −0.001 % with 25 days of training |
| Conformal calibration for intervals | Four 2025–2026 papers applying conformal inference to exactly this problem |
| Regional / centralised architecture | GET.transform: ~80 % of world TSOs use centralised forecasting |
| Capacity-factor target, not raw MW | Standard normalisation; also how GEFCom2014 published its ground truth (0–1 normalised) |
| Storage sizing sweep as an output | Ember concludes India needs ~10 GWh — the sector is actively asking this question |
| Deviation penalties as the commercial driver | CERC tightened bands to ± 5 % / ± 10 % effective 1 April 2026 |

---

## 9. Citation practice used in this documentation set

1. Every external number carries its source and date inline.
2. Every derived number shows its arithmetic and marks its assumptions **[ASSUMED]**.
3. Where a source is contested — the DSM regulations — the challenge is cited alongside the regulation.
4. Where two figures could be confused — 2.1 TWh vs 2.3 TWh curtailment — the difference is stated explicitly.
5. Access limitations observed during research are recorded, so a reader reproducing the work does not repeat a dead end.

---

*Last verified: September 2026.*
