# Feasibility and Viability

> Three questions: **Can the data be obtained? Can the accuracy be achieved? Can the thing be operated and sustained?**

---

## 1. Data feasibility

The most common reason a forecasting project fails is not modelling — it is discovering that the required data does not exist in usable form. Every input is therefore resolved to a named, open, accessible source before any modelling begins.

| Requirement | Source | Access | Cost | Licence | Status |
|---|---|---|---|---|---|
| Regional solar generation, measured | Elia `ods032` / `ods087` | Open Data Portal, REST + CSV export | Free | Open | ✅ Confirmed |
| Regional wind generation, measured | Elia `ods031` / `ods086` | Same | Free | Open | ✅ Confirmed |
| **Operator's own forecast (benchmark)** | Bundled in the above — day-ahead, week-ahead, P10/P90 | Same | Free | Open | ✅ Confirmed |
| Regional demand, measured + forecast | Elia `ods001` / `ods002` / `ods003` | Same | Free | Open | ✅ Confirmed |
| Monitored capacity (fleet growth) | Bundled in generation datasets | Same | Free | Open | ✅ Confirmed |
| Weather forecast, 16 days ahead | Open-Meteo Forecast API | REST, **no API key** for non-commercial | Free | Open | ✅ Confirmed |
| Weather history, 1940 → present | Open-Meteo Historical API | Same | Free | Open | ✅ Confirmed |
| Multi-model NWP (ECMWF, ICON, GFS, GEM) | Open-Meteo model-specific endpoints | Same | Free | Open | ✅ Confirmed |
| Indian solar/wind capacity factors, hourly | Zenodo `10.5281/zenodo.7824872` (Univ. of Bristol) | Direct download, CSV + NetCDF + GeoJSON | Free | Open | ✅ Confirmed |
| Indian installed capacity by state | CEA / MNRE published reports | Web, PDF | Free | Public | ⚠️ Manual extraction |
| Cross-check region (Germany, 4 TSO zones) | Open Power System Data | Direct CSV download | Free | Open | ✅ Confirmed |
| Asset-level case study (wind) | SDWPF, KDD Cup 2022 — 134 turbines, x/y layout | Academic release | Free | Research | ✅ Confirmed |
| Asset-level case study (solar, India) | Kaggle — 2 Indian plants, inverter level | Kaggle account | Free | Open | ✅ Confirmed |

**Nothing in the critical path requires a paid subscription, a commercial API key, an NDA, or a data-sharing agreement.** This is a deliberate constraint, not a happy accident — it means the system is reproducible by anyone, including the judging panel.

### 1.1 The one material data gap, stated plainly

India does not publish a public, open, sub-hourly, *metered* regional renewable generation series. Grid-India and CEA publish primarily daily aggregates, largely as PDFs.

**How we handle it rather than hide it:**

| Approach | What it gives | What it does not give |
|---|---|---|
| Validate on Belgium (metered, 15-min, with operator benchmark) | A defensible accuracy claim against professional ground truth | Indian relevance |
| Transfer to India (Bristol hourly capacity factors + CEA capacity) | Demonstration that the pipeline moves to a new grid on open data alone | An accuracy claim — the Indian data is reanalysis-modelled, not metered |

Scoring a model against modelled data partly scores the model that produced it. **Phase B therefore validates transferability, not accuracy.** This is stated in the pitch in one sentence, before anyone asks. The Belgian phase carries the accuracy claim; the Indian phase carries the relevance claim. Neither is asked to do the other's job.

---

## 2. Accuracy feasibility

The question that decides whether the project is credible: **are the targets we are setting actually attainable?**

### 2.1 Published benchmarks

From [GET.transform's review of international forecasting practice (2024)](https://www.get-transform.eu/wp-content/uploads/2024/01/GET.transform-Brief_VRE-Forecasting-Solar-Wind.pdf):

| Context | Reported accuracy |
|---|---|
| Day-ahead forecasts, **regional aggregates** | **10–20 % RMSE** (normalised to installed capacity) |
| Single wind farm MAE | 7–19 %, depending on terrain complexity |
| Chile — plant operators | ~13 % MAE (2017) improving to 9–10 % (2019) |
| Chile — professional external providers | ~9 % MAE |
| Mexico — wind: operators vs professionals | 17 % vs 13 % MAE |
| Mexico — solar: operators vs professionals | 9 % vs 6 % MAE |

Two structural findings from the same review inform our design directly:

- **"Forecast error increases nearly linearly with the prediction horizon."** This validates treating `lead_hours` as a first-class feature and reporting metrics per lead hour rather than as a single average.
- **Combination forecasts beat single models**, with error minimised around five to six NWP models and a recommended minimum of one global plus one regional model. Our three-model Open-Meteo ingest (ECMWF + ICON + GFS) sits at the practical floor of this recommendation.

### 2.2 Our targets

| Metric | Target | Basis |
|---|---|---|
| Regional solar, day-ahead nRMSE (daylight-only) | **8–13 %** | Within the published 10–20 % regional band; better end achievable because regional aggregation cancels independent cloud noise |
| Regional wind, day-ahead nRMSE | **10–15 %** | Consistent with the 7–19 % single-farm MAE band, aggregated |
| Skill score vs persistence | **> 0.40** | Persistence is very weak at 24–72 h; this should be comfortable |
| Gap to Elia's published day-ahead forecast | **Within 10–25 % relative** | Elia is a professional operational system with proprietary inputs; parity is not the goal |
| PICP for the nominal 80 % band | **0.78 – 0.82** | Conformal calibration provides a finite-sample guarantee |
| Ramp detection (> 20 %/h) with ≥ 6 h lead | **> 70 %** **[ASSUMED]** | No published benchmark located; to be measured and reported as observed |

These are deliberately **conservative and falsifiable**. A target stated inside a published range, with the range cited, is far more defensible than an impressive round number with no provenance. Where no benchmark exists, the target is marked as an assumption to be measured rather than claimed.

### 2.3 Why these targets are attainable with this approach

| Factor | Effect |
|---|---|
| Regional aggregation | Independent cloud noise partially cancels; regional nRMSE is structurally better than single-plant |
| Physics baseline removes ~90 % of variance | The ML model has a small, structured residual to learn rather than a large seasonal signal |
| Multi-model NWP | Best-practice combination approach, at the recommended minimum configuration |
| Bias correction on recent forecasts | Corrects the systematic NWP bias that dominates sub-problem 1 |
| 12+ months of 15-minute Elia history | Ample for gradient boosting on ~60 features; more than sufficient for the residual |
| Elia's own forecast as a bundled benchmark | We can measure the gap continuously instead of guessing at it |

---

## 3. Technical feasibility

### 3.1 Compute budget

| Workload | Resource | Time |
|---|---|---|
| Historical weather backfill (10 grid points × 3 models × 2 years, hourly) | API calls | ~20 min, one-off |
| Feature engineering, full history | 1 CPU core | < 2 min |
| LightGBM training, one quantile head | 1 CPU core | 10–40 s |
| Full model set (2 tech × 3 quantiles) | 1 CPU core | < 5 min |
| Conformal calibration | 1 CPU core | < 10 s |
| Walk-forward backtest, 12 folds | 1 CPU core | ~10 min |
| Operational inference, one region, 72 h | 1 CPU core | < 1 s |

**No GPU is required anywhere in the critical path.** The entire system trains and serves on a laptop. Deep sequence models **[STRETCH]** would benefit from a GPU, which is precisely why they sit in phase 6 rather than the core design.

This matters beyond convenience: a system that runs on commodity hardware can be deployed at a state load dispatch centre without a procurement cycle.

### 3.2 Dependency risk

| Dependency | Risk | Mitigation |
|---|---|---|
| Open-Meteo availability | API down or rate-limited during demonstration | `--replay` mode serves a cached week of forecast runs; all model artifacts on disk |
| Open-Meteo non-commercial terms | Commercial deployment would require their paid tier or a direct ECMWF/IMD feed | Adapter pattern — the weather source is one interface with multiple implementations |
| Elia portal schema change | Ingestion breaks | `pandera` schema validation fails loudly at the boundary rather than corrupting downstream data |
| `pvlib` / `windpowerlib` | Mature, widely used, actively maintained | Pinned versions; both are peer-reviewed reference implementations |

### 3.3 Engineering complexity, honestly assessed

| Component | Complexity | Note |
|---|---|---|
| Ingestion + validation | Low | REST calls and schema checks |
| Physics feature engineering | **Medium** | `pvlib` has a learning curve; archetype weighting is genuinely novel work |
| Residual GBDT + quantile heads | Low | Well-trodden; LightGBM does the heavy lifting |
| Conformal calibration | Low–Medium | Conceptually subtle, ~50 lines to implement |
| Decision engine (rules) | Low | Arithmetic and thresholds |
| Decision engine (LP) **[STRETCH]** | Medium | ~40 lines of PuLP, but constraint formulation needs care |
| Dashboard | Medium | Fan charts and net-load stacks are non-trivial to render well |
| Indian transfer | Low | By design — only `site_master`, grid points and truth source change |

The riskiest component is the **archetype weighting**, because it is the part with no textbook procedure. This is mitigated by the clear-sky fitting method in §4.4 of the technical approach, which converts guessed weights into estimated weights with a stated procedure.

---

## 4. Regulatory and market viability

### 4.1 The regulatory timing is unusually favourable

India's Central Electricity Regulatory Commission has **tightened deviation settlement norms with effect from 1 April 2026** ([Energetica India, 2026](https://www.energetica-india.net/news/cerc-notifies-phased-x-factor-reduction-for-wind-and-solar-tightens-deviation-bands-from-april-2026)):

| Parameter | Before | From 1 April 2026 |
|---|---|---|
| Permitted deviation band — solar and wind-solar hybrid | ± 10 % | **± 5 %** |
| Permitted deviation band — wind | ± 15 % | **± 10 %** |

And the "X factor" — which determines how deviations are computed relative to available capacity versus scheduled generation — is phased down to zero:

| Financial year | Solar / hybrid | Wind |
|---|---|---|
| 2026-27 | 100 % | 100 % |
| 2027-28 | 90 % | 95 % |
| 2028-29 | 75 % | 85 % |
| 2029-30 | 55 % | 65 % |
| 2030-31 | 30 % | 35 % |
| 2031 onwards | **0 %** | **0 %** |

**What this means for viability:** the permitted forecast error for a solar generator has been halved, and the tolerance mechanism is being withdrawn entirely over five years. Forecast accuracy moved from a nice-to-have to a direct, escalating line item on every renewable generator's P&L — in the current financial year.

State commissions are moving in parallel; Karnataka notified its own Forecasting, Scheduling and Deviation Settlement regulations for wind, solar and hybrid sellers in 2026.

> **Stated honestly:** these regulations face active legal challenge. The Karnataka High Court has stayed CERC's revised DSM, and Delhi High Court proceedings are pending. Industry bodies including NSEFI have argued the bands are too rigid and may raise renewable tariffs. The *direction* of travel — tighter tolerances, greater financial consequence for forecast error — is nonetheless consistent across CERC and multiple state commissions, and that direction is what makes the product viable regardless of the specific numbers that survive litigation.

### 4.2 Market structure

The same international review finds that **approximately 80 % of the world's transmission system operators rely primarily on centralised power forecasts**, including all European countries and the U.S. and Canadian ISOs. The stated rationale: better consistency and accuracy, neutral results, cost-efficient coverage of small distributed generation, and easier quality control.

This validates the architecture directly. A **regional, centralised, multi-asset** forecasting platform is the configuration the sector has converged on — not a per-plant tool.

### 4.3 Who pays, and for what

| Customer | Problem they have | What they buy |
|---|---|---|
| **State Load Dispatch Centres** | Must schedule conventional generation against uncertain renewable output | Regional forecast + net-load outlook + ramp alerts |
| **Renewable generators / IPPs** | DSM penalties on deviation, now at ± 5 % | Plant-level forecast + scheduling support |
| **Distribution companies** | Procurement planning, RPO compliance | Day-ahead demand-net-renewables outlook |
| **Energy traders** | Position-taking in day-ahead and real-time markets | Probabilistic forecast; P10/P90 drives position sizing |
| **Storage operators** | Arbitrage and dispatch timing | Charge/discharge schedule optimised against forecast |
| **Regulators / planners** | Where to site storage and transmission | Curtailment analytics and the battery sizing curve |

The **probabilistic output is the commercial differentiator** for traders and storage operators specifically. Position sizing and dispatch timing are decisions under uncertainty; a point forecast cannot inform them at all.

---

## 5. Operational viability

### 5.1 What running this actually involves

| Activity | Frequency | Effort |
|---|---|---|
| Weather ingestion | 4× daily (per NWP run cycle) | Automated |
| Forecast generation | 4× daily | Automated, < 1 s per region |
| Actuals ingestion | Every 15 min | Automated |
| Model retraining | Nightly | Automated, < 5 min |
| Conformal recalibration | Nightly | Automated, < 10 s |
| Drift review | Weekly | ~15 min human |
| Archetype re-fit | Quarterly, or on fleet change | ~1 hour human |

Steady-state operation is effectively unattended. The human effort is in review, not operation.

### 5.2 Scaling

| Dimension | Current | Scaling path |
|---|---|---|
| Regions | 1–5 | Linear; a region is a config entry plus grid points |
| Technologies | Solar, wind | Adding hydro or biomass is a new physics module behind the same interface |
| Resolution | Hourly / 15 min | 5-minute needs a higher-frequency weather source **[STRETCH]** |
| Horizon | 24–72 h | 0–6 h nowcasting requires satellite ingestion — a separate model, same platform |
| Assets | Regional fleets | Plant-level is the *same* pipeline with a narrower `site_master` |

The architecture does not need rewriting for any of these. That is the direct payoff of the capacity-factor target and the archetype abstraction.

### 5.3 Cost to operate

| Item | Cost |
|---|---|
| Weather data | ₹0 (Open-Meteo non-commercial) |
| Generation data | ₹0 (Elia open portal) |
| Compute | One small VM — approximately ₹2,000–4,000/month **[ASSUMED]** |
| Storage | Parquet; a few GB per region-year — negligible |
| Software licences | ₹0 — entire stack is open source |

For a commercial deployment, the material change is a paid weather feed (commercial Open-Meteo tier, or a direct ECMWF/IMD contract). That is a single adapter implementation, deliberately isolated behind one interface.

---

## 6. Risks to viability, and responses

| Risk | Likelihood | Impact | Response |
|---|---|---|---|
| Cannot match Elia's forecast | **High** | Low | Never claimed. The stated target is "beat persistence, close the gap to the TSO" — and closing to within a stated margin using only open data is itself the result. |
| Indian data too coarse for a credible demo | Medium | Medium | Two-region design exists precisely for this. Belgium carries accuracy; India carries transfer. |
| Archetype weights do not converge | Medium | Medium | Fall back to a single average archetype; accuracy degrades but the pipeline stands. Report the degradation. |
| Conformal intervals too wide to be useful | Low | Medium | Wide-but-honest is more useful than narrow-and-wrong. Report the width; it is a property of the problem, not a bug. |
| Open-Meteo terms block commercial use | High (for commercialisation) | Low | Adapter pattern; swapping to a paid feed is one implementation. |
| DSM regulations struck down in litigation | Medium | Medium | The regulatory *direction* is consistent across CERC and multiple state commissions. The curtailment problem exists regardless of settlement rules. |
| Curtailment not flagged in source data | **High** | Medium | Heuristic detection (§4.5 of technical approach) plus explicit `sample_weight = 0`. Stated as a limitation. |

---

## 7. Summary judgement

| Dimension | Assessment | Basis |
|---|---|---|
| **Data availability** | ✅ Strong | Every critical input is open, free, confirmed accessible, and licence-clear |
| **Accuracy attainability** | ✅ Credible | Targets sit inside published international benchmark ranges, cited |
| **Compute feasibility** | ✅ Trivial | Runs on a laptop; no GPU in the critical path |
| **Engineering complexity** | ⚠️ Moderate | Archetype weighting is the genuine unknown; mitigation specified |
| **Regulatory fit** | ✅ Strong, and unusually well-timed | CERC bands halved effective 1 April 2026 |
| **Market structure fit** | ✅ Strong | ~80 % of TSOs use centralised regional forecasting — the architecture matches |
| **Operating cost** | ✅ Negligible | Fully open-source stack, one small VM |
| **Indian ground truth** | ⚠️ Known gap | Explicitly managed via the two-region design, stated rather than hidden |

**The project is feasible as scoped.** The two honest caveats — Indian ground-truth quality and archetype weighting — are both identified, both have specified mitigations, and both are stated in the pitch before a judge has to find them.

---

*Quantified benefits are in [`03-impact-and-benefits.md`](03-impact-and-benefits.md). All sources are in [`04-research-and-references.md`](04-research-and-references.md).*
