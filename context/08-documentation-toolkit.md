# Documentation Toolkit

> Supporting material that strengthens the submission: definitions, data dictionary, demo script, objection handling, and the things reviewers actually check.

---

## 1. Glossary

Getting this vocabulary right signals domain familiarity immediately. Getting it wrong is the fastest way to lose a domain expert in the room.

### Power system

| Term | Meaning |
|---|---|
| **Net load** | Demand minus renewable generation — what conventional plant and storage must cover |
| **Duck curve** | The net-load shape created by midday solar: a deep midday belly and a steep evening ramp |
| **Curtailment** | Deliberately reducing renewable output below what the resource allows, because the grid cannot absorb it |
| **Must-run** | Generation that cannot be turned down — nuclear, run-of-river hydro, thermal at technical minimum |
| **Technical minimum** | The lowest stable output of a thermal unit, typically ~55 % of rated for Indian coal |
| **Ramp rate** | Rate of change of net load, MW/hour. The binding constraint on the evening solar drop-off. |
| **Spinning reserve** | Synchronised capacity held back, available within minutes |
| **Capacity factor (CF)** | Actual output ÷ maximum possible over a period. The normalised forecasting target. |
| **Monitored capacity** | Installed capacity currently being measured — changes as a fleet grows |
| **DSM** | Deviation Settlement Mechanism — the Indian framework charging generators for schedule deviation |
| **SLDC / RLDC / NLDC** | State / Regional / National Load Despatch Centre |
| **SoC** | State of charge — stored energy as a fraction of capacity |
| **Round-trip efficiency** | Energy out ÷ energy in for a storage cycle |

### Solar

| Term | Meaning |
|---|---|
| **GHI** | Global Horizontal Irradiance — total shortwave on a horizontal surface (W/m²) |
| **DNI** | Direct Normal Irradiance — beam component perpendicular to the sun |
| **DHI** | Diffuse Horizontal Irradiance — scattered sky component |
| **POA** | Plane of Array — irradiance on the tilted panel surface. What actually matters. |
| **GTI** | Global Tilted Irradiance — synonym for POA global |
| **Clear-sky index (kt)** | `GHI / clearsky_GHI`. Normalises away season, latitude and time of day. **The key solar feature.** |
| **Solar zenith angle** | Angle between vertical and the sun; 90° at horizon |
| **Air mass** | Relative atmospheric path length the beam traverses |
| **Cell temperature** | Panel operating temperature — typically 20–30 °C above ambient in sun |
| **Clipping** | Inverter limiting output when DC exceeds AC rating |
| **DC/AC ratio** | Installed DC capacity ÷ inverter AC rating. Above ~1.2, clipping becomes common. |
| **Soiling** | Dust and dirt accumulation reducing output. Material in India. |
| **Albedo** | Ground reflectance contributing to rear/diffuse POA |

### Wind

| Term | Meaning |
|---|---|
| **Hub height** | Height of the turbine rotor centre — where wind speed matters |
| **Wind shear** | Increase of wind speed with height, modelled by the power law with exponent `alpha` |
| **Power curve** | Manufacturer mapping of wind speed to power output |
| **Cut-in / rated / cut-out** | Speeds at which a turbine starts (~3 m/s), reaches rated output (~12 m/s), and shuts down for safety (~25 m/s) |
| **Air density correction** | IEC 61400-12 normalisation of wind speed to standard density (1.225 kg/m³) |
| **Wake effect** | Downstream turbines receiving slowed, turbulent air from upstream ones |
| **Turbulence intensity** | Ratio of wind speed standard deviation to mean |
| **Yaw / nacelle direction** | Rotational orientation of the turbine housing |
| **Pitch angle** | Blade rotation used to regulate power above rated speed |

### Forecasting

| Term | Meaning |
|---|---|
| **NWP** | Numerical Weather Prediction — physics-based atmospheric models (ECMWF, ICON, GFS) |
| **Reanalysis** | Retrospective best-estimate weather state. More accurate than any forecast; not available forward. |
| **Lead time** | Hours between forecast issue and the valid hour |
| **Run / valid time** | When a forecast was issued vs the hour it describes. **Both are required as keys.** |
| **Persistence** | Baseline: tomorrow equals today |
| **Smart persistence** | `kt(t) × clearsky(t+h)` — persistence in clear-sky index rather than in power |
| **Skill score** | `1 − RMSE_model / RMSE_reference`. Scale-free and comparable. |
| **Pinball loss** | Quantile loss; the standard probabilistic forecasting metric |
| **CRPS** | Continuous Ranked Probability Score — generalises MAE to distributions |
| **PICP** | Prediction Interval Coverage Probability — fraction of actuals inside the band |
| **Conformal prediction** | Distribution-free method giving finite-sample coverage guarantees |
| **Heteroscedastic** | Error variance that changes with conditions — exactly the case here |
| **Censored label** | An observation clipped by an external constraint — e.g. curtailed output |
| **Train/serve skew** | Features computed differently at training and inference. The classic production failure. |

---

## 2. Metric definitions

Precise definitions, so a number can be reproduced.

```python
nRMSE  = sqrt(mean((y_pred - y_true)**2)) / capacity_mw
nMAE   = mean(abs(y_pred - y_true))       / capacity_mw
MBE    = mean(y_pred - y_true)            / capacity_mw     # signed

skill  = 1 - RMSE_model / RMSE_persistence

# pinball loss at quantile alpha
pinball(alpha, q, y) = (alpha)     * (y - q)   if y >= q
                       (1 - alpha) * (q - y)   if y <  q

PICP   = mean((y_true >= p10) & (y_true <= p90))     # target ≈ 0.80
ACE    = PICP - nominal_coverage                      # target ≈ 0

# ramp hit rate
ramp_hit_rate = (
    detected_ramps_with_lead_ge_Y / actual_ramps_exceeding_X_pct_per_hour
)
```

**Reporting conventions, applied without exception:**

- Solar metrics are **daylight-only** (solar elevation > 0). Stated alongside every number.
- All metrics reported **per lead hour**, not averaged across the horizon.
- Normalised by **installed capacity**, never by mean output — mean-normalisation makes low-output periods dominate.
- Curtailed and unavailable intervals excluded from both training and evaluation.
- Comparisons against Elia's forecast use a **matched information cutoff**.

---

## 3. Data dictionary — quick reference

| Field | Type | Unit | Source | Notes |
|---|---|---|---|---|
| `run_ts_utc` | timestamp | UTC | NWP | Model initialisation. **Part of the primary key.** |
| `valid_ts_utc` | timestamp | UTC | NWP | Hour being forecast. **Part of the primary key.** |
| `lead_hours` | int | h | derived | `valid − run`. A feature, not metadata. |
| `ghi_wm2` | float | W/m² | Open-Meteo | `shortwave_radiation` |
| `dni_wm2` | float | W/m² | Open-Meteo | `direct_normal_irradiance` |
| `dhi_wm2` | float | W/m² | Open-Meteo | `diffuse_radiation` |
| `clearsky_ghi_wm2` | float | W/m² | pvlib | Ineichen–Perez model |
| `clearsky_index_kt` | float | 0–1.2 | derived | `ghi / clearsky_ghi`. Can exceed 1 under cloud-edge enhancement. |
| `poa_global_wm2` | float | W/m² | pvlib | Transposition using tilt/azimuth |
| `cell_temperature_c` | float | °C | pvlib | SAPM model |
| `thermal_derate` | float | ratio | derived | `1 + gamma × (T_cell − 25)` |
| `physics_pac_mw` | float | MW | derived | Deterministic baseline; used as a feature |
| `wind_speed_100m_ms` | float | m/s | Open-Meteo | Note: API returns km/h by default — convert |
| `ws_hub_ms` | float | m/s | derived | Power-law shear to hub height |
| `air_density_kgm3` | float | kg/m³ | derived | `P / (287.05 × T_K)` |
| `ws_density_corrected_ms` | float | m/s | derived | IEC 61400-12 normalisation |
| `power_curve_cf` | float | 0–1 | derived | Piecewise manufacturer curve |
| `dP_dv` | float | 1/(m/s) | derived | Power-curve gradient — uncertainty driver |
| `power_mw` | float | MW | Elia | Metered output |
| `monitored_capacity_mw` | float | MW | Elia | Changes as fleet grows |
| `curtailed_mw` | float | MW | Elia / derived | **Censored-label flag** |
| `y_capacity_factor` | float | 0–1 | derived | `power / capacity`. **The training target.** |
| `sample_weight` | float | 0 or 1 | derived | 0 where curtailed or unavailable |

**Unit traps worth writing on the wall:**

- Open-Meteo returns wind speed in **km/h** unless `wind_speed_unit=ms` is set. A factor of 3.6 error in wind speed becomes a factor of ~47 error in power.
- Elia timestamps are **Europe/Brussels with DST**. Store UTC, display local.
- Irradiance is **instantaneous W/m²**; energy is **Wh/m²**. Do not mix.
- Elia publishes **MW**, not MWh. At 15-minute resolution, energy is `MW × 0.25`.

---

## 4. Demo script — eight minutes

| Time | Beat | What is shown | Point being made |
|---|---|---|---|
| 0:00–0:45 | **The problem, measured** | Ember figures: 2.1 TWh curtailed FY2025-26; 34 GWh lost on 30 March 2026 alone | This is a measured problem, not a hypothetical one |
| 0:45–1:30 | **The clock** | CERC deviation bands halved to ± 5 % effective 1 April 2026, phasing to zero tolerance by 2031 | The cost of forecast error doubled *this* financial year |
| 1:30–2:45 | **The core idea** | Physics-plus-residual diagram. Show a physics-only forecast, then the residual correction | Most of the answer is computable; learning only corrects the remainder |
| 2:45–4:00 | **It works** | Skill-vs-lead-hour chart: persistence, physics, our model, Elia's forecast — four lines | Measured against a professional operational benchmark, not one we chose |
| 4:00–5:00 | **It is honest** | Coverage plot: PICP vs nominal across the band | Our 80 % band covers 80 %. Almost nobody shows this. |
| 5:00–6:30 | **It decides** | Live dashboard: pick an over-generation day, show the action queue with MWh, ₹ and confidence | This is the actual ask in the problem statement |
| 6:30–7:15 | **It plans** | Battery sizing sweep: curtailment avoided vs MWh installed | Forecasting tool becomes an infrastructure planning instrument |
| 7:15–8:00 | **It transfers** | Same pipeline, Indian configuration. State the modelled-data caveat out loud. | Belgium proves accuracy; India proves transfer |

**Rules for the demo:**

1. Run from `--replay` cache. Say so. Live APIs fail during demos far more often than models do.
2. Show one **specific date** with a real event, not an average. "On 14 March we flagged a 340 MW ramp six hours early" beats any aggregate statistic.
3. Show the **coverage plot**. It takes fifteen seconds and it is the single most differentiating slide.
4. State the Indian data caveat **yourself**, in the demo, before Q&A.

---

## 5. Objection handling

Prepared answers, each grounded in something concrete.

| Objection | Response |
|---|---|
| *"Why not just use an LSTM?"* | At 24–72 h lead, autoregressive lags carry almost no information — this is NWP post-processing, not time-series forecasting. Also, GEFCom2014 was won by a tree ensemble. We benchmark deep models in phase 6 and will report the result either way. |
| *"Your RMSE looks too good."* | It is daylight-only and per lead hour. Including night zeros would roughly halve it and mean nothing. Here is the per-lead-hour curve. |
| *"How do you know the intervals are right?"* | Split-conformal calibration gives a distribution-free finite-sample guarantee, and here is the measured PICP against nominal coverage. |
| *"This only works because Belgium has good data."* | Correct, and that is the point of the two-region design. Belgium is where we can *prove* accuracy against a professional benchmark. India is where we prove the pipeline transfers. We do not claim Indian accuracy. |
| *"Commercial vendors already do this."* | Yes. Our contribution is open data, a transparent method, calibrated intervals, and a decision layer — not beating proprietary vendors on raw accuracy. |
| *"What about extreme weather?"* | Cut-out shutdown is an explicit flag; a whole wind fleet can go to zero in minutes. Handled as a distinct event type with its own alert. Rare-event accuracy is genuinely limited by training data availability — we say so. |
| *"Storage numbers are made up."* | Storage is a declared configuration parameter, not a measured asset — stated explicitly. That is deliberate: it makes the sizing sweep possible, which is arguably the most valuable output. |
| *"The physics model is doing all the work."* | Partly true at rung 1, and we show that decomposition explicitly — physics-only is one of the four lines on the skill chart. The residual model adds N % on top, and that number is reported separately. |
| *"Have you tested on live data?"* | Phase 0 onward runs against the live Open-Meteo forecast endpoint. The replay cache is for demo reliability, not because the live path is absent. |
| *"What's your biggest weakness?"* | Train/serve skew — we train on reanalysis and serve on forecasts. Here is the bias-correction and conformal mitigation, and here is what remains unresolved. |

---

## 6. Judging criteria map

| Typical criterion | Where the evidence is |
|---|---|
| **Novelty / innovation** | Physics-plus-residual hybrid; conformal-calibrated intervals; battery sizing sweep as an output; two-region transfer design |
| **Technical depth** | `01-technical-approach.md` §3–§6; interface contracts and schemas in `06-system-architecture.md` |
| **Feasibility** | `02-feasibility-and-viability.md` — every input confirmed open and accessible; no GPU; targets inside published benchmark ranges |
| **Impact** | `03-impact-and-benefits.md` — third-party measured figures, arithmetic shown, assumptions marked |
| **Scalability** | `02` §5.2 and `06` §9 — region is a config file; no rewrite needed |
| **Use of data** | `04-research-and-references.md` — full schemas, access notes, licence status |
| **Presentation** | `05-high-level-architecture.md` §9 (the 60-second version) and the demo script above |
| **Team understanding** | `07-risks-assumptions-limitations.md` — especially §5, "what would change our mind" |

---

## 7. What reviewers actually check

Ranked by how often it catches a project out.

1. **"Show me your train/test split."** Random splits on time series are the most common fatal flaw. Answer: walk-forward with a gap, splits defined before modelling.
2. **"What's your baseline?"** A project with no baseline has no result. Answer: persistence and physics, both reported on every chart.
3. **"Where did this number come from?"** Every external figure cited with source and date; every derived figure shows its arithmetic.
4. **"What happens when it's wrong?"** P10/P90 plus the coverage plot.
5. **"Does it run?"** Live path plus replay cache. Demonstrate, do not describe.
6. **"What did you not build?"** `07` §4 answers this explicitly.
7. **"Who would use this and why?"** `02` §4.3 — the per-stakeholder table with the problem each one has.

---

## 8. Reproducibility checklist

| Item | Status |
|---|---|
| All data sources open and free | ✅ Every critical input |
| No API keys in the critical path | ✅ Open-Meteo non-commercial; Elia open portal |
| Environment pinned | `requirements.txt` with exact versions |
| Random seeds fixed | Set in all training scripts |
| Config separated from code | `config/regions/*.yaml` |
| Evaluation harness frozen before modelling | Built in phase 0 |
| Model versions tracked | MLflow registry; version in every API response |
| Replay data cached | `artifacts/` — one known-good week |
| Golden-file integration test | Fixed fixture week, output compared |
| Train/serve feature parity test | The single most important test in the repository |

---

## 9. Writing conventions used across this set

| Convention | Why |
|---|---|
| Every external number carries source and date inline | Verifiable |
| Every derived number shows its arithmetic | Falsifiable |
| Assumptions marked **[ASSUMED]** with their basis | Honest |
| Contested sources cited alongside the challenge (e.g. DSM litigation) | Complete |
| Status markers **[BUILT]** / **[PLANNED]** / **[STRETCH]** used accurately | Credible |
| Limitations stated before a reader finds them | Persuasive |
| Numbers that could be confused (2.1 TWh vs 2.3 TWh) distinguished explicitly | Careful |

The cumulative effect matters more than any single document. A reviewer who checks three claims and finds all three properly sourced stops checking and starts trusting. A reviewer who finds one overstated claim re-examines everything.
