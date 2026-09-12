# Risks, Assumptions and Limitations

> This is the document that separates a prototype from engineering. A reviewer who discovers an unstated limitation discounts every other claim in the submission. Stating them first converts a weakness into evidence of rigour.

---

## 1. Risk register

Scored as Likelihood × Impact. **Response** is what we actually do, not a reassurance.

### 1.1 Data risks

| # | Risk | L | I | Response |
|---|---|---|---|---|
| D1 | **Open-Meteo unavailable or rate-limited during demonstration** | M | H | `--replay` mode serves a cached week of forecast runs plus all model artifacts. Build the live path, demo from cache, state which is shown. |
| D2 | **Elia portal schema changes mid-project** | L | M | `pandera` validation fails loudly at the boundary. Previous forecast continues to serve; no silent corruption. |
| D3 | **Curtailment not flagged in source data** | **H** | M | Heuristic detection — output flat-lining well below physics estimate under good conditions. `qc_flag = CURTAILED`, `sample_weight = 0`. Stated as a limitation, not solved. |
| D4 | **Indian dataset too coarse (hourly, modelled) for a credible demo** | M | M | Two-region design exists for this. Belgium carries accuracy; India carries transfer. Never conflated. |
| D5 | **Monitored capacity changes during training window** | **H** | L | Elia publishes `monitored_capacity`. Training on capacity factor absorbs fleet growth automatically — this is a designed-for case, not a risk. |
| D6 | **Weather grid points poorly represent the fleet** | M | M | Capacity-weighted sampling. Weights refined against observed generation shape. Sensitivity tested by varying point count. |
| D7 | **Open-Meteo non-commercial terms block productionisation** | H (for commercial) | L | Adapter pattern — weather source is one interface. Swapping to a paid feed or IMD is one implementation. |

### 1.2 Modelling risks

| # | Risk | L | I | Response |
|---|---|---|---|---|
| M1 | **Temporal leakage via short lags** | M | **H** | Structurally prevented: no lag < 24 h enters the feature set. Walk-forward splits with a gap. Documented as a design rule, enforced in code review. |
| M2 | **Archetype weights fail to converge** | M | M | Fall back to a single average archetype. Accuracy degrades; the pipeline stands. Report the degradation rather than hiding it. |
| M3 | **Train/serve skew** — reanalysis training, forecast serving | **H** | **H** | Open-Meteo provides the same variable names both sides. Residual bias corrected by `nwp_bias_lag_7d` fitted on recent live forecasts. **Explicitly named as a known residual limitation.** |
| M4 | **Conformal intervals too wide to be actionable** | L | M | Wide-but-honest beats narrow-and-wrong. Width is a property of the problem. Report it; do not tune it away. |
| M5 | **Quantile crossing** (`p10 > p50`) | M | L | Property test enforces monotonicity; post-hoc sorting applied if violated. |
| M6 | **Night rows inflate solar accuracy** | **H** | M | Daylight-only metrics for solar, stated alongside every number. |
| M7 | **Model degrades silently after deployment** | M | H | Candidate-must-beat-incumbent promotion gate. Drift monitors on feature and error distributions. |
| M8 | **Deep models do not beat LightGBM** | **H** | L | Expected. Reported as a documented negative result — which is a stronger finding than a marginal win. |

### 1.3 Delivery and scope risks

| # | Risk | L | I | Response |
|---|---|---|---|---|
| S1 | **Cannot match Elia's operational forecast** | **H** | L | Never claimed. Target is "beat persistence, close the gap to the TSO within a stated margin using only open data". |
| S2 | **Phase 6 stretch goals consume core build time** | M | M | Phases end at demonstrable checkpoints. Phase 6 begins only after phase 5 is complete. |
| S3 | **Dashboard takes longer than expected** | M | M | API is the contract; dashboard consumes it. A working API with a thin UI beats a beautiful UI over a broken API. |
| S4 | **Integration left to the end** | M | **H** | Three interface contracts fixed on day one; end-to-end fixture test from phase 0. |

### 1.4 External risks

| # | Risk | L | I | Response |
|---|---|---|---|---|
| E1 | **CERC DSM regulations struck down in litigation** | M | M | Karnataka HC has stayed the revised DSM; Delhi HC proceedings pending. Cited alongside the regulation. The regulatory *direction* is consistent across CERC and multiple state commissions, and the curtailment problem exists regardless of settlement rules. |
| E2 | **Operators will not trust a model enough to reduce reserves** | **H** | M | Institutional change, not technical. This is why calibration evidence (PICP, reliability diagram) is in the core build rather than treated as a refinement. |
| E3 | **Commercial forecasting vendors already serve this market** | H | L | True. Differentiation is open data, transparency, calibrated intervals, and the decision layer — not raw accuracy against incumbents. |

---

## 2. Assumptions register

Every assumption used anywhere in this documentation set, with its basis and what happens if it is wrong.

### 2.1 Technical assumptions

| Assumption | Value | Basis | If wrong |
|---|---|---|---|
| Belgian PV archetype mix | 45 % rooftop-S, 30 % rooftop-EW, 25 % ground | Typical European residential/commercial mix | Fitted by clear-sky optimisation; converges to observed shape |
| Belgian wind split | 55 % onshore / 45 % offshore | Elia publishes the split — to be replaced with the actual figure | Trivially corrected |
| Shear exponent | 0.20 onshore, 0.11 offshore | Standard values for terrain roughness class | Small bias; absorbed by residual model |
| Wake loss | 8 % onshore, 12 % offshore | Typical industry range | Absorbed by residual model |
| Inverter efficiency | 98.5 % | Modern string inverters | Negligible |
| Temperature coefficient | −0.0035 /°C | Typical crystalline silicon | Small; absorbed by residual |
| Soiling derate | 3 % | Moderate; highly site-specific | **Significant for Indian sites** — dust loading is far higher; must be re-estimated |
| Round-trip efficiency | 88 % | Current Li-ion grid-scale | Changes storage economics proportionally |
| Grid emission factor | 0.71 tCO₂/MWh | CEA CO₂ Baseline Database, Indian grid average | Marginal emissions at the curtailment hour would differ |

### 2.2 Impact assumptions

| Assumption | Value | Basis | Sensitivity |
|---|---|---|---|
| Curtailment addressable by anticipation | 25 % | Deliberately below the ~36 % non-transmission residual implied by Ember's Q1 2026 split | Linear on the headline benefit |
| Recoverable share of addressable curtailment | 40 % | Judgement based on storage availability and response time | Linear |
| Avoided-cost value of energy | ₹3.00/kWh | Typical Indian RE PPA range | Linear |
| DSM exposure on breaching intervals | ₹0.50/kWh | Illustrative; actual slabs vary by state | Linear |
| Baseline operator forecast error | 12 % | GET.transform: Mexico operators 9 %, Chile 13 % | Determines the size of the improvement |
| Reserve margin held against RE uncertainty | 8 % | Judgement | Linear on the reserve benefit |
| Reserve reduction from calibrated P90 | 20 % | Judgement — the least defensible number in the set | Linear; this is the softest figure we publish |
| Plant capacity factor (illustrative 100 MW solar) | 19 % | Typical Indian solar CF | Linear |
| Compute cost | ₹2,000–4,000/month | Small cloud VM | Negligible either way |

> **The reserve-reduction assumption (20 %) is the weakest number in this documentation set.** It depends on an operator changing institutional practice on the strength of a calibration plot. We publish it because the mechanism is real, and we flag it because the magnitude is genuinely uncertain. If challenged, concede it immediately and fall back to the curtailment and DSM figures, which rest on measured data.

### 2.3 Scope assumptions

| Assumption | Rationale |
|---|---|
| 24–72 h is the operative horizon | Stated in the brief. Drives the exclusion of short lags. |
| Regional, not plant-level | ~80 % of TSOs use centralised regional forecasting (GET.transform) |
| Solar and wind only | Hydro and biomass are a new physics module behind the same interface |
| Hourly resolution for modelling | Matches NWP native resolution; 15-min Elia data is aggregated up |
| One storage asset per region | Multiple assets are a straightforward extension of the simulator |
| Demand forecast taken as given | Elia publishes one. Forecasting demand is a separate problem. |

---

## 3. Known limitations

Stated plainly. These are properties of the system as designed, not bugs to be fixed.

### 3.1 What the system cannot do

| Limitation | Why | Mitigation / honest framing |
|---|---|---|
| **Cannot recover transmission-constrained curtailment** | Forecasting does not build wires. Ember attributes ~64 % of Q1 2026 Indian curtailment to transmission. | We address the anticipation-driven remainder and say so explicitly. |
| **Cannot forecast below ~6 h with full skill** | Needs satellite cloud advection or sky cameras, which are separate data sources and a separate model | Nowcasting is a named extension, not a claim |
| **Cannot beat a well-resourced commercial TSO forecast** | They have proprietary NWP, on-site measurements, and years of tuning | Target is stated as "close the gap", never "beat" |
| **Cannot validate accuracy on Indian data** | No open metered sub-hourly Indian regional series exists | Two-region design; Belgium carries accuracy, India carries transfer |
| **Cannot model plant-specific outages** | Requires maintenance schedules not publicly available | `availability_pct` is an input where it exists; otherwise absorbed as noise |
| **Cannot price actions precisely** | Requires live market and imbalance prices | Prices are configuration with stated assumptions, not claimed market data |

### 3.2 Where the accuracy claim is bounded

1. **Daylight-only for solar.** Including night zeros would roughly halve the reported nRMSE and mean nothing.
2. **Per lead hour, not averaged.** A single 72-hour-average number hides where the skill actually is. We report the curve.
3. **Belgium, not India.** The Indian figures demonstrate that the pipeline runs, not how accurate it is.
4. **Against a matched information cutoff.** Elia's day-ahead forecast is issued at a specific time; comparing against forecasts issued at a different moment would be meaningless.
5. **Curtailed intervals excluded from training and from evaluation.** Including them would score the model against censored labels.

### 3.3 The train/serve skew we have not fully eliminated

This is the most technically significant residual limitation, and it is worth stating precisely.

**The problem:** ideally a model is trained on *archived forecasts* — what the NWP said 48 hours before each historical hour. In practice, publicly available historical weather is *reanalysis* — the best retrospective estimate, which is systematically more accurate than any forecast.

**The consequence:** a model trained on reanalysis learns a cleaner irradiance-to-power relationship than it will encounter at serving time.

**What we do about it:**

- Use Open-Meteo for both training and serving, so variable definitions, units and grid interpolation are identical.
- Fit a `nwp_bias_lag_7d` correction on the last seven days of *live* forecasts versus observed outcomes, which absorbs the systematic component of the gap.
- Re-fit conformal calibration on live forecast errors, so the *interval* is correct even where the point forecast carries residual bias.

**What remains:** the point forecast is likely slightly optimistic relative to true operational performance. The conformal layer catches this in the interval width — which is one more reason the calibration stage is in the core build rather than treated as a refinement.

**Full resolution** requires archiving our own forecasts for several months and retraining on them — which is the correct production answer, and is named as such rather than claimed.

---

## 4. What we deliberately did not build

Naming these prevents the reading that they were forgotten.

| Not built | Why |
|---|---|
| **End-to-end deep learning** | Tree ensembles won GEFCom2014. Deep models need more data and more tuning for a gain LightGBM often erases. Benchmarked in phase 6 as a documented experiment. |
| **Sub-6-hour nowcasting** | Different data sources (satellite, sky cameras), different model class. Out of the stated 24–72 h scope. |
| **Full MILP unit commitment** | The rule-based engine answers the brief. LP dispatch is phase 6. |
| **Multi-tenant auth and RBAC** | Not a research question. Straightforward, and not what the work is being judged on. |
| **Own NWP model** | Running a weather model is a different discipline entirely. |
| **Real-time streaming** | NWP updates four times daily. Streaming infrastructure would add complexity with no benefit at this cadence. |
| **Demand forecasting** | Elia publishes one. A separate, well-studied problem. |
| **Spatio-temporal GNN** | Genuine research territory. SDWPF turbine coordinates make it possible; it is a stretch goal, not a plan. |

---

## 5. What would change our mind

Intellectual honesty about falsification.

| Finding | What it would mean |
|---|---|
| Physics baseline does not beat persistence at 24–72 h | The physics-first thesis is wrong for this horizon. Fall back to pure ML. |
| Residual model does not beat direct GBDT on raw target | Residual learning adds no value here. Simplify. |
| Conformal intervals require > 50 % of capacity width | The problem may be too uncertain at 72 h for reserve sizing. Reduce the advertised horizon. |
| Archetype weighting does not beat a single average archetype | The regional physics refinement is over-engineering. Remove it. |
| Elia's forecast beats ours by > 50 % relative | The open-data approach has a harder ceiling than assumed. Report it as a finding. |
| Ramp detection lead time < 2 hours | The operational value proposition weakens substantially; reframe around scheduling rather than ramp response. |

**Each of these is a measurable outcome of the evaluation harness, not a matter of opinion.** Building the harness in phase 0 is what makes it possible to be wrong in public and say so.

---

## 6. How to use this document under questioning

| If asked | Answer with |
|---|---|
| "How accurate is it?" | Skill score vs persistence, per lead hour, daylight-only for solar, plus the gap to Elia's forecast. Never a bare RMSE. |
| "Will this work in India?" | Belgium proves accuracy, India proves transfer. State the data limitation before they do. |
| "What if the forecast is wrong?" | That is what P10/P90 is for, and here is the coverage plot showing the band is honest. |
| "Why not deep learning?" | GEFCom2014 was won by a tree ensemble. We benchmark deep models in phase 6 and will report the result either way. |
| "Isn't this already solved commercially?" | Yes, by proprietary vendors. Our contribution is open data, transparent method, calibrated intervals and the decision layer. |
| "What's your biggest weakness?" | Train/serve skew from training on reanalysis. Here is what we do about it and what remains. |

The last row matters most. Having a genuine, specific, well-understood answer to "what is your biggest weakness" is among the strongest signals a technical reviewer can receive — and having no answer is among the weakest.
