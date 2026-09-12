# AI-Powered Renewable Generation Forecasting Platform

**Documentation set — v1.0**

---

## Proposed Solution (one page)

A **regional renewable generation forecasting and grid-action platform** that ingests numerical weather predictions, historical generation records and fleet-level site parameters, and produces **probabilistic 24–72 hour forecasts** of solar and wind output — then converts those forecasts into **sized, priced, confidence-rated grid actions**: curtail, charge storage, discharge storage, or commit backup.

Three design choices separate this from a conventional forecasting notebook.

**1. Physics carries the deterministic part; machine learning carries only the residual.**
A solar plant's output at 11:00 on 14 March is ~90 % determined by orbital mechanics, panel geometry and clear-sky radiative transfer. All of that is computable exactly, today, with no training data. We compute it with `pvlib` (solar) and IEC-normalised power curves (wind), and the gradient-boosting model learns only what physics got wrong — soiling, wake losses, local microclimate, sensor drift. This is why the platform can forecast a site with **zero generation history**, and why it needs far less training data than an end-to-end deep model.

**2. Every forecast is an interval, and the interval is verified.**
Point forecasts are operationally useless for reserve sizing. We emit P10/P50/P90 via quantile regression, then apply **split-conformal calibration** so the 80 % band demonstrably covers ~80 % of outcomes. We publish the coverage plot. An uncalibrated band is a decoration; a calibrated one is a reserve requirement.

**3. The output is an action, not a chart.**
The brief's users — grid operators, utilities, traders — do not need a curve. They need to know that on Tuesday between 11:00 and 14:00 the region will be 340 MW over-generating with 80 % confidence, that charging the battery absorbs 210 MWh of it, and that the remaining 130 MWh must be curtailed at a cost of ₹X. The decision engine carries the uncertainty band through to the recommendation, so an operator can see which actions are decisions and which are suggestions.

---

## Decisions locked

| Decision | Choice | Rationale |
|---|---|---|
| **Forecast scale** | Regional / system-level | Matches the stated user (grid operator). Spatial averaging also makes the problem tractable — individual cloud noise cancels. |
| **Technology coverage** | Solar **and** wind, one shared pipeline | Identical Layer-0/Layer-2 schema; only the physics module differs. Matches the problem statement literally. |
| **Primary region** | Belgium (Elia), then India | Elia is the only open portal publishing generation **+ its own forecast + load** on one 15-minute timebase. India is where the problem matters. |
| **Weather backbone** | Open-Meteo (multi-model) | Same variable names historically and forward — eliminates train/serve skew. ECMWF + ICON + GFS satisfies the "one global + one regional model" best practice. |
| **Model** | Physics baseline + LightGBM residual + quantile heads + conformal calibration | Highest accuracy per unit of effort; explainable via SHAP; trains in seconds. |
| **Horizon** | 24–72 h, hourly, issued daily | NWP-dominated band. Autoregressive lags below 24 h are excluded as leakage. |
| **Storage** | Parameterised virtual BESS | No open BESS dispatch series exists for either region. Configurable storage is more useful anyway — it enables a sizing study. |
| **Benchmark** | Beat persistence; match Elia's published day-ahead forecast | Honest and achievable. An external professional benchmark is worth more than any absolute RMSE. |

---

## Document index

| # | File | What it covers | Use it for |
|---|---|---|---|
| 01 | [`01-technical-approach.md`](01-technical-approach.md) | Data model, feature engineering, model ladder, uncertainty quantification, decision engine, evaluation protocol | The technical section of the submission; onboarding a new engineer |
| 02 | [`02-feasibility-and-viability.md`](02-feasibility-and-viability.md) | Data availability, accuracy targets vs published benchmarks, compute budget, regulatory fit, deployment path, business model | The "can you actually build this" question |
| 03 | [`03-impact-and-benefits.md`](03-impact-and-benefits.md) | Quantified curtailment losses, DSM penalty exposure, emissions, per-stakeholder benefit, value model with worked arithmetic | The "why does this matter" question |
| 04 | [`04-research-and-references.md`](04-research-and-references.md) | Datasets, literature, standards, regulations, tooling — all with links and access notes | Citations; defending any claim under questioning |
| 05 | [`05-high-level-architecture.md`](05-high-level-architecture.md) | Conceptual architecture, layer model, data flow, technology stack | Slides; the 60-second explanation |
| 06 | [`06-system-architecture.md`](06-system-architecture.md) | Component-level design, module contracts, schemas, API surface, deployment topology, sequence diagrams | Implementation; the deep-dive question |
| 07 | [`07-risks-assumptions-limitations.md`](07-risks-assumptions-limitations.md) | Risk register with mitigations, explicit assumptions, known limitations, what we deliberately did not build | Credibility. The section that separates a prototype from engineering. |
| 08 | [`08-documentation-toolkit.md`](08-documentation-toolkit.md) | Glossary, metric definitions, data dictionary, demo script, objection-handling FAQ, judging-criteria map | Presentation prep and Q&A defence |

---

## How to read this set

- **Presenting in five minutes?** README (this file) + `05-high-level-architecture.md` + the value model in `03-impact-and-benefits.md`.
- **Writing the submission form?** Files 01, 02, 03, 04 map directly onto the standard Technical Approach / Feasibility and Viability / Impact and Benefits / Research and References sections.
- **Building?** Files 06 and 01, in that order. Fix the three interface contracts in `06-system-architecture.md` before anyone writes model code.
- **Being questioned?** File 07 first, then the FAQ in file 08. The fastest way to lose a technical judge is to be surprised by a limitation you should have named yourself.

---

## Status conventions used throughout

| Marker | Meaning |
|---|---|
| **[BUILT]** | Implemented and demonstrable |
| **[PLANNED]** | Designed, scheduled, not yet implemented |
| **[STRETCH]** | Desirable, explicitly out of current scope |
| **[ASSUMED]** | A modelling assumption, not a measured fact — always stated with its basis |

Applying these markers honestly is worth more than claiming everything is built. A judge who finds one overstated claim discounts every other claim in the document.

---

*All external figures are cited inline with their source and date. Where a number is our own estimate or assumption, it is marked **[ASSUMED]** and its basis is given.*
