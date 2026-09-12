# dev-03 — ML Pipeline

**Stack:** pvlib · windpowerlib · LightGBM · MAPIE · SHAP · MLflow
**Principle:** physics computes what is deterministic; the model learns only the residual.

---

## 1. Pipeline order

```
weather_nwp + site_master
        │
        ├─► features/solar.py   ─┐
        ├─► features/wind.py    ─┤
        ├─► features/temporal.py─┼─► features/build.py ─► archetype.py ─► model matrix
        └─► features/nwp_quality─┘        (per archetype)   (capacity-weighted sum)
                                                                  │
        physics baseline ◄────────────────────────────────────────┤
                │                                                 │
                └──► residual target  y − P_phys ──► LightGBM ──► quantiles ──► conformal ──► P10/P50/P90
```

Build in exactly this order. Each stage is independently testable, and the physics stage is a working
forecaster on its own — so you have something demonstrable after stage 1.

---

## 2. Solar physics — `features/solar.py`

```python
import pvlib, pandas as pd, numpy as np

def solar_features(wx: pd.DataFrame, site) -> pd.DataFrame:
    """wx indexed by valid_ts_utc (tz-aware UTC), columns ghi_wm2/dni_wm2/dhi_wm2/
    temperature_2m_c/wind_speed_10m_ms. `site` is one solar Archetype + lat/lon/elev."""
    idx = wx.index
    loc = pvlib.location.Location(site.lat, site.lon, tz="UTC", altitude=site.elevation_m)

    sp = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx)                       # Ineichen–Perez

    f = pd.DataFrame(index=idx)
    f["solar_zenith_deg"]    = sp["apparent_zenith"]
    f["solar_azimuth_deg"]   = sp["azimuth"]
    f["solar_elevation_deg"] = sp["elevation"]
    f["airmass"]             = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"]).fillna(0)
    f["is_day"]              = (sp["elevation"] > 0).astype("int8")

    f["clearsky_ghi_wm2"] = cs["ghi"]
    f["clearsky_dni_wm2"] = cs["dni"]

    # ── the single most important solar feature ────────────────────────────
    f["clearsky_index_kt"] = np.where(
        cs["ghi"].values > 20,
        wx["ghi_wm2"].values / np.maximum(cs["ghi"].values, 1e-6),
        0.0,
    ).clip(0, 1.3)          # >1 is real: cloud-edge enhancement. Do not clip to 1.

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.tilt, surface_azimuth=site.azimuth,
        solar_zenith=sp["apparent_zenith"], solar_azimuth=sp["azimuth"],
        dni=wx["dni_wm2"], ghi=wx["ghi_wm2"], dhi=wx["dhi_wm2"],
        albedo=site.albedo or 0.2,
    )
    f["poa_global_wm2"] = poa["poa_global"].fillna(0)

    tcell = pvlib.temperature.sapm_cell(
        poa["poa_global"].fillna(0), wx["temperature_2m_c"], wx["wind_speed_10m_ms"],
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
    )
    f["cell_temperature_c"]  = tcell
    f["thermal_derate"]      = 1 + site.gamma_pdc * (tcell - 25)

    dc_cap = site.capacity_mw * site.dc_ac_ratio
    p_dc   = dc_cap * (f["poa_global_wm2"] / 1000.0) * f["thermal_derate"]
    f["physics_pac_mw"]    = np.minimum(p_dc * 0.985, site.capacity_mw)   # inverter clipping
    f["clipping_headroom"] = p_dc / site.capacity_mw

    sr, ss = loc.get_sun_rise_set_transit(idx.normalize().unique())[["sunrise", "sunset"]].T.values
    f["mins_since_sunrise"] = _minutes_since(idx, sr)
    f["mins_to_sunset"]     = _minutes_until(idx, ss)
    return f
```

**Do not clip `clearsky_index_kt` at 1.0.** Values up to ~1.3 are physically real — cloud-edge enhancement
reflects extra light onto the panel. Clipping them removes a genuine high-output signal.

---

## 3. Wind physics — `features/wind.py`

```python
def wind_features(wx: pd.DataFrame, site) -> pd.DataFrame:
    f = pd.DataFrame(index=wx.index)
    f["ws_10m_ms"]  = wx["wind_speed_10m_ms"]
    f["ws_100m_ms"] = wx["wind_speed_100m_ms"]

    # power-law shear extrapolation to hub height
    f["ws_hub_ms"] = f["ws_100m_ms"] * (site.hub_height_m / 100.0) ** site.shear_alpha

    # air density from the ideal gas law, then IEC 61400-12 normalisation
    T_k = wx["temperature_2m_c"] + 273.15
    f["air_density_kgm3"]        = wx["surface_pressure_hpa"] * 100 / (287.05 * T_k)
    f["ws_density_corrected_ms"] = f["ws_hub_ms"] * (f["air_density_kgm3"] / 1.225) ** (1 / 3)

    v = f["ws_density_corrected_ms"]
    f["power_curve_cf"]   = power_curve(v, site.cut_in_ms, site.rated_ms, site.cut_out_ms)
    f["physics_power_mw"] = f["power_curve_cf"] * site.capacity_mw * (1 - site.wake_loss_frac)

    # local gradient — sensitivity, and therefore an uncertainty driver
    dv = 0.5
    f["dP_dv"] = (power_curve(v + dv, site.cut_in_ms, site.rated_ms, site.cut_out_ms) -
                  power_curve(v - dv, site.cut_in_ms, site.rated_ms, site.cut_out_ms)) / (2 * dv)

    f["turbulence_proxy"] = wx["wind_gusts_10m_ms"] / f["ws_10m_ms"].clip(lower=0.5) - 1
    rad = np.radians(wx["wind_direction_100m_deg"])
    f["wind_dir_sin"], f["wind_dir_cos"] = np.sin(rad), np.cos(rad)

    f["below_cutin_flag"]  = (v < site.cut_in_ms).astype("int8")
    f["above_cutout_flag"] = (v > site.cut_out_ms).astype("int8")
    return f


def power_curve(v, cut_in=3.0, rated=12.0, cut_out=25.0):
    """Normalised 0–1. Cubic between cut-in and rated, flat to cut-out, zero outside."""
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated)
    p[ramp] = (v[ramp] ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)
    p[(v >= rated) & (v <= cut_out)] = 1.0
    return p
```

Replace the analytic curve with the manufacturer's table when you have it — `windpowerlib` interpolates one
directly. The cubic form is a good approximation and a fine starting point.

> **Cut-out is a cliff, not a slope.** Above ~25 m/s a whole fleet drops to zero within minutes. A smooth
> model will never predict it, which is why `above_cutout_flag` is an explicit feature and a distinct
> event type in the decision layer.

---

## 4. NWP-quality features — `features/nwp_quality.py`

These predict **our own error**. They are what turns a point forecast into an honest interval, and they are
the family most often omitted.

```python
def nwp_quality_features(wx_by_model: dict[str, pd.DataFrame],
                         recent_errors: pd.DataFrame | None) -> pd.DataFrame:
    models = list(wx_by_model)
    ghi = pd.DataFrame({m: wx_by_model[m]["ghi_wm2"] for m in models})
    ws  = pd.DataFrame({m: wx_by_model[m]["wind_speed_100m_ms"] for m in models})

    f = pd.DataFrame(index=ghi.index)
    f["lead_hours"] = wx_by_model[models[0]]["lead_hours"]

    # disagreement between NWP models — a strong, cheap uncertainty signal
    f["ghi_model_disagreement"] = ghi.std(axis=1)
    f["ws_model_disagreement"]  = ws.std(axis=1)
    f["ghi_model_range"]        = ghi.max(axis=1) - ghi.min(axis=1)

    # blended point input
    f["ghi_blend_wm2"] = ghi.mean(axis=1)
    f["ws_blend_ms"]   = ws.mean(axis=1)

    # ramp in the forecast itself
    f["nwp_ghi_ramp"] = f["ghi_blend_wm2"].diff().fillna(0)
    f["nwp_ws_ramp"]  = f["ws_blend_ms"].diff().fillna(0)

    # rolling bias of the NWP against observations — corrects systematic drift
    if recent_errors is not None and len(recent_errors):
        f["nwp_bias_lag_7d"] = recent_errors["ghi_error"].rolling("7D").mean().reindex(f.index).ffill()
    else:
        f["nwp_bias_lag_7d"] = 0.0
    return f
```

---

## 5. Archetype aggregation — `features/archetype.py`

A region has no single tilt or hub height. Run the physics chain per archetype, then sum by capacity share.

```python
def aggregate_region(wx: pd.DataFrame, cfg, tech: str) -> pd.DataFrame:
    parts, weights = [], []
    for arc in cfg.archetypes[tech]:
        site = merge_site(cfg, arc, tech)
        f = solar_features(wx, site) if tech == "solar" else wind_features(wx, site)
        parts.append(f); weights.append(arc.share)

    out = pd.DataFrame(index=wx.index)
    for col in parts[0].columns:
        if col.endswith("_flag") or col == "is_day":
            out[col] = parts[0][col]                       # geometric, identical across archetypes
        else:
            out[col] = sum(w * p[col] for w, p in zip(weights, parts))
    return out
```

### Fitting the archetype shares

The shares start as assumptions. Refine them against observed generation, on **clear-sky days only**:

```python
from scipy.optimize import minimize

def fit_shares(wx, actual_mw, cfg, tech="solar"):
    """Restricting to clear-sky days is deliberate: the only remaining degrees of freedom
    are geometric, so the fit identifies orientation rather than absorbing cloud error."""
    per_arc = [solar_features(wx, merge_site(cfg, a, tech))["physics_pac_mw"]
               for a in cfg.archetypes[tech]]
    X = np.column_stack(per_arc)

    kt = wx["ghi_wm2"] / wx["clearsky_ghi_wm2"].clip(lower=1)
    clear = (kt > 0.85) & (wx["is_day"] == 1)
    Xc, yc = X[clear], actual_mw[clear].values

    def loss(w): return np.sqrt(np.mean((Xc @ w - yc) ** 2))
    n = X.shape[1]
    res = minimize(loss, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return dict(zip([a.id for a in cfg.archetypes[tech]], res.x.round(4)))
```

This converts three guessed numbers into three estimated numbers with a stated procedure — a meaningful
difference under questioning.

---

## 6. The model — `models/residual_gbdt.py`

```python
import lightgbm as lgb, numpy as np, pandas as pd

QUANTILES = (0.1, 0.5, 0.9)

PARAMS = dict(
    objective="quantile", metric="quantile",
    n_estimators=800, learning_rate=0.04,
    num_leaves=63, min_child_samples=40,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1, n_jobs=-1,
)

def train_residual(X: pd.DataFrame, y_cf: pd.Series, physics_cf: pd.Series,
                   sample_weight: pd.Series) -> dict[float, lgb.LGBMRegressor]:
    """Target is the RESIDUAL in capacity-factor space: y_cf − physics_cf."""
    residual = y_cf - physics_cf
    models = {}
    for a in QUANTILES:
        m = lgb.LGBMRegressor(**{**PARAMS, "alpha": a})
        m.fit(X, residual, sample_weight=sample_weight,
              categorical_feature=[c for c in X.columns if X[c].dtype.name == "category"])
        models[a] = m
    return models


def predict_residual(models, X, physics_cf, capacity_mw) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for a in QUANTILES:
        cf = (physics_cf + models[a].predict(X)).clip(0, 1)
        out[f"p{int(a*100)}_mw"] = cf * capacity_mw
    # quantile crossing is a real failure mode — enforce monotonicity
    cols = ["p10_mw", "p50_mw", "p90_mw"]
    out[cols] = np.sort(out[cols].values, axis=1)
    return out
```

Three decisions worth understanding:

**Train on capacity factor, not MW.** A 50 MW plant and a 9 GW region produce the same target range, so
one model generalises. Fleet growth is absorbed by `monitored_capacity_mw`.

**Train on the residual, not the raw target.** The residual is small and structured; the raw signal is
large and seasonal. This is the largest single accuracy gain for the least code.

**Sort the quantile outputs.** Independently fitted quantile models can cross (`p10 > p50`). Sorting is
the standard fix and costs nothing.

---

## 7. Conformal calibration — `uncertainty/conformal.py`

Quantile regression alone is **not calibrated**. This stage is what makes the interval a statement rather
than a decoration.

```python
import numpy as np, pandas as pd

class SplitConformal:
    """Conformalised quantile regression (CQR). Distribution-free, finite-sample coverage."""

    def __init__(self, alpha: float = 0.2):     # alpha=0.2 → nominal 80% interval
        self.alpha = alpha
        self.q_hat: dict[int, float] = {}

    def calibrate(self, lo, hi, y, lead_hours):
        """Fit one correction per lead hour — uncertainty grows with horizon,
        so a single global correction is too wide early and too narrow late."""
        df = pd.DataFrame({"lo": lo, "hi": hi, "y": y, "lh": lead_hours})
        for lh, g in df.groupby("lh"):
            # nonconformity: how far outside the interval each observation fell
            s = np.maximum(g.lo - g.y, g.y - g.hi)
            n = len(s)
            if n < 30:
                self.q_hat[int(lh)] = 0.0        # too few points; leave uncorrected
                continue
            k = int(np.ceil((n + 1) * (1 - self.alpha)))
            self.q_hat[int(lh)] = float(np.sort(s)[min(k, n) - 1])
        return self

    def apply(self, lo, hi, lead_hours):
        adj = np.array([self.q_hat.get(int(l), 0.0) for l in lead_hours])
        return lo - adj, hi + adj


def coverage_report(lo, hi, y, lead_hours, nominal=0.8) -> pd.DataFrame:
    df = pd.DataFrame({"lo": lo, "hi": hi, "y": y, "lh": lead_hours})
    df["inside"] = (df.y >= df.lo) & (df.y <= df.hi)
    r = df.groupby("lh").agg(picp=("inside", "mean"),
                             mean_width=("hi", lambda s: (s - df.loc[s.index, "lo"]).mean()),
                             n=("inside", "size")).reset_index()
    r["ace"] = r.picp - nominal          # target ≈ 0
    return r
```

Calibrate on a **rolling recent window** (the last 30–60 days), not on the training set. Calibrating on
data the model has seen produces an interval that is confidently wrong.

> **Publish the coverage plot.** An 80 % band that covers 55 % of outcomes is not a conservative forecast —
> it is a false statement, and it will size reserves wrong. This plot is fifteen seconds of screen time and
> almost nobody else will show one.

---

## 8. Evaluation harness — `evaluation/`

Build this **before** the second model exists.

```python
# evaluation/metrics.py
import numpy as np, pandas as pd

def nrmse(y, yhat, capacity):  return float(np.sqrt(np.mean((yhat - y) ** 2)) / capacity)
def nmae(y, yhat, capacity):   return float(np.mean(np.abs(yhat - y)) / capacity)
def mbe(y, yhat, capacity):    return float(np.mean(yhat - y) / capacity)

def skill_score(y, yhat, y_ref, capacity):
    return 1 - nrmse(y, yhat, capacity) / max(nrmse(y, y_ref, capacity), 1e-9)

def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))

def crps_from_quantiles(y, qs: dict[float, np.ndarray]):
    """Approximate CRPS by averaging pinball loss across the available quantiles."""
    return float(np.mean([pinball(y, v, a) for a, v in qs.items()]))

def picp(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))
```

```python
# evaluation/walk_forward.py
def walk_forward(df, train_months=6, test_months=1, gap_days=1):
    """Expanding-window folds with a gap. Never a random split — adjacent hours are
    near-identical, so a random split leaks and inflates every number."""
    start = df.index.min().normalize()
    end   = df.index.max()
    folds = []
    tr_end = start + pd.DateOffset(months=train_months)
    while tr_end + pd.DateOffset(months=test_months) <= end:
        te_start = tr_end + pd.Timedelta(days=gap_days)
        te_end   = te_start + pd.DateOffset(months=test_months)
        folds.append((df.loc[start:tr_end], df.loc[te_start:te_end]))
        tr_end += pd.DateOffset(months=test_months)
    return folds
```

```python
# evaluation/report.py
def per_lead_hour_report(preds: pd.DataFrame, capacity: float,
                         daylight_only: bool = True) -> pd.DataFrame:
    """preds: y_true, p10, p50, p90, persistence, physics, tso (optional), lead_hours, is_day"""
    d = preds[preds.is_day == 1] if daylight_only else preds
    rows = []
    for lh, g in d.groupby("lead_hours"):
        row = dict(
            lead_hours=int(lh), n_rows=len(g),
            nrmse_model=nrmse(g.y_true, g.p50, capacity),
            nrmse_persistence=nrmse(g.y_true, g.persistence, capacity),
            nrmse_physics=nrmse(g.y_true, g.physics, capacity),
            skill_score=skill_score(g.y_true, g.p50, g.persistence, capacity),
            mbe=mbe(g.y_true, g.p50, capacity),
            picp_80=picp(g.y_true, g.p10, g.p90),
            pinball_mean=np.mean([pinball(g.y_true, g[f"p{q}"], q / 100) for q in (10, 50, 90)]),
        )
        if "tso" in g:
            row["nrmse_tso"] = nrmse(g.y_true, g.tso, capacity)
        rows.append(row)
    return pd.DataFrame(rows)
```

### Non-negotiable evaluation rules

| Rule | Reason |
|---|---|
| Walk-forward with a gap, never random | Adjacent hours are near-identical; random splits leak |
| Report **per lead hour**, never one average | A 72-hour average hides where the skill actually is |
| **Daylight-only** for solar | Half a solar series is zeros — including them halves the error and means nothing |
| Normalise by **installed capacity** | Comparable across regions; mean-normalising lets low-output periods dominate |
| Exclude curtailed intervals | Censored labels, not weather outcomes |
| Match the TSO's **information cutoff** | Elia's forecast is issued at a fixed time; comparing against a different issue time is meaningless |

---

## 9. Handling curtailment

Curtailed hours are **censored labels** — measured output is not what the weather allowed. Training on them
teaches the model to under-forecast exactly during the events the platform exists to predict.

```python
# quality/curtailment.py
def detect_curtailment(actual_mw, physics_mw, capacity, kt, is_day) -> pd.Series:
    """Heuristic for sources that do not publish a curtailment flag: output flat-lining
    well below the physics estimate under conditions that should support more."""
    ratio    = actual_mw / physics_mw.clip(lower=capacity * 0.02)
    flatline = actual_mw.rolling(4).std() < capacity * 0.002
    good     = (kt > 0.6) & (is_day == 1)
    return (good & flatline & (ratio < 0.75)).astype("int8")


def sample_weights(df) -> pd.Series:
    w = pd.Series(1.0, index=df.index)
    w[df.get("curtailed_mw", 0) > 0] = 0.0
    w[df.get("qc_flag", "OK").isin(["MISSING", "FROZEN", "OUT_OF_RANGE", "CURTAILED"])] = 0.0
    w[df.get("availability_pct", 100) < 90] = 0.0
    return w
```

Tune the thresholds against a period you can eyeball. The heuristic will have false positives; that costs
a little training data, which is far cheaper than learning a biased relationship.

---

## 10. Experiment tracking

```python
import mlflow

with mlflow.start_run(run_name=f"residual-gbdt-{region}-{tech}"):
    mlflow.log_params({**PARAMS, "region": region, "tech": tech,
                       "n_features": X.shape[1], "target": "residual_cf"})
    models = train_residual(X_tr, y_tr, phys_tr, w_tr)
    report = per_lead_hour_report(preds, capacity)
    mlflow.log_metrics({
        "nrmse_mean":  report.nrmse_model.mean(),
        "nrmse_lh24":  report.loc[report.lead_hours == 24, "nrmse_model"].iat[0],
        "nrmse_lh48":  report.loc[report.lead_hours == 48, "nrmse_model"].iat[0],
        "skill_mean":  report.skill_score.mean(),
        "picp_mean":   report.picp_80.mean(),
    })
    mlflow.log_table(report, "per_lead_hour.json")
    for a, m in models.items():
        mlflow.lightgbm.log_model(m, f"q{int(a*100)}")
```

**Promotion gate — a candidate must beat the incumbent before it ships:**

```python
def promote(candidate_metrics, incumbent_metrics) -> bool:
    """Automatic retraining without a gate is how production models silently degrade —
    one bad data day produces a bad model that deploys unreviewed."""
    return (candidate_metrics["nrmse_mean"] < incumbent_metrics["nrmse_mean"] * 0.995
            and abs(candidate_metrics["picp_mean"] - 0.80) < 0.06)
```

---

## 11. Explainability

```python
import shap

def explain(model, X_row):
    sv = shap.TreeExplainer(model).shap_values(X_row)
    top = pd.Series(sv[0], index=X_row.columns).abs().nlargest(5)
    return {"drivers": top.index.tolist(), "contributions": top.round(4).tolist()}
```

Serve this on `/forecast?explain=true` for a selected hour. On a residual model the attribution is readable
— "low cloud and soiling" rather than an opaque 300-feature soup. That readability is a direct consequence
of the residual design, and it is worth saying so.

---

## 12. Expected results

| Metric | Target | Basis |
|---|---|---|
| Regional solar, day-ahead nRMSE (daylight-only) | 8–13 % | Published range for regional day-ahead is 10–20 % of capacity |
| Regional wind, day-ahead nRMSE | 10–15 % | Consistent with the 7–19 % single-farm MAE band, aggregated |
| Skill score vs persistence | > 0.40 | Persistence is very weak at 24–72 h |
| Gap to the TSO's published forecast | within 10–25 % relative | A professional system with proprietary inputs; parity is not the goal |
| PICP for the nominal 80 % band | 0.78 – 0.82 | Conformal gives a finite-sample guarantee |

---

## 13. What would falsify the design

Each is a measurable outcome of the harness, not a matter of opinion. Building the harness in phase 0 is
what makes it possible to be wrong in public and say so.

| Finding | What it means |
|---|---|
| Physics baseline does not beat persistence at 24–72 h | The physics-first thesis is wrong for this horizon; fall back to pure ML |
| Residual model does not beat direct GBDT on the raw target | Residual learning adds nothing here; simplify |
| Archetype weighting does not beat one average archetype | The regional refinement is over-engineering; remove it |
| Conformal intervals need > 50 % of capacity width | Possibly too uncertain at 72 h for reserve sizing; reduce the advertised horizon |

---

## 14. ML checklist

- [ ] `build_features` is pure — no I/O, no globals, identical in training and serving
- [ ] Target is capacity factor, not MW
- [ ] Model predicts the residual, not the raw target
- [ ] `sample_weight = 0` on curtailed, missing and unavailable intervals
- [ ] No autoregressive lag under 24 hours anywhere in the feature set
- [ ] Quantile outputs sorted to prevent crossing
- [ ] Conformal calibrated per lead hour, on a rolling recent window
- [ ] Walk-forward splits with a gap; no `train_test_split` in the repository
- [ ] Metrics reported per lead hour, daylight-only for solar
- [ ] Persistence and physics baselines on every chart
- [ ] Promotion gate in place before any automatic retraining
