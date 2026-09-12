# Dataset — ZERO BIAS renewable forecasting platform

Built by `scripts/` from the sources in `data.md`. Every table is Parquet; read it
with pandas or query it in place with DuckDB. Nothing here needs credentials.

Run `python scripts/verify_dataset.py` to re-check the whole thing (36 checks).

---

## Layout

```
data/
├── raw/        exactly as downloaded — never modified, never written to by code
├── bronze/     parsed to Parquet, typed, UTC. No business logic.
├── silver/     validated, deduplicated, quality-flagged, nationally aggregated
├── gold/       analysis-ready joined matrix — what a model trains on
└── _reports/   manifest.md + manifest.json (full table/column inventory)
```

`raw/` is append-only. If a transform is wrong, fix the transform and rebuild —
never edit raw. That is what makes the pipeline reproducible from scratch.

## Rebuild from scratch

```bash
python scripts/dl_elia.py                      # Belgian truth + TSO benchmark
python scripts/dl_openmeteo_hist_forecast.py   # weather, day-0 vintage
python scripts/dl_openmeteo_previous_runs.py   # weather, 24-72 h vintages
python scripts/dl_tier2.py                     # India (Zenodo) + Germany (OPSD)
python scripts/build_elia.py
python scripts/build_weather.py
python scripts/build_india.py
python scripts/build_opsd.py
python scripts/build_gold.py
python scripts/verify_dataset.py
python scripts/make_manifest.py
```

Every download script is resumable — it skips files that already exist.

---

## The one thing to understand: forecast vintage

`silver/weather_nwp` and `gold/training_base` carry a `forecast_vintage` column.
It is not metadata; it decides what the row may be used for.

| Vintage | Lead hours | Source | Use for |
|---|---|---|---|
| `day0_best` | 0–23 | Historical Forecast API | Reference and bias estimation **only** |
| `prev_day1` | 24–47 | Previous Runs API | **Training and evaluation** |
| `prev_day2` | 48–71 | Previous Runs API | **Training and evaluation** |
| `prev_day3` | 72–95 | Previous Runs API | Training (only lead=72 falls in scope) |

**Train the 24–72 h model on `prev_day*` rows only.** `gold/training_base_24_72h`
is pre-filtered to exactly that. Using `day0_best` as training data would mean
learning from ~6 h-lead weather and serving at 48 h — train/serve skew, the same
class of bug the design docs name for reanalysis.

`run_ts_is_approx` is `True` on `day0_best` (Open-Meteo does not publish which run
produced the stitched series, so `run_ts` is taken as that day's 00Z) and `False`
on every `prev_day*` row, where the run time is real.

---

## Primary keys

| Table | Key |
|---|---|
| `silver/weather_nwp` | `(run_ts_utc, valid_ts_utc, nwp_model, grid_point_id)` |
| `silver/generation_actuals_*` | `(region_id, ts_utc, tech)` |
| `gold/training_base` | `(run_ts_utc, valid_ts_utc, forecast_vintage)` |

Uniqueness on the weather key is enforced and verified. Storing only `valid_ts`
would overwrite a 48-hour-old forecast with a 6-hour-old one and make every
later backtest silently cheat.

## Conventions

- **Every timestamp is UTC and timezone-aware.** Convert at the presentation layer only.
- **Every column name carries its unit**: `power_mw`, `ghi_wm2`, `wind_speed_100m_ms`.
- **Wind speed is m/s**, never km/h (`wind_speed_unit=ms` on every request, asserted
  in the verifier). Power goes as v³, so this is a ~47× error if missed.
- **Cloud layers stay separate** (`low`/`mid`/`high`). Low cloud destroys PV output;
  cirrus barely dents it.
- **The training target is capacity factor**, not MW — `monitored_capacity_mw`
  changes as the fleet grows.

---

## Findings that differ from `data.md`

These were discovered by running the runbook. `data.md` should be corrected.

1. **The Historical Forecast API does not give lead time.** It returns the best
   available forecast per hour (~0–23 h lead). A 24–72 h model cannot be trained on
   it honestly. Fixed by adding the **Previous Runs API** (`dl_openmeteo_previous_runs.py`),
   which is not mentioned in `data.md`. Its archive starts **2024-03-05**.

2. **`ods031` (wind) has no `Belgium` national row.** The national total must be
   **summed** over 5 `(region, offshore/onshore, gridconnectiontype)` combos.
   `ods032` (solar) is the opposite: it *does* have a `Belgium` row, plus 3 regions
   and 10 provinces, so summing it would triple-count. Both rules are verified —
   all 5 wind segments are present at every one of the 129,504 intervals.

3. **Elia publishes two day-ahead vintages**, not one: `dayaheadforecast` (6 PM
   cutoff) and `dayahead11hforecast` (11 AM cutoff), each with its own P10/P90.
   `silver/tso_forecast_*` keeps all four horizons long-format so the benchmark can
   be matched to the right information cutoff.

4. **Elia's CSV export writes empty text fields as `''`** — the two-character string,
   not null and not empty. A `.notna()` test on `decrementalbidid` flags 99.6 % of the
   wind fleet as curtailed. With quotes stripped the real rate is **2.96 %**.

5. **`decrementalbidid` is a genuine curtailment flag** for wind (downward redispatch).
   The risk register rated "curtailment not flagged in source" as *high likelihood*;
   for Belgian wind it is partly solved. PV still has no flag and needs the
   physics-based heuristic at the feature stage.

6. **OPSD has no per-TSO generation forecast columns** in the 2020-10-06 release,
   contrary to `data.md` §10 — only *load* forecasts. Germany is a second generation
   cross-check region, **not** a second forecast benchmark. Elia remains the only
   external generation-forecast benchmark in the project.

7. **The Indian `modelled_grid_output` is a single combined series**, not separable
   into solar and wind, and it ends **2022-10-31**.

8. **Open-Meteo CSV/JSON pitfalls:** Elia CSVs carry a UTF-8 BOM (read with
   `utf-8-sig`), and `Series.values` on a tz-aware datetime silently strips the
   timezone — both produced real bugs that the verifier caught.

---

## Known limitations

- **India cannot be modelled from this dataset as delivered.** The Zenodo series ends
  2022-10-31 and the lead-stratified weather archive starts 2024-03-05, so they do
  not overlap. India weather was **not** downloaded (it is not part of `data.md`); a
  fetcher is ready at `scripts/dl_openmeteo_india.py` with grid points derived from
  the CEA capacity raster, but it can only ever produce a day-0 transfer demo.
- **Indian output is reanalysis-modelled, not metered.** Scoring against it partly
  scores the model that produced it. India demonstrates transfer, never accuracy.
  The POSOCO daily figures are the only genuine observations in that record.
- **PV curtailment is unflagged.** Belgian PV has no equivalent of `decrementalbidid`.
- **Small negative wind readings** (452 rows, min −24.36 MW, −0.4 % of capacity) are
  real parasitic consumption. They are kept raw and flagged `OUT_OF_RANGE` rather
  than silently rewritten.
- **Tier-3/4 sources not collected**: SDWPF, Kaggle, NSRDB. See `SOURCES.md`.
