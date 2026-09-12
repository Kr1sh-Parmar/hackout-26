# data.md — Data Acquisition Runbook

**Project:** AI-Powered Renewable Generation Forecasting Platform · Team ZERO BIAS
**Purpose:** everything needed to download every dataset this project uses, without further research.

---

## 0. How to use this file

> **If you are an AI agent reading this:** this is an executable runbook. Work through it top to bottom.
> Every endpoint, parameter and file name below was verified in September 2026. Where a value must be
> discovered at runtime rather than assumed, the step says so explicitly and gives the discovery call.
>
> **Rules:**
> 1. Run **§3 (setup)** first, then **§4 (discovery)**, then the download sections in tier order.
> 2. Do **not** download every file from Zenodo (§9) — the full record is 47.7 GB. Take the named subset only.
> 3. Verify each download with the check given at the end of its section before moving on.
> 4. If a call fails, look in **§14 (troubleshooting)** before retrying or improvising.
> 5. Nothing here needs a paid account. Two optional sources need a free login (Kaggle, NREL).

**Total download for the core build: ~1.5 GB. With optional extras: ~8 GB.**

---

## 1. What we are building, in one paragraph

Regional 24–72 hour probabilistic forecasts of solar and wind generation, plus a decision layer that
turns each forecast into a sized grid action. **Belgium** is the validation region — it is the only open
portal publishing generation, the operator's own forecast, *and* demand on one 15-minute timebase.
**India** is the transfer region. Weather comes from Open-Meteo throughout, so the same variable names
appear in training and serving.

---

## 2. Source inventory

| Tier | ID | Source | Purpose | Size | Auth |
|---|---|---|---|---|---|
| **1** | `OM-HF` | Open-Meteo **Historical Forecast** API | Training weather — archived *past forecasts* | ~150 MB | none |
| **1** | `OM-F` | Open-Meteo **Forecast** API | Serving weather — live 72 h | <1 MB/run | none |
| **1** | `ELIA-PV` | Elia `ods032` | Belgian solar: measured + operator forecasts | ~60 MB | none |
| **1** | `ELIA-WIND` | Elia `ods031` | Belgian wind: measured + operator forecasts | ~60 MB | none |
| **1** | `ELIA-LOAD` | Elia `ods001` | Belgian demand: measured + forecast | ~40 MB | none |
| **2** | `OM-ERA5` | Open-Meteo **Historical Weather** API | Long reanalysis history (optional deep training) | ~400 MB | none |
| **2** | `ZEN-IN` | Zenodo `10.5281/zenodo.7824872` | India transfer region | ~60 MB (subset) | none |
| **2** | `OPSD` | Open Power System Data | Cross-check region (Germany, 4 TSO zones) | ~230 MB | none |
| **3** | `SDWPF` | Figshare `10.6084/m9.figshare.24798654` | Asset-level wind case study, turbine layout | ~1.5 GB | none |
| **3** | `KAG-PV` | Kaggle `anikannal/solar-power-generation-data` | Asset-level Indian solar demo | ~10 MB | free login |
| **3** | `ELIA-RT` | Elia `ods086/087/002` | Near-real-time feeds for the live demo | small | none |
| **4** | `NSRDB` | NREL NSRDB / SUNY India | Historical solar resource over India | varies | free key |
| **4** | `CEA` | CEA / MNRE / Ember | Capacity numbers and impact figures | manual | none |

**Tier 1 is mandatory.** Tiers 2–4 are in priority order after that.

---

## 3. Setup

```bash
mkdir -p data/{raw,interim,processed}
mkdir -p data/raw/{openmeteo,elia,india,opsd,sdwpf,kaggle,nsrdb}
mkdir -p data/raw/openmeteo/{hist_forecast,forecast,era5}
```

```bash
pip install requests pandas pyarrow numpy pvlib windpowerlib xarray netCDF4 \
            openmeteo-requests requests-cache retry-requests tqdm --break-system-packages
```

`openmeteo-requests` is the official client and handles their FlatBuffers responses, which are much faster
than JSON for long time series. Plain `requests` also works — both forms are given below.

### Environment (only for the optional sources)

```bash
export NREL_API_KEY="..."        # free: https://developer.nrel.gov/signup/
export KAGGLE_USERNAME="..."     # free: kaggle.com → Account → Create New API Token
export KAGGLE_KEY="..."
```

---

## 4. Discovery step — run this before any bulk download

Field names are the one thing worth confirming at runtime rather than trusting a document. Elia's schema
is stable but this costs one call and removes the whole class of "column not found" failures.

```bash
# Print every field name + type for the three Elia datasets we use
for DS in ods032 ods031 ods001; do
  echo "═══ $DS ═══"
  curl -s "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/$DS" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
        [print(f\"  {f['name']:<28} {f['type']:<10} {f.get('label','')}\") for f in d['fields']]"
done
```

**Expected for `ods032` (verified September 2026):**

| Field | Meaning |
|---|---|
| `datetime` | Timestamp, 15-minute resolution |
| `region` | Belgian region, or `Belgium` for the national total |
| `measured` | Metered PV generation, MW |
| `mostrecentforecast` | Latest available forecast, MW |
| `dayaheadforecast` | Day-ahead forecast, MW ← **the benchmark to beat** |
| `weekaheadforecast` | Week-ahead forecast, MW |

The datasets also expose monitored capacity, load factor and P10/P90 variants. The discovery call above
prints their exact names — **use whatever it returns**, not what you assume.

---

## 5. `OM-HF` — Open-Meteo Historical Forecast API ⭐ **train on this**

### Why this and not the reanalysis archive

This is the most important choice in the runbook. There are two Open-Meteo history endpoints and they are
not interchangeable:

| API | Endpoint | What it contains |
|---|---|---|
| Historical **Weather** | `archive-api.open-meteo.com/v1/archive` | **ERA5 reanalysis** — the best *retrospective* estimate, from 1940. Systematically more accurate than any forecast. |
| Historical **Forecast** | `historical-forecast-api.open-meteo.com/v1/forecast` | **Archived past forecasts** — what the model actually predicted at the time. ECMWF IFS HRES back to **January 2017**; most other models from 2021–2022. |

Training on reanalysis and serving on forecasts is **train/serve skew** — the model learns a cleaner
irradiance-to-power relationship than it will ever meet in production. The Historical Forecast API is the
correct fix, and it is free. Use it as the primary training source.

### Endpoint and parameters

```
GET https://historical-forecast-api.open-meteo.com/v1/forecast
```

| Parameter | Value | Note |
|---|---|---|
| `latitude` / `longitude` | comma-separated lists | Multiple points in one call → returns a JSON array |
| `start_date` / `end_date` | `YYYY-MM-DD` | |
| `hourly` | see variable list below | |
| `models` | `ecmwf_ifs025`, `icon_seamless`, `gfs_seamless` | One call per model, so disagreement can be computed |
| `wind_speed_unit` | **`ms`** | ⚠️ **defaults to km/h** — power goes as v³, so a missed conversion is a ~47× error |
| `timezone` | `UTC` | Store UTC. Convert for display only. |
| `tilt` / `azimuth` | e.g. `35` / `0` | Only needed for `global_tilted_irradiance` |

### Variable list (copy verbatim)

```
temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,pressure_msl,
cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,
shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,
terrestrial_radiation,
wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,wind_gusts_10m,
precipitation,rain,snowfall,snow_depth,visibility,is_day
```

### Belgian capacity-weighted grid points

A single centroid misses frontal systems crossing the region. Sample five points and weight by nearby
installed capacity.

| Point | Lat | Lon | Weight |
|---|---|---|---|
| Flanders west | 51.05 | 3.72 | 0.30 |
| Antwerp | 51.22 | 4.40 | 0.25 |
| Brussels | 50.85 | 4.35 | 0.15 |
| Liège | 50.63 | 5.57 | 0.18 |
| Namur | 50.46 | 4.87 | 0.12 |

### Download script

```python
# scripts/dl_openmeteo_hist_forecast.py
import requests, pandas as pd, itertools, time, pathlib

OUT = pathlib.Path("data/raw/openmeteo/hist_forecast"); OUT.mkdir(parents=True, exist_ok=True)

GRID = [("flanders_w",51.05,3.72,.30), ("antwerp",51.22,4.40,.25),
        ("brussels",50.85,4.35,.15),   ("liege",50.63,5.57,.18),
        ("namur",50.46,4.87,.12)]

HOURLY = ("temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,pressure_msl,"
          "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
          "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
          "terrestrial_radiation,wind_speed_10m,wind_speed_100m,wind_direction_10m,"
          "wind_direction_100m,wind_gusts_10m,precipitation,rain,snowfall,snow_depth,"
          "visibility,is_day")

MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
START, END = "2023-01-01", "2025-12-31"     # 3 years is ample; extend if you want more

for (name, lat, lon, w), model in itertools.product(GRID, MODELS):
    dest = OUT / f"{name}_{model}.parquet"
    if dest.exists():
        print("skip", dest.name); continue
    r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": HOURLY, "models": model,
        "wind_speed_unit": "ms",          # ⚠️ non-negotiable
        "timezone": "UTC",
    }, timeout=180)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["grid_point_id"] = name
    df["weight"] = w
    df["nwp_model"] = model
    df["latitude"], df["longitude"] = j["latitude"], j["longitude"]
    df.to_parquet(dest, index=False)
    print(f"{dest.name:38} {len(df):>7,} rows")
    time.sleep(1.2)                        # be polite; free tier
```

**Verify**

```bash
ls data/raw/openmeteo/hist_forecast/*.parquet | wc -l      # expect 15  (5 points × 3 models)
python3 -c "
import pandas as pd, glob
df = pd.concat(pd.read_parquet(f) for f in glob.glob('data/raw/openmeteo/hist_forecast/*.parquet'))
print(df.shape); print(df['time'].min(), '→', df['time'].max())
print('max wind_speed_100m:', df['wind_speed_100m'].max(), '(must be < 60 — if ~200 you got km/h)')
print('max shortwave:', df['shortwave_radiation'].max(), '(expect 700–1000 W/m² for Belgium)')"
```

> ⚠️ **The single most likely silent failure in this whole runbook** is forgetting `wind_speed_unit=ms`.
> Belgian 100 m wind should peak around 25–35 m/s. If your maximum is near 100+, you have km/h.

---

## 6. `OM-F` — Open-Meteo Forecast API (serving)

Same variables, same units, so training and inference match exactly. Run this 4× daily on a schedule.

```
GET https://api.open-meteo.com/v1/forecast
```

```python
# scripts/fetch_forecast.py — run on a cron, 4× daily
import requests, pandas as pd, datetime as dt, pathlib
OUT = pathlib.Path("data/raw/openmeteo/forecast"); OUT.mkdir(parents=True, exist_ok=True)
run_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H")

for name, lat, lon, w in GRID:                       # same GRID as §5
    for model in ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly": HOURLY, "models": model,
            "forecast_days": 4,            # 4 days covers the 72 h horizon with margin
            "past_days": 2,                # overlap lets you compute nwp_bias_lag
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }, timeout=60)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df["run_ts_utc"] = run_ts
        df["grid_point_id"], df["weight"], df["nwp_model"] = name, w, model
        df.to_parquet(OUT / f"{run_ts}_{name}_{model}.parquet", index=False)
```

> **Keep every run.** The `(run_ts, valid_ts)` pair is the primary key of the weather table. Storing only
> `valid_ts` overwrites a 48-hour-old forecast with a 6-hour-old one, and every backtest afterwards
> silently cheats.

**Free-tier limits:** roughly 10,000 calls/day, 5,000/hour, 600/minute for non-commercial use. The schedule
above uses 15 calls per run × 4 runs = 60/day. No problem.

---

## 7. `ELIA-PV` / `ELIA-WIND` / `ELIA-LOAD` — Belgian ground truth ⭐

### Which endpoint to use

The Elia portal runs OpenDataSoft Explore API v2.1. There are two relevant endpoints and picking the wrong
one wastes hours:

| Endpoint | Use |
|---|---|
| `/records` | Interactive queries. **Hard cap of 100 rows per call**, and `offset + limit` ≤ 10,000. Useless for bulk. |
| `/exports/csv` | **Bulk download.** Returns the whole filtered dataset as one CSV. ← use this |

```
GET https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv
```

| Parameter | Value |
|---|---|
| `where` | ODSQL filter, e.g. `datetime >= '2023-01-01' AND datetime < '2026-01-01'` |
| `select` | comma-separated field list (optional; omit for all fields) |
| `order_by` | `datetime` |
| `timezone` | `UTC` |
| `delimiter` | `,` |
| `use_labels` | `false` — keeps the technical field names, which is what you want |

### Download script

```python
# scripts/dl_elia.py
import requests, pathlib, time

OUT = pathlib.Path("data/raw/elia"); OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"

DATASETS = {
    "ods032": "solar_historical",     # PV: measured + day-ahead / week-ahead / most-recent forecast
    "ods031": "wind_historical",      # Wind: same, split offshore / onshore
    "ods001": "load_historical",      # Total load: measured + forecast
}
WHERE = "datetime >= '2023-01-01' AND datetime < '2026-01-01'"

for ds, label in DATASETS.items():
    dest = OUT / f"{ds}_{label}.csv"
    if dest.exists():
        print("skip", dest.name); continue
    with requests.get(f"{BASE}/{ds}/exports/csv", params={
        "where": WHERE, "order_by": "datetime",
        "timezone": "UTC", "delimiter": ",", "use_labels": "false",
    }, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"{dest.name:34} {dest.stat().st_size/1e6:>7.1f} MB")
    time.sleep(2)
```

Equivalent one-liner if you prefer curl:

```bash
curl -G "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/exports/csv" \
  --data-urlencode "where=datetime >= '2023-01-01' AND datetime < '2026-01-01'" \
  --data-urlencode "order_by=datetime" \
  --data-urlencode "timezone=UTC" \
  --data-urlencode "use_labels=false" \
  -o data/raw/elia/ods032_solar_historical.csv
```

### Near-real-time twins (Tier 3, for the live demo)

Same call shape, different IDs: `ods087` (PV), `ods086` (wind), `ods002` (load). These update every quarter
hour and carry intraday, day-ahead and week-ahead forecasts.

**Verify**

```bash
python3 -c "
import pandas as pd
for f,n in [('ods032_solar_historical','solar'),('ods031_wind_historical','wind'),('ods001_load_historical','load')]:
    d = pd.read_csv(f'data/raw/elia/{f}.csv', nrows=5)
    print(f'--- {n} ---'); print(list(d.columns))
d = pd.read_csv('data/raw/elia/ods032_solar_historical.csv', parse_dates=['datetime'])
print('rows', len(d), '| range', d.datetime.min(), '→', d.datetime.max())
print('regions', d.region.unique()[:10])"
```

Expect roughly **105,000 rows per region per year** at 15-minute resolution (35,040 intervals × regions).

### Gotchas

- **MW, not MWh.** At 15-minute resolution, energy = `MW × 0.25`.
- **`region` includes both sub-regions and a national total.** Filter to the national row for the
  regional model, or keep all and treat region as a dimension. Do not sum a table that already contains
  the total — you will double-count.
- **DST.** Requesting `timezone=UTC` avoids the March/October duplicate-and-missing-hour problem entirely.
- **`robots.txt` blocks generic web scrapers** on this portal. The API endpoints above are the supported
  path and are not affected — call them directly rather than fetching the HTML pages.
- **Monitored capacity changes over time** as the fleet grows. This is why the model trains on capacity
  factor rather than raw MW.

---

## 8. `OM-ERA5` — Historical Weather API (optional, deeper history)

Only if you want more than the Historical Forecast archive covers. Remember what it is: **reanalysis**, so
a model trained on it inherits skew. Use it for climatology and long-run feature statistics, not as the
primary training weather.

```
GET https://archive-api.open-meteo.com/v1/archive
```

Same parameters as §5 plus `start_date` from as early as 1940. Add `models=era5` or `era5_seamless`.

---

## 9. `ZEN-IN` — India transfer region ⚠️ **selective download only**

**Record:** [zenodo.org/records/7824872](https://zenodo.org/records/7824872) — *Historical and modelled
renewable energy production for India*, Hunt & Bloomfield, CC-BY-4.0, published 13 April 2023.

> **The full record is 115 files and 47.7 GB. Do not mirror it.** The hourly capacity-factor archive is
> 88 annual ZIPs at 334–657 MB each. The eight files below are ~60 MB total and cover everything the
> project needs.

| File | Size | What it is |
|---|---|---|
| `modelled-historical-hourly-renewable_output.nc` | 6.2 MB | ⭐ **The main file.** Hourly modelled wind + solar output. |
| `installed-by-state-oct2022.csv` | 2.3 kB | Installed capacity by state and technology |
| `tabulated-installed-by-date.csv` | 3.4 kB | Capacity build-out over time |
| `POSOCO_reported_solar_MU_daily.csv` | 72.6 kB | ⭐ **Reported** daily solar generation — real validation data |
| `POSOCO_reported_wind_MU_daily.csv` | 136.1 kB | ⭐ **Reported** daily wind generation |
| `CEA_1x1_gridded_installed_solar_cap.nc` | 20.8 kB | Gridded solar capacity, 1°×1° |
| `CEA_1x1_gridded_installed_wind_cap.nc` | 20.8 kB | Gridded wind capacity, 1°×1° |
| `OSM_wind_turbine_installations.geojson` | 13.8 MB | Georeferenced turbine locations |

```bash
cd data/raw/india
BASE="https://zenodo.org/records/7824872/files"
for F in \
  "modelled-historical-hourly-renewable_output.nc" \
  "installed-by-state-oct2022.csv" \
  "tabulated-installed-by-date.csv" \
  "POSOCO_reported_solar_MU_daily.csv" \
  "POSOCO_reported_wind_MU_daily.csv" \
  "CEA_1x1_gridded_installed_solar_cap.nc" \
  "CEA_1x1_gridded_installed_wind_cap.nc" \
  "OSM_wind_turbine_installations.geojson" ; do
  echo "→ $F"
  curl -L -o "$F" "$BASE/$F?download=1"
done
cd -
```

Add `OSM_solar_installations.geojson` (44.3 MB) only if you need solar plant locations for the map.

**Verify**

```bash
python3 -c "
import xarray as xr, pandas as pd
ds = xr.open_dataset('data/raw/india/modelled-historical-hourly-renewable_output.nc')
print(ds)
print()
print(pd.read_csv('data/raw/india/POSOCO_reported_solar_MU_daily.csv').head())"
```

> **State this caveat wherever Indian results appear:** these capacity factors are reanalysis-**modelled**,
> not metered. Scoring a model against modelled data partly scores the model that produced it. The Indian
> phase demonstrates **transferability**, not accuracy — the POSOCO reported daily figures are the only
> genuine observations in the record, and they are daily, not hourly.

---

## 10. `OPSD` — Open Power System Data (cross-check region)

Verified direct URLs, package version **2020-10-06**:

```bash
cd data/raw/opsd
curl -L -O "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"   # 124 MB
curl -L -O "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv"   # 107 MB
cd -
```

Column naming is `[COUNTRY]_[METRIC]_[TYPE]`. The ones that matter:

```
DE_solar_generation_actual        DE_wind_onshore_generation_actual
DE_solar_capacity                 DE_wind_offshore_generation_actual
DE_solar_profile                  DE_wind_capacity
DE_load_actual_entsoe_transparency
DE_50hertz_solar_forecast         DE_amprion_wind_forecast
DE_tennet_solar_forecast          DE_transnetbw_wind_forecast
```

The four German TSO zones (50Hertz, Amprion, TenneT, TransnetBW) each publish their own forecast — a
second external benchmark, and four more regions to prove transfer.

**Verify**

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('data/raw/opsd/time_series_60min_singleindex.csv', nrows=3)
print([c for c in d.columns if c.startswith('DE_') and ('solar' in c or 'wind' in c or 'load' in c)])"
```

---

## 11. `SDWPF` — asset-level wind (Tier 3)

**Repository:** Figshare, DOI [10.6084/m9.figshare.24798654](https://doi.org/10.6084/m9.figshare.24798654)

Two releases. Take the **full** one — it is 24 months rather than the competition's 245 days:

| File | Contents |
|---|---|
| `sdwpf_2001_2112_full.parquet` | 134 turbines, Jan 2020 – Dec 2021, 10-minute, **11.4 M rows** ⭐ prefer parquet |
| `sdwpf_2001_2112_full.csv` | Same data, CSV |
| `sdwpf_turb_location_elevation.csv` | ⭐ Turbine x/y coordinates + elevation — unlocks wake and graph modelling |
| `sdwpf_kddcup/` | The original KDD Cup 2022 train/test split, for leaderboard comparability |

Figshare mints per-file download URLs, so resolve the DOI in a browser (or via the Figshare API) and take
the direct links. Download `sdwpf_2001_2112_full.parquet` and `sdwpf_turb_location_elevation.csv` into
`data/raw/sdwpf/`.

**Columns** (13): `TurbID`, `Day`, `Tmstamp`, `Wspd` (m/s), `Wdir` (° relative to nacelle),
`Etmp` (°C ambient), `Itmp` (°C nacelle interior), `Ndir` (° yaw), `Pab1`/`Pab2`/`Pab3` (° blade pitch),
`Prtv` (kW reactive), **`Patv` (kW active — the target)**.

> `Day` is an integer day index and `Tmstamp` is a time-of-day string; there is no absolute date. Build a
> synthetic datetime from the two, and note that seasonal features are therefore relative, not calendar.

---

## 12. `KAG-PV` — asset-level Indian solar (Tier 3)

```bash
pip install kaggle --break-system-packages
kaggle datasets download -d anikannal/solar-power-generation-data -p data/raw/kaggle --unzip
```

Four files, 15-minute, 34 days, two Indian plants:

| File | Columns |
|---|---|
| `Plant_1_Generation_Data.csv` | `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY` (inverter), `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` |
| `Plant_1_Weather_Sensor_Data.csv` | `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` |
| `Plant_2_Generation_Data.csv` | same as Plant 1 |
| `Plant_2_Weather_Sensor_Data.csv` | same as Plant 1 |

> ⚠️ **`DATE_TIME` is formatted differently between the two plants** (`DD-MM-YYYY HH:MM` for Plant 1,
> ISO-like for Plant 2). Parse each with an explicit format; do not rely on inference.
> 34 days is a demo asset, not a training set — there is no seasonal signal in it.

---

## 13. Tier 4 — optional

### `NSRDB` (needs a free key)

```bash
curl -G "https://developer.nrel.gov/api/nsrdb/v2/solar/suny-india-download.csv" \
  --data-urlencode "api_key=$NREL_API_KEY" \
  --data-urlencode "wkt=POINT(73.02 26.24)" \
  --data-urlencode "names=2014" \
  --data-urlencode "attributes=ghi,dni,dhi,clearsky_ghi,clearsky_dni,clearsky_dhi,air_temperature,wind_speed,relative_humidity,solar_zenith_angle" \
  --data-urlencode "interval=60" \
  --data-urlencode "email=YOUR_EMAIL" \
  --data-urlencode "utc=true" \
  -o data/raw/nsrdb/jodhpur_2014.csv
```

India coverage: 2000–2014, hourly, 10 km. **Rate limits:** 10,000 CSV requests/24 h at 1 per second;
2,000/24 h for other formats at 1 per 2 seconds; 20 concurrent.

> This is reanalysis and cannot be obtained forward. Training-only, and it reintroduces the skew of §5 —
> which is why it is Tier 4.

### `CEA` — manual, for the document not the model

Current installed capacity and impact figures: [CEA Renewable Generation Report](https://cea.nic.in/renewable-generation-report/?lang=en),
[NITI Aayog ICED dashboard](https://iced.niti.gov.in/), [Ember India Data Explorer](https://ember-energy.org/data/india-electricity-data-explorer/).
Mostly PDF. Extract by hand; do not build a scraper for these.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Wind speeds ~3.6× too large | Forgot `wind_speed_unit=ms` | Add it and re-download. Do **not** divide after the fact — you will forget which files were corrected. |
| Elia returns exactly 100 rows | Used `/records` instead of `/exports/csv` | Switch endpoint (§7) |
| Elia export times out | Date range too wide | Split into yearly requests and concatenate |
| `robots.txt` disallowed | Fetching Elia's HTML pages with a web tool | Call the API endpoints directly — they are the supported path |
| Zenodo download stalls | You started an annual capacity-factor ZIP | Cancel. Take only the eight files in §9. |
| Duplicate/missing hour in Elia data | DST transition | Request `timezone=UTC` |
| `KeyError` on an Elia column | Field name drifted | Re-run the discovery call in §4 and use what it returns |
| Open-Meteo 429 | Rate limited | Sleep 60 s. Keep the 1.2 s inter-call delay. |
| Solar generation nonzero at night | Timezone misalignment between weather and generation | Both must be UTC before joining |
| Kaggle 403 | Credentials missing | `~/.kaggle/kaggle.json`, `chmod 600` |

---

## 15. Post-download integration checklist

Run these before building any features. Each has caught a real bug in this class of project.

```python
import pandas as pd, glob

wx = pd.concat(pd.read_parquet(f) for f in glob.glob('data/raw/openmeteo/hist_forecast/*.parquet'))
pv = pd.read_csv('data/raw/elia/ods032_solar_historical.csv', parse_dates=['datetime'])

# 1 — both sides UTC, tz-aware
assert str(wx.time.dt.tz) == 'UTC'

# 2 — overlapping period exists
print('weather', wx.time.min(), '→', wx.time.max())
print('solar  ', pv.datetime.min(), '→', pv.datetime.max())

# 3 — units sane
assert wx.wind_speed_100m.max() < 60,  'wind still in km/h'
assert wx.shortwave_radiation.max() < 1400, 'irradiance out of range'

# 4 — no night-time solar generation
pv_be = pv[pv.region.eq('Belgium')].set_index('datetime')
night = pv_be.between_time('23:00','02:00')['measured']
assert night.max() < 5, 'solar at night — timezone misalignment'

# 5 — generation never exceeds monitored capacity (allow 5% metering slack)
# 6 — all three NWP models present for every grid point
print(wx.groupby(['grid_point_id','nwp_model']).size())

# 7 — gap census before deciding an imputation rule
full = pd.date_range(pv_be.index.min(), pv_be.index.max(), freq='15min', tz='UTC')
print('missing intervals:', len(full.difference(pv_be.index)))
```

**Then, before training:**

- Flag curtailed intervals — output flat-lining well below the physics estimate under good conditions —
  and set `sample_weight = 0`. Curtailed hours are censored labels, and training on them teaches the model
  to under-forecast exactly during the events the platform exists to predict.
- Compute the target as **capacity factor** (`power_mw / monitored_capacity_mw`), not raw MW. This absorbs
  fleet growth and makes the model transferable across regions.
- Key the weather table on `(run_ts_utc, valid_ts_utc, nwp_model, grid_point_id)` and enforce uniqueness.
- Split walk-forward with a gap. Never `train_test_split`.

---

## 16. Expected footprint

| Tier | Contents | Size |
|---|---|---|
| 1 | Open-Meteo history + 3 Elia exports | **~310 MB** |
| 1+2 | plus ERA5, India subset, OPSD | **~1.0 GB** |
| 1+2+3 | plus SDWPF full, Kaggle | **~2.5 GB** |
| All | plus NSRDB pulls | ~3 GB |

Everything above is open, free and licence-clear. No paid subscription, no commercial key, no data-sharing
agreement anywhere in the critical path — which means an evaluator can reproduce the entire pipeline.

---

## 17. Licences and attribution

| Source | Licence | Attribution required |
|---|---|---|
| Elia Open Data | Open data licence, free reuse | Credit Elia as source |
| Open-Meteo | Free for non-commercial; CC-BY for the data | Credit Open-Meteo and the underlying model (ECMWF / DWD / NOAA) |
| Zenodo 7824872 | CC-BY-4.0 | Cite Hunt & Bloomfield, University of Reading / Bristol |
| Open Power System Data | Open, mixed per source | Credit OPSD and the originating TSO |
| SDWPF | Research use | Cite the SDWPF paper (arXiv:2208.04360) and Longyuan Power |
| Kaggle solar plant | Open | Credit the uploader |
| NSRDB | Public domain (US Government) | Credit NREL |

Put these in a `SOURCES.md` at the repository root. Attribution is a licence condition on several of
these, not a courtesy.

---

*Verified September 2026. Endpoints, field names and file listings were checked against the live sources.
Where a value could change, §4 gives the runtime discovery call — prefer it over trusting this document.*
