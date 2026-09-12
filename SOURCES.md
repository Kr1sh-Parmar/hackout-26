# SOURCES.md — data provenance and attribution

Attribution is a **licence condition** on several of these sources, not a courtesy
(data.md §17). Everything below is open, free and licence-clear: no paid
subscription, no commercial key, no data-sharing agreement anywhere in the
critical path — an evaluator can reproduce the entire pipeline.

| Dataset | Source | Licence | Required attribution |
|---|---|---|---|
| Belgian PV generation + operator forecasts (`ods032`) | [Elia Open Data](https://opendata.elia.be) | Open data licence, free reuse | Credit **Elia** as source |
| Belgian wind generation + operator forecasts (`ods031`) | Elia Open Data | Open data licence, free reuse | Credit **Elia** as source |
| Belgian total load + forecasts (`ods001`) | Elia Open Data | Open data licence, free reuse | Credit **Elia** as source |
| Weather — archived past forecasts | [Open-Meteo Historical Forecast API](https://open-meteo.com) | Free non-commercial; data CC-BY | Credit **Open-Meteo** and the underlying models (**ECMWF**, **DWD ICON**, **NOAA GFS**) |
| Weather — previous model runs (lead-time stratified) | [Open-Meteo Previous Runs API](https://open-meteo.com) | Free non-commercial; data CC-BY | As above |
| Indian modelled hourly renewable output, POSOCO daily reported, CEA gridded capacity, OSM turbine locations | [Zenodo 10.5281/zenodo.7824872](https://zenodo.org/records/7824872) | CC-BY-4.0 | Cite **Hunt & Bloomfield**, University of Reading / Bristol (2023) |
| German generation, capacity and load, 4 TSO zones | [Open Power System Data](https://data.open-power-system-data.org/time_series/2020-10-06) | Open, mixed per source | Credit **OPSD** and the originating TSO |

## Not collected, and why

| Source | Tier | Status |
|---|---|---|
| `SDWPF` asset-level wind (Figshare) | 3 | **Not collected.** 1.5 GB; Figshare mints per-file download URLs that must be resolved interactively. Turbine-level case study, not needed for the regional pipeline. |
| `KAG-PV` Kaggle Indian solar plants | 3 | **Not collected.** Requires a free Kaggle login (`~/.kaggle/kaggle.json`). 34 days of data — a demo asset, not a training set. |
| `NSRDB` NREL India solar resource | 4 | **Not collected.** Requires a free NREL API key. Reanalysis, so it reintroduces train/serve skew — which is why data.md ranks it Tier 4. |
| `OM-ERA5` Open-Meteo reanalysis archive | 2 | **Deliberately skipped.** The Historical Forecast + Previous Runs archives already cover both the Belgian and Indian windows. ERA5 is reanalysis and would reintroduce the very skew the design is built to avoid. |
| `CEA` / MNRE / Ember capacity and impact figures | 4 | Manual PDF extraction, for the documents rather than the model. |

## Citations

- Hunt, E. & Bloomfield, H. (2023). *Historical and modelled renewable energy
  production for India.* Zenodo. https://doi.org/10.5281/zenodo.7824872
- Elia Group. *Open Data Portal.* https://opendata.elia.be
- Open-Meteo. *Free Weather API.* https://open-meteo.com
- Open Power System Data. *Time series package, version 2020-10-06.*
