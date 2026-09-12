# Dataset manifest

Generated from `data/`. 26 tables, 12,701,978 rows, 364.4 MB processed + 638.9 MB raw (49 files).

## Tables

| Table | Rows | Cols | Size | Time column | From | To |
|---|---:|---:|---:|---|---|---|
| `bronze/elia_load` | 129,504 | 10 | 8.2 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 23:45 |
| `bronze/elia_solar` | 1,812,944 | 18 | 82.4 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 21:45 |
| `bronze/elia_wind` | 647,520 | 21 | 23.1 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 23:45 |
| `bronze/india_modelled_hourly` | 384,240 | 5 | 6.4 MB | ts_utc | 1979-01-01 00:00 | 2022-10-31 23:00 |
| `bronze/opsd_15min` | 201,604 | 41 | 33.7 MB | ts_utc | 2014-12-31 23:00 | 2020-09-30 23:45 |
| `bronze/opsd_60min` | 50,401 | 42 | 4.9 MB | ts_utc | 2014-12-31 23:00 | 2020-09-30 23:00 |
| `bronze/weather_hist_forecast` | 485,640 | 32 | 10.2 MB | valid_ts_utc | 2023-01-01 00:00 | 2026-09-10 23:00 |
| `bronze/weather_previous_runs` | 242,880 | 87 | 13.1 MB | valid_ts_utc | 2024-03-05 00:00 | 2026-09-10 23:00 |
| `silver/generation_actuals_solar` | 129,496 | 9 | 2.1 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 21:45 |
| `silver/generation_actuals_wind` | 129,504 | 9 | 3.4 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 23:45 |
| `silver/generation_actuals_wind_segment` | 259,008 | 7 | 2.9 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 23:45 |
| `silver/india_gridded_capacity` | 188 | 5 | 0.0 MB | — | — | — |
| `silver/india_installed_by_date` | 70 | 7 | 0.0 MB | — | — | — |
| `silver/india_installed_by_state` | 38 | 9 | 0.0 MB | — | — | — |
| `silver/india_modelled_hourly` | 384,240 | 5 | 6.4 MB | ts_utc | 1979-01-01 00:00 | 2022-10-31 23:00 |
| `silver/india_posoco_daily` | 34,704 | 6 | 0.1 MB | date | 2012-04-01 00:00 | 2022-10-31 00:00 |
| `silver/load` | 129,504 | 8 | 6.5 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 23:45 |
| `silver/opsd_generation_15min` | 3,224,073 | 5 | 39.2 MB | ts_utc | 2015-01-01 00:15 | 2020-09-30 23:30 |
| `silver/opsd_generation_60min` | 806,029 | 5 | 3.5 MB | ts_utc | 2015-01-01 00:00 | 2020-09-30 23:00 |
| `silver/opsd_load_15min` | 1,007,990 | 4 | 18.7 MB | ts_utc | 2015-01-01 00:15 | 2020-09-30 23:30 |
| `silver/opsd_load_60min` | 252,000 | 4 | 2.4 MB | ts_utc | 2015-01-01 00:00 | 2020-09-30 23:00 |
| `silver/tso_forecast_solar` | 517,792 | 7 | 7.9 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 21:45 |
| `silver/tso_forecast_wind` | 514,633 | 7 | 8.5 MB | ts_utc | 2023-01-01 00:00 | 2026-09-10 21:45 |
| `silver/weather_nwp` | 1,214,280 | 37 | 18.0 MB | valid_ts_utc | 2023-01-01 00:00 | 2026-09-10 23:00 |
| `gold/training_base` | 98,616 | 128 | 43.7 MB | valid_ts_utc | 2023-01-01 00:00 | 2026-09-10 23:00 |
| `gold/training_base_24_72h` | 45,080 | 128 | 19.1 MB | valid_ts_utc | 2024-03-05 00:00 | 2026-09-10 23:00 |

## Raw files

| File | Size |
|---|---:|
| `raw/elia/ods001_load_historical_2023.csv` | 3.4 MB |
| `raw/elia/ods001_load_historical_2024.csv` | 3.5 MB |
| `raw/elia/ods001_load_historical_2025.csv` | 3.4 MB |
| `raw/elia/ods001_load_historical_2026.csv` | 2.4 MB |
| `raw/elia/ods031_wind_historical_2023.csv` | 25.1 MB |
| `raw/elia/ods031_wind_historical_2024.csv` | 24.7 MB |
| `raw/elia/ods031_wind_historical_2025.csv` | 24.7 MB |
| `raw/elia/ods031_wind_historical_2026.csv` | 17.2 MB |
| `raw/elia/ods032_solar_historical_2023.csv` | 64.1 MB |
| `raw/elia/ods032_solar_historical_2024.csv` | 64.6 MB |
| `raw/elia/ods032_solar_historical_2025.csv` | 65.0 MB |
| `raw/elia/ods032_solar_historical_2026.csv` | 46.0 MB |
| `raw/india/CEA_1x1_gridded_installed_solar_cap.nc` | 0.0 MB |
| `raw/india/CEA_1x1_gridded_installed_wind_cap.nc` | 0.0 MB |
| `raw/india/installed-by-state-oct2022.csv` | 0.0 MB |
| `raw/india/modelled-historical-hourly-renewable_output.nc` | 6.2 MB |
| `raw/india/OSM_wind_turbine_installations.geojson` | 13.8 MB |
| `raw/india/POSOCO_reported_solar_MU_daily.csv` | 0.1 MB |
| `raw/india/POSOCO_reported_wind_MU_daily.csv` | 0.1 MB |
| `raw/india/tabulated-installed-by-date.csv` | 0.0 MB |
| `raw/openmeteo/hist_forecast/antwerp_ecmwf_ifs025.parquet` | 0.9 MB |
| `raw/openmeteo/hist_forecast/antwerp_gfs_seamless.parquet` | 1.0 MB |
| `raw/openmeteo/hist_forecast/antwerp_icon_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/hist_forecast/brussels_ecmwf_ifs025.parquet` | 0.9 MB |
| `raw/openmeteo/hist_forecast/brussels_gfs_seamless.parquet` | 1.0 MB |
| `raw/openmeteo/hist_forecast/brussels_icon_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/hist_forecast/flanders_w_ecmwf_ifs025.parquet` | 0.9 MB |
| `raw/openmeteo/hist_forecast/flanders_w_gfs_seamless.parquet` | 1.0 MB |
| `raw/openmeteo/hist_forecast/flanders_w_icon_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/hist_forecast/liege_ecmwf_ifs025.parquet` | 0.9 MB |
| `raw/openmeteo/hist_forecast/liege_gfs_seamless.parquet` | 1.0 MB |
| `raw/openmeteo/hist_forecast/liege_icon_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/hist_forecast/namur_ecmwf_ifs025.parquet` | 0.9 MB |
| `raw/openmeteo/hist_forecast/namur_gfs_seamless.parquet` | 1.0 MB |
| `raw/openmeteo/hist_forecast/namur_icon_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/india/grid_points.json` | 0.0 MB |
| `raw/openmeteo/previous_runs/antwerp_ecmwf_ifs025.parquet` | 1.8 MB |
| `raw/openmeteo/previous_runs/antwerp_gfs_seamless.parquet` | 1.7 MB |
| `raw/openmeteo/previous_runs/antwerp_icon_seamless.parquet` | 1.8 MB |
| `raw/openmeteo/previous_runs/brussels_ecmwf_ifs025.parquet` | 1.8 MB |
| `raw/openmeteo/previous_runs/brussels_gfs_seamless.parquet` | 1.1 MB |
| `raw/openmeteo/previous_runs/brussels_icon_seamless.parquet` | 1.2 MB |
| `raw/openmeteo/previous_runs/flanders_w_ecmwf_ifs025.parquet` | 1.8 MB |
| `raw/openmeteo/previous_runs/flanders_w_gfs_seamless.parquet` | 1.7 MB |
| `raw/openmeteo/previous_runs/flanders_w_icon_seamless.parquet` | 1.8 MB |
| `raw/openmeteo/previous_runs/liege_ecmwf_ifs025.parquet` | 1.1 MB |
| `raw/openmeteo/previous_runs/liege_icon_seamless.parquet` | 1.1 MB |
| `raw/opsd/time_series_15min_singleindex.csv` | 112.1 MB |
| `raw/opsd/time_series_60min_singleindex.csv` | 130.3 MB |