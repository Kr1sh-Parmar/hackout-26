"""Pandera schemas for the three tables the decision/API layer trusts.

Real files carry more columns than the contract cares about (open-meteo raw
fields, feature-engineering intermediates) -- `strict="filter"` drops what we
did not declare instead of erroring on it, which is what lets these schemas
validate the actual silver/gold parquet rather than a hand-typed fixture.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

NWP_MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
QC_FLAGS = ["OK", "MISSING", "FROZEN", "OUT_OF_RANGE", "CURTAILED"]

weather_nwp_schema = DataFrameSchema(
    {
        "region_id": Column(str),
        "run_ts_utc": Column("datetime64[ns, UTC]"),
        "valid_ts_utc": Column("datetime64[ns, UTC]"),
        "lead_hours": Column(int, Check.in_range(0, 168)),
        "forecast_vintage": Column(str, nullable=True, required=False),
        "run_ts_is_approx": Column(bool, nullable=True, required=False),
        "nwp_model": Column(str, Check.isin(NWP_MODELS)),
        "grid_point_id": Column(str),
        "weight": Column(float, Check.in_range(0, 1)),
        "latitude": Column(float, nullable=True, required=False),
        "longitude": Column(float, nullable=True, required=False),
        "elevation_m": Column(float, nullable=True, required=False),
        "ghi_wm2": Column(float, Check.in_range(0, 1400), nullable=True),
        "dni_wm2": Column(float, Check.in_range(0, 1100), nullable=True),
        "dhi_wm2": Column(float, Check.in_range(0, 700), nullable=True),
        "direct_radiation_wm2": Column(float, nullable=True, required=False),
        "terrestrial_radiation_wm2": Column(float, nullable=True, required=False),
        # nullable: a handful of hours near each model's forecast horizon edge
        # (mostly gfs_seamless) come back null from open-meteo -- verified against
        # the real silver file rather than assumed.
        "temperature_2m_c": Column(float, Check.in_range(-60, 60), nullable=True),
        "dew_point_2m_c": Column(float, nullable=True, required=False),
        "relative_humidity_2m_pct": Column(
            float, Check.in_range(0, 100), nullable=True, required=False
        ),
        "surface_pressure_hpa": Column(float, Check.in_range(800, 1100), nullable=True),
        "pressure_msl_hpa": Column(float, nullable=True, required=False),
        # observed range is -1..101, not 0..100 -- provider rounding noise at the
        # clear/overcast extremes, not a real out-of-bounds value
        "cloud_cover_pct": Column(float, Check.in_range(-1, 101), nullable=True, required=False),
        "cloud_cover_low_pct": Column(
            float, Check.in_range(-1, 101), nullable=True, required=False
        ),
        "cloud_cover_mid_pct": Column(
            float, Check.in_range(-1, 101), nullable=True, required=False
        ),
        "cloud_cover_high_pct": Column(
            float, Check.in_range(-1, 101), nullable=True, required=False
        ),
        "wind_speed_10m_ms": Column(float, Check.in_range(0, 90), nullable=True),
        "wind_speed_100m_ms": Column(float, Check.in_range(0, 110), nullable=True),
        "wind_direction_10m_deg": Column(
            float, Check.in_range(0, 360), nullable=True, required=False
        ),
        "wind_direction_100m_deg": Column(float, Check.in_range(0, 360), nullable=True),
        "wind_gusts_10m_ms": Column(float, nullable=True, required=False),
        "precipitation_mm": Column(float, Check.ge(0), nullable=True, required=False),
        "rain_mm": Column(float, Check.ge(0), nullable=True, required=False),
        "snowfall_cm": Column(float, Check.ge(0), nullable=True, required=False),
        "snow_depth_m": Column(float, Check.ge(0), nullable=True, required=False),
        "visibility_m": Column(float, Check.ge(0), nullable=True, required=False),
        "is_day": Column(int, Check.isin([0, 1])),
    },
    unique=["run_ts_utc", "valid_ts_utc", "nwp_model", "grid_point_id"],
    strict="filter",
    coerce=True,
)

generation_schema = DataFrameSchema(
    {
        "region_id": Column(str),
        "ts_utc": Column("datetime64[ns, UTC]"),
        "tech": Column(str),
        # nullable: qc_flag=MISSING rows (and a handful of un-backfilled
        # monitored_capacity_mw rows) genuinely have no reading -- verified
        # against the real silver file. The `>= 0` check stays strict: small
        # negative readings do occur (self-consumption noise) and are exactly
        # what validate(..., strict=False) is for -- drop them, don't allow them.
        "power_mw": Column(float, Check.ge(0), nullable=True),
        "monitored_capacity_mw": Column(float, Check.gt(0), nullable=True),
        "load_factor": Column(float, nullable=True, required=False),
        "curtailed_mw": Column(float, nullable=True, required=False),
        "availability_pct": Column(float, nullable=True, required=False),
        "qc_flag": Column(str, Check.isin(QC_FLAGS)),
    },
    checks=Check(
        lambda df: (df["power_mw"] <= df["monitored_capacity_mw"] * 1.05)
        | df["power_mw"].isna()
        | df["monitored_capacity_mw"].isna(),
        element_wise=False,
        error="power_mw exceeds monitored_capacity_mw by more than 5%",
    ),
    unique=["region_id", "ts_utc", "tech"],
    strict="filter",
    coerce=True,
)

# `y_capacity_factor`/`clearsky_index_kt` are the feature-layer's names, not the
# gold demo table's (`y_solar_cf`/`y_wind_cf`) -- this schema is the training
# matrix contract owned by src/features/build.py, so columns are marked
# required=False and this is validated against synthetic rows, not the gold
# parquet, in tests/unit.
training_matrix_schema = DataFrameSchema(
    {
        "y_capacity_factor": Column(float, Check.in_range(0, 1.05), nullable=True, required=False),
        "sample_weight": Column(float, Check.in_range(0, 1), required=False),
        "lead_hours": Column(int, Check.in_range(1, 168), required=False),
        # upper bound above 1.0 is deliberate -- cloud-edge enhancement is real
        "clearsky_index_kt": Column(float, Check.in_range(0, 1.35), nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)

__all__ = ["weather_nwp_schema", "generation_schema", "training_matrix_schema"]
