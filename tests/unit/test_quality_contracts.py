"""The pandera contracts, and the boundaries they are actually enforced at.

These schemas were written, verified against the real silver files, and then
imported by nothing at all -- `pandera` was a dependency for dead code. They now
run at three boundaries: both silver writers, the live provider fetch, and the
feature builder's serving output. What follows pins the behaviour that makes
that safe to do.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.quality.schemas import SCHEMA_FOR_TABLE, generation_schema
from src.quality.validators import validate


def _generation(n: int = 4, **overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "region_id": ["BE"] * n,
            "ts_utc": pd.date_range("2026-09-10", periods=n, freq="15min", tz="UTC"),
            "tech": ["wind"] * n,
            "power_mw": [100.0] * n,
            "monitored_capacity_mw": [1000.0] * n,
            "qc_flag": ["OK"] * n,
            # an undeclared column, of the kind every real table carries
            "provider_note": ["whatever"] * n,
        }
    )
    return df.assign(**overrides)


def test_validate_never_reshapes_the_frame():
    """The trap that made this dangerous to wire in.

    Every schema is `strict="filter"`, so pandera's return value has the
    undeclared columns REMOVED. Validating at a write boundary and persisting
    the result would have silently stripped most of each silver table on its
    way to disk -- a schema is a check on the data, not a projection of it.
    """
    df = _generation()

    out = validate(df, generation_schema, "generation_actuals_wind", strict=True)

    assert list(out.columns) == list(df.columns)
    assert "provider_note" in out.columns
    pd.testing.assert_frame_equal(out, df)


def test_strict_raises_and_lenient_drops_only_the_bad_rows():
    df = _generation(n=4)
    df.loc[2, "power_mw"] = 5000.0  # 5x the monitored fleet

    with pytest.raises(SchemaErrors):
        validate(df, generation_schema, "generation_actuals_wind", strict=True)

    cleaned = validate(df, generation_schema, "generation_actuals_wind", strict=False)

    assert len(cleaned) == 3
    assert 5000.0 not in set(cleaned["power_mw"])
    assert list(cleaned.columns) == list(df.columns)


def test_parasitic_draw_is_data_and_a_sign_flip_is_not():
    """A turbine at zero wind DRAWS power -- yaw motors, blade heating -- so a
    small negative reading is a measurement. Measured on the built data: 452 of
    129,504 wind rows are negative, worst -0.45% of capacity, and the ingest
    pipeline already flags every one `OUT_OF_RANGE`. Rejecting them deleted rows
    the pipeline had deliberately marked and punched holes in a 15-minute series.
    The check that matters is the one that catches the whole feed inverting.
    """
    parasitic = _generation(n=1, power_mw=[-9.4], qc_flag=["OUT_OF_RANGE"])
    validate(parasitic, generation_schema, "wind", strict=True)  # must not raise

    sign_flip = _generation(n=1, power_mw=[-800.0])
    with pytest.raises(SchemaErrors):
        validate(sign_flip, generation_schema, "wind", strict=True)


def test_every_mapped_table_has_a_schema_that_is_actually_reachable():
    """`SCHEMA_FOR_TABLE` is what the build scripts look tables up in. A typo in
    a key is silent -- the table just stops being validated."""
    assert set(SCHEMA_FOR_TABLE) == {
        "weather_nwp",
        "generation_actuals_solar",
        "generation_actuals_wind",
    }
    for name, schema in SCHEMA_FOR_TABLE.items():
        assert schema is not None, name
        assert schema.strict == "filter", f"{name}: writers rely on filter, not raise-on-extra"
