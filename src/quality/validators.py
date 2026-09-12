"""Run a pandera schema against a frame, with a policy for what to do about
rows that fail: raise (ingestion should stop) or drop-and-warn (serve degraded
rather than not at all).
"""

from __future__ import annotations

import pandas as pd
import structlog
from pandera.errors import SchemaErrors

logger = structlog.get_logger(__name__)


def validate(df: pd.DataFrame, schema, name: str, strict: bool = True) -> pd.DataFrame:
    """Validate `df` against `schema`.

    Args:
        df: the frame to check.
        schema: a pandera DataFrameSchema.
        name: label for log lines, e.g. "weather_nwp".
        strict: True re-raises on any failure. False logs a warning and
            returns `df` with the failing rows dropped.

    Returns:
        The caller's frame -- with the failing rows dropped when strict=False,
        and otherwise unchanged. Deliberately NOT pandera's return value: these
        schemas are `strict="filter"`, so validating and writing the result
        would silently strip every undeclared column from the table on its way
        to disk. The schema is a CHECK on the data, not a projection of it.
    """
    try:
        schema.validate(df, lazy=True)
        return df
    except SchemaErrors as exc:
        cases = exc.failure_cases
        logger.warning(
            "schema_validation_failed",
            name=name,
            n_failures=len(cases),
            strict=strict,
            sample=cases.head(10).to_dict("records"),
        )
        if strict:
            raise
        bad_idx = pd.Index(cases["index"].dropna().unique())
        cleaned = df.drop(index=bad_idx.intersection(df.index))
        logger.warning("dropped_failing_rows", name=name, n_dropped=len(df) - len(cleaned))
        return cleaned
