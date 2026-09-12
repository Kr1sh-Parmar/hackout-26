"""FastAPI dependencies. No Redis -- the store is a lru_cached DuckDB-over-
parquet reader, which is all a read-mostly decision API needs."""

from __future__ import annotations

import functools

from ..core.config import Settings, get_settings as _get_settings
from ..core.store import ParquetStore


def get_settings() -> Settings:
    return _get_settings()


@functools.lru_cache
def get_store() -> ParquetStore:
    return ParquetStore(get_settings().data_root)
