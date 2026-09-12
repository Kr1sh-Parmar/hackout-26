"""Proves the replay path works with NO network at all: snapshot a gold cycle,
flip REPLAY_MODE on, block every socket, and confirm every table still reads
back non-empty data through `read_table`.
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

from src.core.config import get_settings
from src.ingest.replay import TABLES, load_replay, read_table, snapshot

REGION = "BE"
RUN_TS = pd.Timestamp("2026-09-09T00:00:00Z")


def _write_gold(gold_root, table: str, run_ts: pd.Timestamp) -> None:
    dest = gold_root / table / f"region_id={REGION}" / f"run_date={run_ts:%Y-%m-%d}"
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"region_id": [REGION], "run_ts_utc": [run_ts], "value": [1.0]})
    df.to_parquet(dest / "part-0.parquet", index=False)


class _NetworkTouchingStore:
    """Stands in for ParquetStore -- any call means replay mode failed to
    short-circuit before reaching the "live" path."""

    def __getattr__(self, name):
        def _fail(*a, **k):
            raise AssertionError(f"read_table hit the live store ({name}) while REPLAY_MODE=true")

        return _fail


@pytest.fixture
def blocked_network(monkeypatch):
    """Any attempt to open a socket fails the test immediately."""

    def _no_network(*a, **k):
        raise AssertionError("replay path touched the network")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)


def test_snapshot_and_replay_round_trip(tmp_path, monkeypatch, blocked_network):
    gold_root = tmp_path / "gold"
    for table in TABLES:
        _write_gold(gold_root, table, RUN_TS)

    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("REPLAY_MODE", "true")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.replay_mode is True

    written = snapshot(REGION, RUN_TS, gold_root=gold_root, settings=settings)
    assert set(written) == set(TABLES)

    store = _NetworkTouchingStore()
    for table in TABLES:
        df = read_table(REGION, table, store)
        assert not df.empty
        assert df["region_id"].iloc[0] == REGION

    get_settings.cache_clear()


def test_read_table_delegates_to_store_when_replay_mode_off(monkeypatch):
    monkeypatch.setenv("REPLAY_MODE", "false")
    get_settings.cache_clear()
    assert get_settings().replay_mode is False

    calls = []

    class _Store:
        def read_forecast(self, region_id):
            calls.append(region_id)
            return pd.DataFrame({"region_id": [region_id]})

    out = read_table(REGION, "forecast", _Store())
    assert calls == [REGION]
    assert out["region_id"].iloc[0] == REGION

    get_settings.cache_clear()


def test_load_replay_returns_empty_frame_when_nothing_snapshotted(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    settings = get_settings()

    out = load_replay(REGION, "forecast", settings)

    assert out.empty
    get_settings.cache_clear()
