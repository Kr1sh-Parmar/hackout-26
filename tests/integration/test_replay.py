"""Proves the replay path works with NO network at all: snapshot a gold cycle,
flip REPLAY_MODE on, block every socket, and confirm every table still reads
back non-empty data through `read_table`.
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

from src.core.config import get_settings
from src.ingest.replay import STATIC_TABLES, TABLES, load_replay, read_table, snapshot

REGION = "BE"
RUN_TS = pd.Timestamp("2026-09-09T00:00:00Z")


def _write_gold(gold_root, table: str, run_ts: pd.Timestamp) -> None:
    dest = gold_root / table / f"region_id={REGION}" / f"run_date={run_ts:%Y-%m-%d}"
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"region_id": [REGION], "run_ts_utc": [run_ts], "value": [1.0]})
    df.to_parquet(dest / "part-0.parquet", index=False)


def _contract_row(table: str, run_ts: pd.Timestamp) -> dict:
    """One row carrying every column the API's response model requires.

    Built from the same column constants the producers export, so if a contract
    grows a field this fixture follows it instead of quietly under-filling.
    """
    from src.decisions.events import EVENT_COLUMNS
    from src.evaluation.drift import DRIFT_COLUMNS
    from src.decisions.net_load import OUTLOOK_COLUMNS
    from src.decisions.recommend import Action, Flag, RECOMMEND_COLUMNS
    from src.models.explain import EXPLAIN_COLUMNS
    from src.models.predict import PREDICT_COLUMNS

    columns = {
        "forecast": PREDICT_COLUMNS,
        "outlook": OUTLOOK_COLUMNS,
        "events": EVENT_COLUMNS,
        "actions": RECOMMEND_COLUMNS,
        "explain": EXPLAIN_COLUMNS,
        "drift": DRIFT_COLUMNS,
    }[table]

    row: dict = {"region_id": REGION, "run_ts_utc": run_ts}
    for col in columns:
        if col in row:
            continue
        if col.endswith(("_ts_utc", "_from", "_to")):
            row[col] = run_ts
        elif col == "tech":
            row[col] = "solar"
        elif col == "flag":
            row[col] = list(Flag)[0].value
        elif col == "action":
            row[col] = list(Action)[0].value
        elif col == "rank":
            row[col] = 0  # top driver; the /explain route filters on rank < top_n
        elif col in ("lead_hours", "severity", "n_obs"):
            row[col] = 24
        elif col == "decisive":
            row[col] = True
        elif col in ("event_id", "linked_event_id", "feature"):
            row[col] = "x"
        elif col in ("description", "rationale", "model_version", "calibration_date"):
            row[col] = "x"
        elif col in ("calibrated", "retrain"):
            row[col] = True
        elif col in ("reason", "drifted_features"):
            row[col] = "x"
        else:
            row[col] = 0.0
    return row


def _write_contract_gold(gold_root, table: str, run_ts: pd.Timestamp) -> None:
    dest = gold_root / table / f"region_id={REGION}" / f"run_date={run_ts:%Y-%m-%d}"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_contract_row(table, run_ts)]).to_parquet(dest / "part-0.parquet", index=False)


def _write_backtest_gold(gold_root) -> None:
    """`backtest` is not partitioned by run -- flat files, one per tech, exactly
    as `scripts/backtest.py` writes them."""
    dest = gold_root / "backtest"
    dest.mkdir(parents=True, exist_ok=True)
    for tech in ("solar", "wind"):
        pd.DataFrame(
            {
                "lead_hours": [24, 48],
                "nrmse_model": [0.05, 0.06],
                "picp_80": [0.79, 0.80],
                "region_id": [REGION, REGION],
                "tech": [tech, tech],
            }
        ).to_parquet(dest / f"region_id={REGION}_tech={tech}.parquet", index=False)


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

    _write_backtest_gold(gold_root)
    written = snapshot(REGION, RUN_TS, gold_root=gold_root, settings=settings)
    assert set(written) == set(TABLES) | set(STATIC_TABLES)

    store = _NetworkTouchingStore()
    for table in TABLES + STATIC_TABLES:
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


def test_api_serves_the_snapshot_when_gold_is_gone(tmp_path, monkeypatch):
    """The integrity claim, end to end: with `data/gold` pointed at an empty
    directory, every operational endpoint still answers from the frozen copy.

    Before the routes went through `read_table`, REPLAY_MODE only flipped a flag
    in /health -- the endpoints read gold regardless, so replay "worked" on a
    machine that happened to have gold and would have served nothing on one that
    did not. That is exactly the failure this mode exists to prevent.
    """
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import app

    gold_root = tmp_path / "gold"
    for table in TABLES:
        _write_contract_gold(gold_root, table, RUN_TS)
    _write_backtest_gold(gold_root)

    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("REPLAY_MODE", "true")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "nothing-here"))
    get_settings.cache_clear()
    deps.get_store.cache_clear()
    try:
        snapshot(REGION, RUN_TS, gold_root=gold_root, settings=get_settings())
        client = TestClient(app)
        # /storage/sweep and /backtest are in this list on purpose: they were the
        # last two routes reading the store directly, so with gold gone they
        # returned an empty payload -- a demo showing a forecast beside a blank
        # accuracy panel and a flat sizing curve, with no error to explain it.
        paths = [
            ("/forecast", {}),
            ("/outlook", {}),
            ("/events", {}),
            ("/actions", {}),
            ("/explain", {}),
            ("/storage/sweep", {}),
            ("/backtest", {"tech": "solar"}),
        ]
        for path, extra in paths:
            r = client.get(path, params={"region_id": REGION, **extra})
            assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text}"
            body = r.json()
            assert body["replay_mode"] is True, path
            assert body["data"], f"{path} served nothing from the snapshot"
            # Provenance must name the snapshot's run, not ambient gold state.
            assert body["issued_at"].startswith("2026-09-09"), path
    finally:
        get_settings.cache_clear()
        deps.get_store.cache_clear()


def test_load_replay_returns_empty_frame_when_nothing_snapshotted(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    settings = get_settings()

    out = load_replay(REGION, "forecast", settings)

    assert out.empty
    get_settings.cache_clear()
