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


# --- The features the console depends on: runs, acknowledgements, the headline ---

RUN_TS_OLD = pd.Timestamp("2026-09-08T12:00:00Z")


def _write_scored_backtest_gold(gold_root) -> None:
    """A backtest table with the columns `summarise` weights by, as backtest.py writes it."""
    dest = gold_root / "backtest"
    dest.mkdir(parents=True, exist_ok=True)
    for tech in ("solar", "wind"):
        pd.DataFrame(
            {
                "lead_hours": [48, 24, 36],
                "n_rows": [300, 100, 0],
                "nrmse_model": [0.06, 0.05, None],
                "nrmse_persistence": [0.12, 0.10, None],
                "picp_80": [0.80, 0.70, None],
                "region_id": [REGION] * 3,
                "tech": [tech] * 3,
                "folds": [22] * 3,
            }
        ).to_parquet(dest / f"region_id={REGION}_tech={tech}.parquet", index=False)


@pytest.fixture
def replay_api(tmp_path, monkeypatch):
    """Two frozen cycles and nothing in gold: the API must serve both, distinctly."""
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import app

    gold_root = tmp_path / "gold"
    for run in (RUN_TS_OLD, RUN_TS):
        for table in TABLES:
            _write_contract_gold(gold_root, table, run)
    _write_scored_backtest_gold(gold_root)

    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("REPLAY_MODE", "true")
    get_settings.cache_clear()
    deps.get_store.cache_clear()
    try:
        for run in (RUN_TS_OLD, RUN_TS):
            snapshot(REGION, run, gold_root=gold_root, settings=get_settings())
        yield TestClient(app)
    finally:
        get_settings.cache_clear()
        deps.get_store.cache_clear()


def test_runs_lists_every_frozen_cycle_newest_first(replay_api):
    body = replay_api.get("/runs", params={"region_id": REGION}).json()
    assert [pd.Timestamp(r) for r in body["runs"]] == [RUN_TS, RUN_TS_OLD]
    assert pd.Timestamp(body["latest"]) == RUN_TS
    assert body["replay_mode"] is True


def test_pinned_run_serves_that_snapshot_not_the_latest(replay_api):
    """Replay used to ignore `run_ts` outright: a run picker would have put the old
    run's label over the newest run's numbers."""
    for run in (RUN_TS_OLD, RUN_TS):
        for path in ("/forecast", "/outlook", "/events", "/actions", "/explain", "/storage/sweep"):
            r = replay_api.get(path, params={"region_id": REGION, "run_ts": run.isoformat()})
            assert r.status_code == 200, f"{path} @ {run} -> {r.status_code}: {r.text}"
            assert pd.Timestamp(r.json()["issued_at"]) == run, path
    fc = replay_api.get("/forecast", params={"region_id": REGION, "run_ts": RUN_TS_OLD.isoformat()})
    assert pd.Timestamp(fc.json()["data"][0]["valid_ts_utc"]) == RUN_TS_OLD

    # Filters that leave no rows must not relabel the response as the latest run:
    # these rows sit at lead 24, so a 1 h horizon returns nothing.
    empty = replay_api.get(
        "/forecast",
        params={"region_id": REGION, "run_ts": RUN_TS_OLD.isoformat(), "horizon_hours": 1},
    )
    assert empty.json()["data"] == []
    assert pd.Timestamp(empty.json()["issued_at"]) == RUN_TS_OLD


def test_unknown_run_is_404_naming_the_runs_that_exist(replay_api):
    for path in ("/forecast", "/actions", "/storage/sweep"):
        r = replay_api.get(path, params={"region_id": REGION, "run_ts": "2020-01-01T00:00:00Z"})
        assert r.status_code == 404, path
        assert r.json()["error"] == "RunNotFound"
        assert "2026-09-09" in r.json()["hint"], path


def test_acknowledgement_round_trip_is_persisted_server_side(replay_api, tmp_path):
    action = replay_api.get("/actions", params={"region_id": REGION}).json()["data"][0]
    assert action["action_id"] and action["acknowledged_at"] is None

    url = f"/actions/{action['action_id']}/ack"
    first = replay_api.post(url, params={"region_id": REGION})
    assert first.status_code == 200, first.text
    acked_at = pd.Timestamp(first.json()["acknowledged_at"])
    # Idempotent: a double click must not move the acknowledgement time.
    assert (
        pd.Timestamp(replay_api.post(url, params={"region_id": REGION}).json()["acknowledged_at"])
        == acked_at
    )
    assert (tmp_path / "data" / "ops" / "acks" / f"{REGION}.json").exists()

    served = replay_api.get("/actions", params={"region_id": REGION}).json()["data"][0]
    assert pd.Timestamp(served["acknowledged_at"]) == acked_at

    assert replay_api.delete(url, params={"region_id": REGION}).json()["acknowledged_at"] is None
    assert (
        replay_api.get("/actions", params={"region_id": REGION}).json()["data"][0][
            "acknowledged_at"
        ]
        is None
    )


def test_acknowledgements_are_scoped_to_their_run(replay_api):
    """Tomorrow's cycle recommending the same action is a NEW recommendation."""
    old = replay_api.get("/actions", params={"region_id": REGION, "run_ts": RUN_TS_OLD.isoformat()})
    new = replay_api.get("/actions", params={"region_id": REGION})
    old_id, new_id = old.json()["data"][0]["action_id"], new.json()["data"][0]["action_id"]
    assert old_id != new_id

    r = replay_api.post(
        f"/actions/{new_id}/ack", params={"region_id": REGION, "run_ts": RUN_TS_OLD.isoformat()}
    )
    assert r.status_code == 404 and r.json()["error"] == "ActionNotFound"


def test_unknown_action_cannot_be_acknowledged(replay_api):
    r = replay_api.post("/actions/not-an-action/ack", params={"region_id": REGION})
    assert r.status_code == 404
    assert r.json()["error"] == "ActionNotFound" and r.json()["hint"]


def test_backtest_summary_is_summarise_not_a_client_rederivation(replay_api):
    """The headline is a ROW-WEIGHTED mean of per-lead nRMSE with skill from the
    aggregated errors. A pooled RMS -- what the frontend first computed -- reads
    6.05% for solar where the published headline is 5.36%."""
    body = replay_api.get("/backtest", params={"region_id": REGION, "tech": "solar"}).json()
    s = body["summary"]
    assert s["nrmse_mean"] == pytest.approx((100 * 0.05 + 300 * 0.06) / 400)
    assert s["nrmse_mean"] != pytest.approx(((100 * 0.05**2 + 300 * 0.06**2) / 400) ** 0.5)
    assert s["skill_mean"] == pytest.approx(1 - s["nrmse_mean"] / s["nrmse_persistence"])
    assert s["folds"] == 22
    assert s["n_rows"] == 400
    assert s["leads_scored"] == 2  # the zero-row lead hour is not scored
    assert s["leads_in_band"] == 1  # 0.80 is inside 0.78-0.82, 0.70 is not
    assert [d["lead_hours"] for d in body["data"]] == [24, 36, 48]
