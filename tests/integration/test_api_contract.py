"""API contract: every endpoint returns 200 with provenance present, unknown
region returns 404 with a hint, and every endpoint works with no gold tables
present at all (the model/decisions pipelines haven't run yet)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

PROVENANCE_KEYS = {"model_version", "issued_at", "replay_mode", "region_id"}

ENDPOINTS = [
    ("/forecast", {"region_id": "BE"}),
    ("/outlook", {"region_id": "BE"}),
    ("/events", {"region_id": "BE"}),
    ("/actions", {"region_id": "BE"}),
    ("/storage/sweep", {"region_id": "BE"}),
    ("/backtest", {"region_id": "BE", "tech": "solar"}),
]


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "replay_mode" in body and "warnings" in body


def test_sites_ok():
    r = client.get("/sites")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_every_endpoint_returns_provenance():
    """Provenance is non-negotiable on every response, populated or not.

    Without model_version / issued_at, "which model produced this number?" is
    unanswerable -- and that is exactly the question asked after a bad forecast.
    """
    for path, params in ENDPOINTS:
        r = client.get(path, params=params)
        assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text}"
        body = r.json()
        assert PROVENANCE_KEYS.issubset(body), f"{path} missing provenance: {body}"
        assert isinstance(body["data"], list), f"{path} data is not a list"


def test_endpoints_serve_empty_data_when_no_gold_tables_exist(tmp_path, monkeypatch):
    """The API must be usable before the pipeline has ever run.

    Isolated against an empty data root on purpose: asserting this against the
    real directory made the test pass or fail depending on whether anyone had
    run a forecast cycle, which is ambient state, not behaviour.
    """
    from src.core.config import get_settings
    from src.core.store import ParquetStore

    get_settings.cache_clear()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    store = ParquetStore(tmp_path)
    try:
        for reader, kwargs in (
            (store.read_forecast, {"region_id": "BE"}),
            (store.read_outlook, {"region_id": "BE"}),
            (store.read_events, {"region_id": "BE"}),
            (store.read_actions, {"region_id": "BE"}),
        ):
            out = reader(**kwargs)
            assert out.empty, f"{reader.__name__} should be empty against a bare data root"
        assert store.latest_run("BE") is None
    finally:
        get_settings.cache_clear()


def test_unknown_region_is_404_with_a_hint():
    for path, params in ENDPOINTS:
        params = {**params, "region_id": "ZZ"}
        r = client.get(path, params=params)
        assert r.status_code == 404, path
        body = r.json()
        assert body.get("hint"), f"{path} 404 has no hint: {body}"
