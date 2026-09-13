"""API contract: every endpoint returns 200 with provenance present, unknown
region returns 404 with a hint, and every endpoint works with no gold tables
present at all (the model/decisions pipelines haven't run yet).

The contract itself is checked against the synthetic frozen cycles of the shared
`replay_api` fixture, so it holds on every machine -- a clean checkout has no
`data/`, and asserting against the real tree made CI's verdict depend on it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import get_settings

client = TestClient(app)

PROVENANCE_KEYS = {"model_version", "issued_at", "replay_mode", "region_id"}

ENDPOINTS = [
    ("/forecast", {"region_id": "BE"}),
    ("/outlook", {"region_id": "BE"}),
    ("/events", {"region_id": "BE"}),
    ("/actions", {"region_id": "BE"}),
    ("/storage/sweep", {"region_id": "BE"}),
    ("/backtest", {"region_id": "BE", "tech": "solar"}),
    ("/explain", {"region_id": "BE"}),
]

# Endpoints that answer "what is happening on the grid now" and so refuse to
# answer from nothing. /health, /sites, /backtest and /storage/sweep are exempt
# by design -- historical or static, and /health must stay reachable precisely
# when everything else is broken.
OPERATIONAL = ["/forecast", "/outlook", "/events", "/actions", "/explain"]


@pytest.fixture
def never_stale(monkeypatch):
    """Pin freshness so the suite does not start failing as the gold data ages.

    Without this, a test against real gold passes on the day someone runs a cycle
    and fails a week later -- ambient state, not behaviour.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("STALE_AFTER_MINUTES", "100000000")
    yield
    get_settings.cache_clear()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "replay_mode" in body and "warnings" in body


def test_sites_ok():
    r = client.get("/sites")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_every_endpoint_returns_provenance(replay_api):
    """Provenance is non-negotiable on every response, populated or not.

    Without model_version / issued_at, "which model produced this number?" is
    unanswerable -- and that is exactly the question asked after a bad forecast.
    """
    for path, params in ENDPOINTS:
        r = replay_api.get(path, params=params)
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


def test_no_run_at_all_is_503_not_an_empty_200(tmp_path, monkeypatch):
    """ "Nothing is happening on the grid" and "no model has ever run here" are
    different statements, and an operator sizing reserves is entitled to tell
    them apart. An empty 200 conflates them."""
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    from src.api import deps

    deps.get_store.cache_clear()
    try:
        for path in OPERATIONAL:
            r = client.get(path, params={"region_id": "BE"})
            assert r.status_code == 503, f"{path} -> {r.status_code}: {r.text}"
            body = r.json()
            assert body["error"] == "ForecastUnavailable", path
            assert body.get("hint"), f"{path} 503 has no hint: {body}"
    finally:
        get_settings.cache_clear()
        deps.get_store.cache_clear()


def test_replay_mode_never_errors_on_age(tmp_path, monkeypatch):
    """Replay is frozen data ON PURPOSE -- it is an integrity feature, not a
    fault. Erroring on its age would break the offline demo path, which is the
    one path that has to work when the venue Wi-Fi does not."""
    get_settings.cache_clear()
    monkeypatch.setenv("REPLAY_MODE", "true")
    monkeypatch.setenv("STALE_AFTER_MINUTES", "0")
    try:
        r = client.get("/forecast", params={"region_id": "BE"})
        # Either served (200) or genuinely absent (503 unavailable) -- but never
        # refused for being old, which is what replay mode exists to permit.
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            assert r.json()["error"] == "ForecastUnavailable"
    finally:
        get_settings.cache_clear()


def test_explain_drivers_are_ranked_by_absolute_contribution(never_stale):
    """An unranked attribution is a 44-row table, not an answer.

    Checked against REAL gold on purpose: a synthetic single driver is trivially
    ranked. With no cycle run on this machine there is nothing to check.
    """
    r = client.get("/explain", params={"region_id": "BE", "tech": "solar"})
    if r.status_code == 503 and r.json().get("error") == "ForecastUnavailable":
        pytest.skip("no gold forecast run on this machine; run scripts/run_cycle.py")
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    if not data:
        pytest.skip("no explain gold table yet; run scripts/run_cycle.py")
    first = [d for d in data if d["valid_ts_utc"] == data[0]["valid_ts_utc"]]
    mags = [abs(d["contribution"]) for d in sorted(first, key=lambda d: d["rank"])]
    assert mags == sorted(mags, reverse=True), f"drivers not ranked: {mags}"


def test_unknown_region_is_404_with_a_hint():
    for path, params in ENDPOINTS:
        params = {**params, "region_id": "ZZ"}
        r = client.get(path, params=params)
        assert r.status_code == 404, path
        body = r.json()
        assert body.get("hint"), f"{path} 404 has no hint: {body}"


@pytest.mark.parametrize("path,params", ENDPOINTS)
def test_every_served_response_logs_its_provenance(path, params, replay_api):
    """dev-01 12: structured logs carry region_id, run_ts and model_version.

    Without model_version in the line, the log tells you a bad forecast was
    served but not which model served it -- which is the only question anyone
    asks of these logs.
    """
    from structlog.testing import capture_logs

    with capture_logs() as entries:
        assert replay_api.get(path, params=params).status_code == 200

    served = [e for e in entries if e.get("event") == "served"]
    assert served, f"{path} served a response without logging it"
    assert {"region_id", "run_ts", "model_version"} <= set(served[0])


def test_served_backtest_headline_is_the_published_one():
    """/backtest's summary must reproduce artifacts/backtest.json to float precision:
    both come from `summarise`, over the same per-lead table."""
    import json
    import pathlib

    artifact = pathlib.Path("artifacts/backtest.json")
    if not artifact.exists():
        pytest.skip("no artifacts/backtest.json; run scripts/backtest.py")
    for published in json.loads(artifact.read_text()):
        r = client.get(
            "/backtest", params={"region_id": published["region"], "tech": published["tech"]}
        )
        assert r.status_code == 200, r.text
        served = r.json()["summary"]
        if served is None:
            pytest.skip("no backtest gold table to serve")
        for key in ("nrmse_mean", "nrmse_persistence", "nrmse_tso", "skill_mean", "picp_mean"):
            assert served[key] == pytest.approx(published[key], rel=1e-9), (published["tech"], key)
        if served["folds"] is not None:
            assert served["folds"] == published["folds"]


def test_sites_say_which_regions_are_physics_only():
    from src.core.config import load_region

    for site in client.get("/sites").json():
        assert site["physics_only"] == load_region(site["region_id"]).physics_only
