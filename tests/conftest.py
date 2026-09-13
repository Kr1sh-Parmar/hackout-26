"""Shared test helpers.

- `read_built_parquet`: the built dataset (`data/`, ~1 GB) is gitignored and
  reproducible from `scripts/`. A clean checkout -- CI included -- holds none of it,
  so a test that reads it SKIPS with a reason instead of erroring. With the dataset
  built, those tests run exactly as before.
- `replay_api`: a TestClient serving two synthetic frozen cycles from a temp dir
  with nothing in gold. The API contract is checked against it, so the contract is
  verified on every machine, not only on one that happens to hold data.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

REGION = "BE"
RUN_TS = pd.Timestamp("2026-09-09T00:00:00Z")
RUN_TS_OLD = pd.Timestamp("2026-09-08T12:00:00Z")


def read_built_parquet(path: str) -> pd.DataFrame:
    if not pathlib.Path(path).exists():
        pytest.skip(f"needs the built dataset ({path}); it is gitignored, see context/data.md")
    return pd.read_parquet(path)


def contract_row(table: str, run_ts: pd.Timestamp) -> dict:
    """One row carrying every column the API's response model requires.

    Built from the same column constants the producers export, so if a contract
    grows a field this fixture follows it instead of quietly under-filling.
    """
    from src.decisions.events import EVENT_COLUMNS
    from src.decisions.net_load import OUTLOOK_COLUMNS
    from src.decisions.recommend import RECOMMEND_COLUMNS, Action, Flag
    from src.evaluation.drift import DRIFT_COLUMNS
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


def write_contract_gold(gold_root: pathlib.Path, table: str, run_ts: pd.Timestamp) -> None:
    dest = gold_root / table / f"region_id={REGION}" / f"run_date={run_ts:%Y-%m-%d}"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([contract_row(table, run_ts)]).to_parquet(dest / "part-0.parquet", index=False)


def write_scored_backtest_gold(gold_root: pathlib.Path) -> None:
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
    """Two frozen cycles (RUN_TS_OLD, RUN_TS) and nothing in gold: the API must serve both."""
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import app
    from src.core.config import get_settings
    from src.ingest.replay import TABLES, snapshot

    gold_root = tmp_path / "gold"
    for run in (RUN_TS_OLD, RUN_TS):
        for table in TABLES:
            write_contract_gold(gold_root, table, run)
    write_scored_backtest_gold(gold_root)

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
