"""Operator acknowledgements -- the one thing this API writes.

Everything else is a read layer over pipeline output. An acknowledgement is
operator state, so it lives beside the data (`DATA_ROOT/ops/acks/{region}.json`),
never inside a gold table the next cycle would overwrite. It is written in
replay mode too: an operator acknowledging an action during a demo is using the
product, not mutating the frozen forecast.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading

import pandas as pd

from .deps import get_settings

# ponytail: one process-wide lock. Enough for a single uvicorn worker; several
# workers writing acks need a file lock or a real database.
_LOCK = threading.Lock()


def _utc_iso(ts) -> str:
    t = pd.Timestamp(ts)
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()


def action_id(region_id: str, run_ts, action: str, valid_from) -> str:
    """Stable id for one action in one run.

    Keyed by the run as well as the window: the same CHARGE_BESS at the same
    hour recommended by tomorrow's cycle is a new recommendation, and must not
    arrive already acknowledged.
    """
    key = f"{region_id}|{_utc_iso(run_ts)}|{action}|{_utc_iso(valid_from)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _path(region_id: str):
    return get_settings().data_root / "ops" / "acks" / f"{region_id}.json"


def load(region_id: str) -> dict[str, str]:
    """action_id -> ISO acknowledgement time. A corrupt file raises rather than
    reading as empty: the next write would otherwise erase every acknowledgement."""
    try:
        return json.loads(_path(region_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write(region_id: str, acks: dict[str, str]) -> None:
    path = _path(region_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(acks, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash mid-write never leaves half a file


def acknowledge(region_id: str, aid: str) -> str:
    """Idempotent: acknowledging twice keeps the FIRST time, which is the one that matters."""
    with _LOCK:
        acks = load(region_id)
        if aid not in acks:
            acks[aid] = dt.datetime.now(dt.timezone.utc).isoformat()
            _write(region_id, acks)
        return acks[aid]


def withdraw(region_id: str, aid: str) -> None:
    with _LOCK:
        acks = load(region_id)
        if acks.pop(aid, None) is not None:
            _write(region_id, acks)
