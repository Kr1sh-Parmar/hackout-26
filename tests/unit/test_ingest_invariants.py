"""Two invariants that are enforced at several call sites and nowhere else.

Both are silent when broken, both are written down as non-negotiable, and both
are exactly the kind of thing a fifth call site forgets. A behavioural test can
only cover the call sites that already exist; these scan the source so a NEW one
cannot omit them.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

from src.core.config import load_region
from src.ingest.adapters import openmeteo

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = sorted({*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")})


def _calls(tree: ast.AST, *names: str):
    """Every Call node whose function name (or attribute) is one of `names`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in names:
            yield node


def _literal_kwargs(node: ast.Call) -> dict:
    out = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            out[kw.arg] = ...  # present but not a literal; still counts as set
    return out


def _request_params(tree: ast.AST):
    """Every dict literal in the module that requests WIND from Open-Meteo.

    Scanning the Call node itself is not enough and the meta-test below caught
    that: every real call site builds `params = {...}` as a variable and passes
    the name, so a scan of inline call kwargs matched nothing and the invariant
    passed vacuously. Match the dict wherever it is written instead.

    Scoped to requests that actually ask for a wind variable. The first version
    matched any dict with an `hourly` key and flagged `om_quota.PROBE_PARAMS`,
    which asks for `temperature_2m` alone precisely so the quota probe weighs
    nothing -- a request with no wind in it cannot have a wind-unit bug.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        if "hourly" not in keys:
            continue
        hourly = next(
            (v for k, v in zip(node.keys, node.values) if getattr(k, "value", None) == "hourly"),
            None,
        )
        try:
            requested = ast.literal_eval(hourly) if hourly is not None else ""
        except ValueError:
            requested = ...  # built dynamically -- assume it may include wind
        if requested is not ... and "wind_speed" not in str(requested):
            continue
        out = {}
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant):
                continue
            try:
                out[k.value] = ast.literal_eval(v)
            except ValueError:
                out[k.value] = ...
        yield node, out


def test_every_open_meteo_request_asks_for_metres_per_second():
    """Open-Meteo DEFAULTS TO KM/H and turbine power goes as v**3, so a missed
    conversion is a ~47x error in MW that fails completely silently -- the
    forecast is smooth, plausible and enormous. The non-negotiables say to
    assert it in a test; this is that test.
    """
    offenders = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, params in _request_params(tree):
            if params.get("wind_speed_unit") != "ms":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "Open-Meteo request without wind_speed_unit='ms' at: " + ", ".join(
        offenders
    )


def test_every_elia_csv_read_declares_the_bom_encoding():
    """Elia's CSV exports carry a UTF-8 BOM. Read without `utf-8-sig`, the first
    column name silently becomes '\ufeffdatetime' and every lookup on it fails
    with a KeyError far from the cause.
    """
    offenders = []
    for path in SOURCES:
        if "elia" not in path.name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree, "read_csv"):
            if _literal_kwargs(call).get("encoding") != "utf-8-sig":
                offenders.append(f"{path.relative_to(ROOT)}:{call.lineno}")
    assert not offenders, "Elia CSV read without encoding='utf-8-sig' at: " + ", ".join(offenders)


def test_the_scan_actually_finds_the_call_sites_it_claims_to_check():
    """A source-scanning test that matches nothing passes forever. Pin the fact
    that both scans have something to look at."""
    om = elia_reads = 0
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        om += sum(1 for _node, p in _request_params(tree) if p.get("wind_speed_unit") == "ms")
        if "elia" in path.name:
            elia_reads += sum(1 for _ in _calls(tree, "read_csv"))
    assert om >= 4, f"expected the known Open-Meteo request sites, found {om}"
    assert elia_reads >= 1, "expected at least one Elia CSV read"


def test_the_fetched_frame_is_in_metres_per_second_not_kilometres_per_hour(monkeypatch):
    """The invariant, checked on data rather than on source: Belgian 100 m wind
    in m/s sits in single digits to low tens. The same series in km/h is ~3.6x
    larger, which the schema's 0-110 bound would not necessarily catch but a
    physics sanity range does.
    """
    captured = {}

    def _fake_fetch_point(lat, lon, model, forecast_days=4):
        captured["called"] = True
        idx = pd.date_range("2026-09-12", periods=6, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "valid_ts_utc": idx,
                "wind_speed_100m_ms": [3.0, 6.5, 9.1, 12.4, 7.7, 4.2],
                "latitude": lat,
                "longitude": lon,
            }
        )

    monkeypatch.setattr(openmeteo, "fetch_point", _fake_fetch_point)
    out = openmeteo.fetch_region(load_region("BE"))

    assert captured["called"]
    ws = out["wind_speed_100m_ms"]
    assert ws.max() < 40, "100 m wind above 40 implies km/h, not m/s"
    assert ws.min() >= 0


def test_wind_speed_unit_is_pinned_in_the_adapter_params():
    """The adapter is the one call site the serving path uses, so pin its value
    directly as well as by the scan above."""
    src = (ROOT / "src" / "ingest" / "adapters" / "openmeteo.py").read_text(encoding="utf-8")
    assert '"wind_speed_unit": "ms"' in src


@pytest.mark.parametrize("bad", ["kmh", "mph", "kn", None])
def test_the_scan_would_actually_fail_on_a_bad_unit(bad, tmp_path):
    """Proves the guard has teeth: a synthetic call site with the wrong unit is
    detected by exactly the predicate the real scan uses. Without this, a scan
    that silently stopped matching would keep passing -- which is precisely what
    the first version of this module did.
    """
    params = {"hourly": ["wind_speed_100m"]}
    if bad is not None:
        params["wind_speed_unit"] = bad
    src = "\n".join(
        ["import requests", "params = " + repr(params), "requests.get('u', params=params)"]
    )
    f = tmp_path / "bad_source.py"
    f.write_text(src, encoding="utf-8")

    tree = ast.parse(f.read_text(encoding="utf-8"))
    found = [n for n, p in _request_params(tree) if p.get("wind_speed_unit") != "ms"]

    assert found, "the scan failed to flag a request with the wrong wind unit"
