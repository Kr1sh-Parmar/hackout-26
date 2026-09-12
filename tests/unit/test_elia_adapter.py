"""The Elia adapter had no tests at all, and it carries two documented traps
that already produced wrong numbers once.

Nothing here touches the network: the HTTP layer is stubbed, because a green
build that depended on Elia being up would be a weather report, not a test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.adapters import elia
from src.ingest.adapters.elia import LoadUnavailable, climatology_load, fetch_load_forecast
from src.quality.curtailment import curtailment_flag

START = pd.Timestamp("2026-09-12T00:00:00Z")
END = pd.Timestamp("2026-09-13T00:00:00Z")


def _record(minute_offset: int, **fields) -> dict:
    ts = START + pd.Timedelta(minutes=minute_offset)
    row = {
        "datetime": ts.isoformat(),
        "dayaheadforecast": None,
        "weekaheadforecast": None,
        "mostrecentforecast": None,
        "mostrecentconfidence10": None,
        "mostrecentconfidence90": None,
    }
    row.update(fields)
    return row


@pytest.fixture
def stub_pages(monkeypatch):
    """Replace the paged HTTP fetch with a canned list of records."""

    def _install(rows):
        calls = []

        def _fake_page(ds, where, offset, timeout):
            calls.append(offset)
            return rows[offset : offset + elia.PAGE]

        monkeypatch.setattr(elia, "_page", _fake_page)
        return calls

    return _install


# ---------------------------------------------------------------- vintages


def test_day_ahead_wins_and_the_vintage_is_recorded(stub_pages):
    """Precedence is not cosmetic: `dayaheadforecast` is the number the market
    clears against, and an operator must be able to tell it from a substitute.
    Blending vintages silently would make the net-load band unattributable."""
    stub_pages(
        [
            _record(0, dayaheadforecast=9000.0, weekaheadforecast=1.0, mostrecentforecast=2.0),
            _record(15, dayaheadforecast=9100.0, weekaheadforecast=1.0, mostrecentforecast=2.0),
            _record(30, dayaheadforecast=9200.0),
            _record(45, dayaheadforecast=9300.0),
        ]
    )

    out = fetch_load_forecast(START, END)

    assert len(out) == 1, "four 15-minute records collapse to one hour"
    assert out.loc[0, "demand_vintage"] == "dayaheadforecast"
    assert out.loc[0, "demand_da_mw"] == pytest.approx(9150.0)
    # the decision layer reads `demand_mw`; it must be the forecast, never an actual
    assert out.loc[0, "demand_mw"] == out.loc[0, "demand_da_mw"]


def test_falls_back_down_the_vintage_order_per_interval(stub_pages):
    """Day-ahead does not exist for the tail of a 72 h horizon, so week-ahead
    has to carry it -- and say that it did."""
    stub_pages(
        [
            _record(0, dayaheadforecast=9000.0),
            _record(15, dayaheadforecast=9000.0),
            _record(30, dayaheadforecast=9000.0),
            _record(45, dayaheadforecast=9000.0),
            _record(60, weekaheadforecast=8000.0),
            _record(75, weekaheadforecast=8000.0),
            _record(90, weekaheadforecast=8000.0),
            _record(105, mostrecentforecast=7000.0),
        ]
    )

    out = fetch_load_forecast(START, END).sort_values("valid_ts_utc").reset_index(drop=True)

    assert list(out["demand_vintage"]) == ["dayaheadforecast", "weekaheadforecast"]


def test_no_usable_vintage_raises_rather_than_inventing_demand(stub_pages):
    """Running the decision layer on invented demand is called out in the design
    docs as a project-killer. Absent demand must fail loudly so the caller can
    fall back to climatology and LABEL it."""
    stub_pages([_record(0), _record(15)])  # every vintage null

    with pytest.raises(LoadUnavailable):
        fetch_load_forecast(START, END)


def test_an_empty_response_raises(stub_pages):
    stub_pages([])

    with pytest.raises(LoadUnavailable):
        fetch_load_forecast(START, END)


def test_a_persistently_failing_endpoint_raises_load_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("elia is down")

    monkeypatch.setattr(elia, "_page", _boom)
    monkeypatch.setattr(elia.time, "sleep", lambda _s: None)  # do not really back off

    with pytest.raises(LoadUnavailable, match="ods001"):
        fetch_load_forecast(START, END, tries=2)


def test_paging_continues_past_the_first_page(stub_pages):
    """The /records endpoint caps at 100 rows. Stopping after one page would
    silently truncate a 4-day horizon to the first 25 hours."""
    rows = [_record(15 * i, dayaheadforecast=9000.0 + i) for i in range(elia.PAGE + 8)]
    calls = stub_pages(rows)

    out = fetch_load_forecast(START, END)

    assert len(calls) > 1, "did not page"
    assert len(out) == pytest.approx(len(rows) / 4, abs=1)


# ---------------------------------------------------------------- fallback


def test_climatology_is_serviceable_and_labels_itself():
    """Demand is strongly hour-of-week periodic, so this is a real substitute --
    but an operator must never mistake it for the TSO's own number."""
    hist_index = pd.date_range("2026-07-01", periods=24 * 70, freq="h", tz="UTC")
    # a clean weekday/weekend split the profile has to recover
    weekend = hist_index.dayofweek >= 5
    history = pd.DataFrame(
        {"ts_utc": hist_index, "demand_mw": [7000.0 if w else 10000.0 for w in weekend]}
    )
    target = pd.date_range("2026-09-14", periods=24 * 7, freq="h", tz="UTC")  # Mon..Sun

    out = climatology_load(history, target)

    assert (out["demand_vintage"] == "climatology").all()
    assert len(out) == len(target)
    is_weekend = pd.DatetimeIndex(out["valid_ts_utc"]).dayofweek >= 5
    assert out.loc[is_weekend, "demand_mw"].mean() == pytest.approx(7000.0)
    assert out.loc[~is_weekend, "demand_mw"].mean() == pytest.approx(10000.0)
    # the band must bracket the point forecast, not sit beside it
    assert (out["demand_da_p10_mw"] < out["demand_mw"]).all()
    assert (out["demand_da_p90_mw"] > out["demand_mw"]).all()


# ---------------------------------------------------------------- the CSV trap


def test_the_two_apostrophe_empty_field_is_not_a_curtailment():
    """Elia's CSV export writes an empty text field as the two-character string
    `''`, so the column is 100% non-null and ~99.2% empty. A `.notna()` test
    flagged 99.6% of the wind fleet as curtailed against a real 2.96%, and every
    downstream sample weight was wrong in a way that looked entirely plausible.
    """
    bids = pd.Series(["''", "", "  ''  ", None, '""', "BID-4711", " 'BID-4712' "])

    flags = curtailment_flag(bids)

    assert list(flags) == [False, False, False, False, False, True, True]
    # the naive test this replaced, kept as the contrast that makes the point
    assert bids.notna().sum() == 6, "the column really is almost entirely non-null"
