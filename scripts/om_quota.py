"""Shared Open-Meteo fetch helper with quota-aware pacing.

Open-Meteo weights a request by (variables x days), and the archive endpoints
(historical-forecast + previous-runs) share one rolling hourly bucket.

The trap: a 429 still consumes weight. Retrying the heavy request on a timer
re-saturates the very window you are waiting on -- a retry storm that feeds
itself, which is how this pipeline stalled for an hour. Measured recovery with
zero traffic was ~9 minutes.

So: on 429, stop issuing the real request entirely and poll with a ~free probe
(1 variable, 2 days) until the bucket drains, then resume.
"""

from __future__ import annotations
import sys
import time
import requests

PROBE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PROBE_PARAMS = {
    "latitude": 50.85,
    "longitude": 4.35,
    "hourly": "temperature_2m",
    "start_date": "2025-06-01",
    "end_date": "2025-06-02",
    "timezone": "UTC",
}


class DailyQuotaExhausted(RuntimeError):
    """The archive bucket is spent until UTC midnight. Waiting cannot help."""


def wait_for_quota(poll_s: int = 120, max_wait_s: int = 5400) -> bool:
    """Poll with a negligible-weight request until the bucket has drained."""
    waited = 0
    while waited < max_wait_s:
        time.sleep(poll_s)
        waited += poll_s
        try:
            r = requests.get(PROBE_URL, params=PROBE_PARAMS, timeout=60)
            if r.status_code == 200:
                print(f"    quota recovered after {waited // 60} min", flush=True)
                return True
            # "Daily" is not a window that reopens by waiting -- stop immediately
            # rather than burning 90 minutes discovering that.
            if "Daily" in r.text:
                raise DailyQuotaExhausted(r.text[:160])
        except DailyQuotaExhausted:
            raise
        except Exception:  # noqa: BLE001
            pass
        print(f"    still throttled ({waited // 60} min)", flush=True)
    return False


def date_chunks(start: str, end: str, days: int = 200):
    """Split a date range into chunks. Open-Meteo weights a call by
    (variables x days), so a long range is one indivisible expensive request that
    a partially-drained bucket keeps rejecting. Smaller chunks fit sooner and make
    progress incremental instead of all-or-nothing."""
    import datetime as _dt

    a = _dt.date.fromisoformat(start)
    b = _dt.date.fromisoformat(end)
    out = []
    while a <= b:
        c = min(a + _dt.timedelta(days=days - 1), b)
        out.append((a.isoformat(), c.isoformat()))
        a = c + _dt.timedelta(days=1)
    return out


def fetch_json(
    url: str, params: dict, label: str, tries: int = 6, pace_s: float = 8.0
) -> dict | None:
    """One paced, quota-aware GET. Returns parsed JSON or None."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=600)
            if r.status_code == 429:
                if "Daily" in r.text:
                    raise DailyQuotaExhausted(r.text[:160])
                print(f"    {label}: quota hit -- pausing all traffic", flush=True)
                if not wait_for_quota():
                    print(f"    {label}: quota never recovered", file=sys.stderr)
                    return None
                continue  # retry without counting an attempt-sleep
            r.raise_for_status()
            time.sleep(pace_s)  # be polite between successful calls
            return r.json()
        except DailyQuotaExhausted:
            raise
        except Exception as e:  # noqa: BLE001
            print(
                f"    {label}: attempt {attempt} failed: {str(e)[:120]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(20 * attempt)
    return None
