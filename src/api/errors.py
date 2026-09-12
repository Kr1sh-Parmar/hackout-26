"""Domain errors that always carry an actionable `hint`.

`DomainError` is the base; the FastAPI handler (registered in main.py) turns
any of these into `{"error": <class name>, "message": <msg>, "hint": <hint>}`
at the given status code.
"""

from __future__ import annotations


class DomainError(Exception):
    status_code: int = 400

    def __init__(self, msg: str, hint: str):
        super().__init__(msg)
        self.msg = msg
        self.hint = hint


class RegionNotConfigured(DomainError):
    status_code = 404

    def __init__(self, region_id: str, known: list[str]):
        super().__init__(
            msg=f"no config for region {region_id!r}",
            hint=f"configured regions: {known or 'none'}. Add a YAML file under "
            f"config/regions/ with a matching region_id field.",
        )


class ForecastUnavailable(DomainError):
    status_code = 503

    def __init__(self, region_id: str):
        super().__init__(
            msg=f"no forecast is available yet for region {region_id!r}",
            hint="the model pipeline has not produced a gold/forecast run for this "
            "region yet; check back after the next scheduled run_cycle, or trigger "
            "one manually.",
        )


class RunNotFound(DomainError):
    status_code = 404

    def __init__(self, region_id: str, run_ts, known: list):
        listed = [t.isoformat() for t in known[:10]]
        super().__init__(
            msg=f"no run at {run_ts.isoformat()} for region {region_id!r}",
            hint=f"available runs, newest first: {listed or 'none'}. "
            f"GET /runs?region_id={region_id} lists them all.",
        )


class ActionNotFound(DomainError):
    status_code = 404

    def __init__(self, region_id: str, action_id: str):
        super().__init__(
            msg=f"no action {action_id!r} in the {region_id!r} run being served",
            hint="action ids come from GET /actions for the same region and run_ts; "
            "a newer cycle may have replaced the run you were looking at -- refetch /actions.",
        )


class StaleForecast(DomainError):
    status_code = 503

    def __init__(self, region_id: str, age_minutes: float):
        super().__init__(
            msg=f"the latest forecast for {region_id!r} is {age_minutes:.0f} minutes old",
            hint="trigger a fresh run_cycle for this region, or confirm the scheduler is running.",
        )
