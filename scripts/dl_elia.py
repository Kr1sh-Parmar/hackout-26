"""Tier-1: Elia Belgian ground truth (ods032 solar, ods031 wind, ods001 load).

Bulk endpoint is /exports/csv -- /records caps at 100 rows. Downloads in yearly
chunks because a 4-year export times out (data.md 14).
"""
from __future__ import annotations
import pathlib, sys, time
import requests

OUT = pathlib.Path("data/raw/elia"); OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"

DATASETS = {
    "ods032": "solar_historical",
    "ods031": "wind_historical",
    "ods001": "load_historical",
}
YEARS = [2023, 2024, 2025, 2026]
END = "2026-09-11"          # today-ish; 2026 chunk stops here


def fetch(ds: str, label: str, year: int) -> pathlib.Path | None:
    dest = OUT / f"{ds}_{label}_{year}.csv"
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  skip {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    hi = f"{year+1}-01-01" if year < 2026 else END
    where = f"datetime >= '{year}-01-01' AND datetime < '{hi}'"
    tmp = dest.with_suffix(".part")
    for attempt in (1, 2, 3):
        try:
            with requests.get(
                f"{BASE}/{ds}/exports/csv",
                params={"where": where, "order_by": "datetime", "timezone": "UTC",
                        "delimiter": ",", "use_labels": "false"},
                stream=True, timeout=900,
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            print(f"  {dest.name:42} {dest.stat().st_size/1e6:>7.1f} MB")
            return dest
        except Exception as e:                                   # noqa: BLE001
            print(f"  attempt {attempt} failed for {dest.name}: {e}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            time.sleep(15 * attempt)
    return None


if __name__ == "__main__":
    failed = []
    for ds, label in DATASETS.items():
        print(f"== {ds} ({label}) ==")
        for y in YEARS:
            if fetch(ds, label, y) is None:
                failed.append(f"{ds}/{y}")
            time.sleep(2)
    print("\nFAILED:", failed or "none")
    sys.exit(1 if failed else 0)
