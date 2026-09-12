"""Tier-2: India transfer region (Zenodo 7824872 subset) + OPSD cross-check region.

Zenodo: the full record is 115 files / 47.7 GB. Take the named 8-file subset only
(~60 MB). Do NOT mirror the annual capacity-factor ZIPs.
"""

from __future__ import annotations
import pathlib
import sys
import time
import requests

ZEN = pathlib.Path("data/raw/india")
ZEN.mkdir(parents=True, exist_ok=True)
OPSD = pathlib.Path("data/raw/opsd")
OPSD.mkdir(parents=True, exist_ok=True)

ZEN_BASE = "https://zenodo.org/records/7824872/files"
ZEN_FILES = [
    "modelled-historical-hourly-renewable_output.nc",  # the main file
    "installed-by-state-oct2022.csv",
    "tabulated-installed-by-date.csv",
    "POSOCO_reported_solar_MU_daily.csv",  # real reported validation data
    "POSOCO_reported_wind_MU_daily.csv",
    "CEA_1x1_gridded_installed_solar_cap.nc",
    "CEA_1x1_gridded_installed_wind_cap.nc",
    "OSM_wind_turbine_installations.geojson",
]

OPSD_BASE = "https://data.open-power-system-data.org/time_series/2020-10-06"
OPSD_FILES = ["time_series_60min_singleindex.csv", "time_series_15min_singleindex.csv"]


def _replace_retry(tmp: pathlib.Path, dest: pathlib.Path, tries: int = 12) -> None:
    """Windows AV scanners hold a handle on a freshly-written large file, so the
    rename races with them. Retry rather than re-download 130 MB."""
    for i in range(tries):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            time.sleep(5)
    raise PermissionError(f"could not rename {tmp} after {tries} tries")


def get(url: str, dest: pathlib.Path, min_bytes: int = 512) -> bool:
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f"  skip {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in (1, 2, 3):
        try:
            with requests.get(
                url, stream=True, timeout=900, headers={"User-Agent": "zero-bias-research/1.0"}
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            _replace_retry(tmp, dest)
            print(f"  {dest.name:52} {dest.stat().st_size / 1e6:>7.1f} MB")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} {dest.name}: {e}", file=sys.stderr)
            try:
                tmp.unlink(missing_ok=True)
            except PermissionError:
                pass
            time.sleep(10 * attempt)
    return False


if __name__ == "__main__":
    failed = []
    print("== Zenodo 7824872 (India) ==")
    for f in ZEN_FILES:
        if not get(f"{ZEN_BASE}/{f}?download=1", ZEN / f):
            failed.append(f"zenodo/{f}")
        time.sleep(1)
    print("== OPSD (Germany cross-check) ==")
    for f in OPSD_FILES:
        if not get(f"{OPSD_BASE}/{f}", OPSD / f):
            failed.append(f"opsd/{f}")
        time.sleep(1)
    print("\nFAILED:", failed or "none")
    sys.exit(1 if failed else 0)
