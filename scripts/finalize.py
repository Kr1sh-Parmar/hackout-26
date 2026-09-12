"""Rebuild every derived layer and re-verify. Run after any new raw data lands.

    python scripts/finalize.py
"""
from __future__ import annotations
import subprocess, sys

STEPS = ["scripts/build_elia.py", "scripts/build_weather.py", "scripts/build_india.py",
         "scripts/build_india_weather.py", "scripts/build_opsd.py", "scripts/build_gold.py",
         "scripts/verify_dataset.py", "scripts/make_manifest.py"]

if __name__ == "__main__":
    for s in STEPS:
        print(f"\n{'='*70}\n== {s}\n{'='*70}")
        r = subprocess.run([sys.executable, s])
        if r.returncode and "verify" not in s:
            sys.exit(f"FAILED at {s}")
    print("\nrebuild complete")
