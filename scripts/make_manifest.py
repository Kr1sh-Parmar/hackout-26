"""Inventory every table in the dataset: rows, columns, time range, size, nulls.

Written to data/_reports/manifest.md + manifest.json so the dataset is
self-describing and a reviewer can see what exists without loading it.
"""
from __future__ import annotations
import json, pathlib
import pandas as pd

ROOT = pathlib.Path("data")
REPORTS = ROOT / "_reports"; REPORTS.mkdir(parents=True, exist_ok=True)
TS_CANDIDATES = ("valid_ts_utc", "ts_utc", "date", "time")


def describe(path: pathlib.Path) -> dict:
    df = pd.read_parquet(path)
    ts = next((c for c in TS_CANDIDATES if c in df.columns), None)
    rec = {
        "table": f"{path.parent.parent.name}/{path.parent.name}",
        "rows": len(df),
        "cols": len(df.columns),
        "bytes": path.stat().st_size,
        "time_col": ts,
        "t_min": str(pd.to_datetime(df[ts]).min()) if ts else None,
        "t_max": str(pd.to_datetime(df[ts]).max()) if ts else None,
        "columns": list(df.columns),
        "null_pct": {c: round(100 * df[c].isna().mean(), 2)
                     for c in df.columns if df[c].isna().any()},
    }
    del df
    return rec


def main() -> None:
    recs = []
    for layer in ("bronze", "silver", "gold"):
        for part in sorted((ROOT / layer).glob("*/part-0.parquet")):
            recs.append(describe(part))
            print(f"  {recs[-1]['table']:44} {recs[-1]['rows']:>10,} rows")

    raw = []
    for f in sorted((ROOT / "raw").rglob("*")):
        if f.is_file() and not f.name.endswith(".part"):
            raw.append({"file": str(f.relative_to(ROOT)).replace("\\", "/"),
                        "bytes": f.stat().st_size})

    (REPORTS / "manifest.json").write_text(
        json.dumps({"tables": recs, "raw_files": raw}, indent=2))

    tot_tbl = sum(r["bytes"] for r in recs)
    tot_raw = sum(r["bytes"] for r in raw)
    lines = [
        "# Dataset manifest", "",
        f"Generated from `data/`. {len(recs)} tables, "
        f"{sum(r['rows'] for r in recs):,} rows, {tot_tbl/1e6:,.1f} MB processed "
        f"+ {tot_raw/1e6:,.1f} MB raw ({len(raw)} files).", "",
        "## Tables", "",
        "| Table | Rows | Cols | Size | Time column | From | To |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for r in recs:
        lines.append(
            f"| `{r['table']}` | {r['rows']:,} | {r['cols']} | {r['bytes']/1e6:.1f} MB | "
            f"{r['time_col'] or '—'} | {(r['t_min'] or '—')[:16]} | {(r['t_max'] or '—')[:16]} |")

    lines += ["", "## Raw files", "",
              "| File | Size |", "|---|---:|"]
    for r in raw:
        lines.append(f"| `{r['file']}` | {r['bytes']/1e6:.1f} MB |")

    (REPORTS / "manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REPORTS/'manifest.md'} and manifest.json")
    print(f"{len(recs)} tables, {tot_tbl/1e6:,.1f} MB processed, {tot_raw/1e6:,.1f} MB raw")


if __name__ == "__main__":
    main()
