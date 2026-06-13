#!/usr/bin/env python3
"""Repair stale tradability flags in gold training datasets (in place).

The gold training datasets carry all-False is_st/is_suspended/is_limit_up/
is_limit_down columns (verified 2026-06-11: dataset rates 0.0 vs market-panel
rates 6.6% ST / 3.8% suspended / 1.1% limit-up), so any consumer trusting the
dataset columns silently gets no tradability filtering.

This script replaces the four flag columns with the real values joined from
runtime/data/v7/silver/market_panel/market_panel.parquet on
(symbol, trade_date). Dataset rows missing from the panel get False.

Memory: streams one parquet row group at a time and swaps the flag columns
via Table.set_column — the full 246-column frame is never materialised
(a full-frame load + groupby-shift OOM-killed this 64GB box before; see
scripts/build_executable_labels_dataset.py).

The fixed file is written to <name>.flagfix.tmp.parquet, verified (row count,
schema, flag rates), then atomically renamed over the original.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FLAG_COLS = ("is_st", "is_suspended", "is_limit_up", "is_limit_down")

DEFAULT_DATASETS = [
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet",
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet",
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_selected_v85.parquet",
]
DEFAULT_PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def load_panel_flags(panel_path: str) -> pd.DataFrame:
    flags = pd.read_parquet(panel_path, columns=["symbol", "trade_date", *FLAG_COLS])
    flags["trade_date"] = pd.to_datetime(flags["trade_date"], errors="coerce")
    n_dup = int(flags.duplicated(subset=["symbol", "trade_date"]).sum())
    if n_dup:
        raise SystemExit(f"panel has {n_dup} duplicate (symbol, trade_date) keys; refusing to join")
    for c in FLAG_COLS:
        flags[c] = flags[c].fillna(False).astype(bool)
    return flags


def fix_dataset(ds_path: Path, flags: pd.DataFrame) -> dict:
    pf = pq.ParquetFile(ds_path)
    schema = pf.schema_arrow
    missing = [c for c in FLAG_COLS if c not in schema.names]
    if missing:
        raise SystemExit(f"{ds_path}: missing expected flag columns {missing}")
    flag_idx = {c: schema.get_field_index(c) for c in FLAG_COLS}

    tmp_path = ds_path.with_suffix(".flagfix.tmp.parquet")
    rows_total = 0
    matched_total = 0
    flag_true = {c: 0 for c in FLAG_COLS}

    writer = pq.ParquetWriter(tmp_path, schema)
    try:
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            keys = table.select(["symbol", "trade_date"]).to_pandas()
            keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce")
            keys["_ord"] = range(len(keys))
            joined = keys.merge(flags, on=["symbol", "trade_date"], how="left")
            if len(joined) != len(keys):
                raise SystemExit(f"{ds_path} rg{rg}: join changed row count {len(keys)} -> {len(joined)}")
            joined = joined.sort_values("_ord")
            matched_total += int(joined["is_st"].notna().sum())
            for c in FLAG_COLS:
                vals = joined[c].fillna(False).astype(bool).to_numpy()
                flag_true[c] += int(vals.sum())
                table = table.set_column(flag_idx[c], schema.field(c), pa.array(vals, type=pa.bool_()))
            writer.write_table(table)
            rows_total += len(keys)
    finally:
        writer.close()

    # verify the temp file before replacing the original
    chk = pq.ParquetFile(tmp_path)
    if chk.metadata.num_rows != pf.metadata.num_rows:
        raise SystemExit(f"{ds_path}: row count mismatch {chk.metadata.num_rows} vs {pf.metadata.num_rows}")
    if chk.schema_arrow.names != schema.names:
        raise SystemExit(f"{ds_path}: schema mismatch after rewrite")

    mode = ds_path.stat().st_mode
    os.replace(tmp_path, ds_path)
    os.chmod(ds_path, mode)
    return {
        "dataset": str(ds_path),
        "rows": rows_total,
        "panel_match_rate": round(matched_total / max(1, rows_total), 4),
        "flag_rates": {c: round(flag_true[c] / max(1, rows_total), 4) for c in FLAG_COLS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    args = ap.parse_args()

    flags = load_panel_flags(args.panel)
    reports = []
    for ds in args.datasets:
        report = fix_dataset(Path(ds), flags)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
