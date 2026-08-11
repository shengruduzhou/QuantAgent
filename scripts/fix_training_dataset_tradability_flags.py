#!/usr/bin/env python3
"""Repair tradability flags in gold datasets from a PIT-certified market panel.

This utility is intentionally fail-closed. Historical ``is_st`` affects both
universe eligibility and board price-limit classification, so missing panel rows
or a panel built from ``current_snapshot_broadcast`` must never be coerced to
``False``. The input panel must carry row-level historical-ST provenance and a
versioned evidence digest before any gold dataset is rewritten.
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
PIT_COLUMNS = ("point_in_time_valid", "is_st_provenance", "st_evidence_sha256")
REQUIRED_PROVENANCE = "dated_pit_st_evidence"

DEFAULT_DATASETS = [
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet",
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet",
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_selected_v85.parquet",
]
DEFAULT_PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def _strict_bool_series(series: pd.Series, *, field: str) -> pd.Series:
    if series.isna().any():
        raise SystemExit(f"panel {field} contains null values; refusing PIT promotion")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
    }
    normalised = series.astype(str).str.strip().str.lower().map(mapping)
    if normalised.isna().any():
        bad = series.loc[normalised.isna()].head(5).tolist()
        raise SystemExit(f"panel {field} contains invalid boolean values: {bad}")
    return normalised.astype(bool)


def load_panel_flags(panel_path: str) -> pd.DataFrame:
    path = Path(panel_path)
    if not path.exists():
        raise SystemExit(f"market panel does not exist: {path}")
    schema = pq.ParquetFile(path).schema_arrow
    required = ["symbol", "trade_date", *FLAG_COLS, *PIT_COLUMNS]
    missing = [column for column in required if column not in schema.names]
    if missing:
        raise SystemExit(
            "market panel cannot certify historical tradability; missing provenance "
            f"columns {missing}. Rebuild affected rows from dated PIT ST evidence first."
        )

    flags = pd.read_parquet(path, columns=required)
    flags["trade_date"] = pd.to_datetime(flags["trade_date"], errors="coerce")
    if flags["trade_date"].isna().any():
        raise SystemExit("panel has invalid trade_date rows; refusing tradability repair")
    duplicate_count = int(flags.duplicated(subset=["symbol", "trade_date"]).sum())
    if duplicate_count:
        raise SystemExit(
            f"panel has {duplicate_count} duplicate (symbol, trade_date) keys; refusing to join"
        )

    pit = _strict_bool_series(flags["point_in_time_valid"], field="point_in_time_valid")
    if not bool(pit.all()):
        bad = flags.loc[~pit, ["symbol", "trade_date"]].head(5)
        sample = [
            f"{row.symbol}@{row.trade_date.date()}"
            for row in bad.itertuples(index=False)
        ]
        raise SystemExit(
            "market panel contains point_in_time_valid=False rows; "
            f"sample={sample}"
        )

    provenance = flags["is_st_provenance"].astype("string")
    unsafe = provenance.isna() | provenance.ne(REQUIRED_PROVENANCE)
    if bool(unsafe.any()):
        counts = provenance.fillna("<missing>").value_counts(dropna=False).head(10).to_dict()
        raise SystemExit(
            "market panel historical ST provenance is not promotion-safe; expected "
            f"{REQUIRED_PROVENANCE!r} for every row, observed={counts}. "
            "current_snapshot_broadcast artifacts must be quarantined/rebuilt."
        )

    digests = flags["st_evidence_sha256"].astype("string").str.strip().str.lower()
    valid_digest = digests.str.fullmatch(r"[0-9a-f]{64}", na=False)
    if not bool(valid_digest.all()):
        raise SystemExit(
            "market panel contains missing/invalid st_evidence_sha256; refusing gold rewrite"
        )

    for column in FLAG_COLS:
        flags[column] = _strict_bool_series(flags[column], field=column)
    return flags


def fix_dataset(ds_path: Path, flags: pd.DataFrame) -> dict:
    if not ds_path.exists():
        raise SystemExit(f"dataset does not exist: {ds_path}")
    pf = pq.ParquetFile(ds_path)
    schema = pf.schema_arrow
    missing = [column for column in FLAG_COLS if column not in schema.names]
    if missing:
        raise SystemExit(f"{ds_path}: missing expected flag columns {missing}")
    flag_idx = {column: schema.get_field_index(column) for column in FLAG_COLS}

    tmp_path = ds_path.with_suffix(".flagfix.tmp.parquet")
    rows_total = 0
    matched_total = 0
    flag_true = {column: 0 for column in FLAG_COLS}
    evidence_digests: set[str] = set()

    writer = pq.ParquetWriter(tmp_path, schema)
    try:
        for row_group in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(row_group)
            keys = table.select(["symbol", "trade_date"]).to_pandas()
            keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce")
            if keys["trade_date"].isna().any():
                raise SystemExit(
                    f"{ds_path} rg{row_group}: invalid dataset trade_date; refusing rewrite"
                )
            keys["_ord"] = range(len(keys))
            joined = keys.merge(
                flags,
                on=["symbol", "trade_date"],
                how="left",
                validate="many_to_one",
            )
            if len(joined) != len(keys):
                raise SystemExit(
                    f"{ds_path} rg{row_group}: join changed row count {len(keys)} -> {len(joined)}"
                )
            joined = joined.sort_values("_ord")
            missing_panel = joined["is_st"].isna()
            if bool(missing_panel.any()):
                sample = joined.loc[missing_panel, ["symbol", "trade_date"]].head(5)
                rendered = [
                    f"{row.symbol}@{row.trade_date.date()}"
                    for row in sample.itertuples(index=False)
                ]
                raise SystemExit(
                    f"{ds_path} rg{row_group}: panel does not cover every dataset row; "
                    f"sample={rendered}. Missing tradability is not neutral/False."
                )
            matched_total += len(joined)
            evidence_digests.update(
                joined["st_evidence_sha256"].astype(str).str.lower().tolist()
            )
            for column in FLAG_COLS:
                values = _strict_bool_series(joined[column], field=column).to_numpy()
                flag_true[column] += int(values.sum())
                table = table.set_column(
                    flag_idx[column],
                    schema.field(column),
                    pa.array(values, type=pa.bool_()),
                )
            writer.write_table(table)
            rows_total += len(keys)
    except Exception:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    check = pq.ParquetFile(tmp_path)
    if check.metadata.num_rows != pf.metadata.num_rows:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(
            f"{ds_path}: row count mismatch {check.metadata.num_rows} vs {pf.metadata.num_rows}"
        )
    if check.schema_arrow.names != schema.names:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"{ds_path}: schema mismatch after rewrite")

    mode = ds_path.stat().st_mode
    os.replace(tmp_path, ds_path)
    os.chmod(ds_path, mode)
    return {
        "dataset": str(ds_path),
        "rows": rows_total,
        "panel_match_rate": round(matched_total / max(1, rows_total), 4),
        "flag_rates": {
            column: round(flag_true[column] / max(1, rows_total), 4)
            for column in FLAG_COLS
        },
        "st_evidence_sha256": sorted(evidence_digests),
        "historical_st_provenance": REQUIRED_PROVENANCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    args = parser.parse_args()

    flags = load_panel_flags(args.panel)
    for dataset in args.datasets:
        report = fix_dataset(Path(dataset), flags)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
