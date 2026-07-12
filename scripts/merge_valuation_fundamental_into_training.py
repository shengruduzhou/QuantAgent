#!/usr/bin/env python3
"""H-020: merge the PIT val_fund daily block into the production training set.

RAM-safe: streams the base dataset row-group by row-group (pyarrow), left-joins
the small daily val_fund block (built by build_valuation_fundamental_features.py
--daily from the SAME keys, so the join is 1:1), overwrites the stale
missing_fundamentals / missing_valuation placeholder flags with the honest ones,
and appends to a new parquet via ParquetWriter. Emits a feature_schema.json with
a fresh feature_version + schema_hash so the trainer's schema-parity gate stays
armed.

No leakage: the block is already PIT (available_at <= trade_date asserted at
build). This step is a pure column attach on identical keys; row count invariant
is asserted per chunk and in aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
BLOCK = REPO / "runtime/data/v7/silver/valuation/val_fund_features.parquet"
OUT = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"

# columns from the daily block to attach (everything except the join keys)
OVERWRITE = ["missing_fundamentals", "missing_valuation"]  # replace stale placeholders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--block", default=str(BLOCK))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--feature-version", default="plus7clean_fund")
    args = ap.parse_args()

    block = pd.read_parquet(args.block)
    block["trade_date"] = pd.to_datetime(block["trade_date"])
    block["symbol"] = block["symbol"].astype(str)
    new_cols = [c for c in block.columns if c not in ("symbol", "trade_date")]
    block = block.set_index(["symbol", "trade_date"])
    # de-dup any accidental duplicate keys (keep last) so the join stays 1:1
    block = block[~block.index.duplicated(keep="last")]
    print(f"block: {len(block):,} rows, {len(new_cols)} new cols", flush=True)

    pf = pq.ParquetFile(args.base)
    writer: pq.ParquetWriter | None = None
    total_in = total_out = 0
    cov_accum: dict[str, list[int]] = {c: [0, 0] for c in ("pb", "roe", "pe_ttm", "valuation_percentile")}
    try:
        for rg in range(pf.num_row_groups):
            chunk = pf.read_row_group(rg).to_pandas()
            n0 = len(chunk)
            total_in += n0
            chunk["_sym"] = chunk["symbol"].astype(str)
            chunk["_td"] = pd.to_datetime(chunk["trade_date"])
            idx = pd.MultiIndex.from_arrays([chunk["_sym"], chunk["_td"]])
            joined = block.reindex(idx)  # 1:1 align, NaN where key absent
            for c in new_cols:
                vals = joined[c].to_numpy()
                if c in OVERWRITE and c in chunk.columns:
                    chunk[c] = vals  # overwrite stale placeholder
                else:
                    chunk[c] = vals
            chunk = chunk.drop(columns=["_sym", "_td"])
            assert len(chunk) == n0, f"row count changed in rg {rg}: {n0}->{len(chunk)}"
            total_out += len(chunk)
            for c, acc in cov_accum.items():
                acc[0] += int(chunk[c].notna().sum()); acc[1] += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression="snappy")
            writer.write_table(table)
            print(f"  rg {rg}: {n0:,} rows written", flush=True)
    finally:
        if writer is not None:
            writer.close()

    assert total_in == total_out, f"ROW COUNT INVARIANT BROKEN {total_in} -> {total_out}"
    print(f"wrote {Path(args.out).name}: {total_out:,} rows (row-count invariant OK)")
    for c, (n, d) in cov_accum.items():
        print(f"  coverage {c:22s} {n/d:.1%}")

    # feature_schema.json (fresh version + hash)
    final_cols = list(pq.ParquetFile(args.out).schema_arrow.names)
    label_like = {c for c in final_cols if c.startswith("forward_return") or c == "label"}
    key_like = {"symbol", "trade_date", "available_at", "source", "source_type",
                "source_reliability", "point_in_time_valid"}
    feats = [c for c in final_cols if c not in label_like and c not in key_like]
    schema_hash = hashlib.sha256(("|".join(feats)).encode()).hexdigest()
    schema = {"feature_version": args.feature_version, "schema_hash": schema_hash,
              "feature_count": len(feats), "new_val_fund_cols": new_cols,
              "feature_columns": feats}
    sp = Path(args.out).with_suffix(".feature_schema.json")
    sp.write_text(json.dumps(schema, indent=2))
    print(f"wrote {sp.name}: feature_version={args.feature_version} "
          f"schema_hash={schema_hash[:12]} feature_count={len(feats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
