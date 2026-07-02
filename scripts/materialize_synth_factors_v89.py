#!/usr/bin/env python3
"""Add discovered DSL factors as columns onto an existing training dataset.

Safe materialization: rather than rebuilding the whole dataset (which would
also touch the proven alpha/gtja/idx/label columns), this evaluates the
accepted PIT-safe DSL factors over the dataset itself and LEFT-JOINs the new
factor columns on (symbol, trade_date). Every pre-existing column is preserved
byte-for-byte; only the new ``llm_*`` columns are added.

The factors are point-in-time (backward-looking windows only), so evaluating
them across the full date span — including the post-cutoff test window — does
NOT leak future information: the value at date t uses only data <= t.

Usage:
  materialize_synth_factors_v89.py --dataset <in.parquet> \
      --definitions <accepted_definitions.json> --output <out.parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.factors.factor_synthesis import compute_synthesized_factors, load_definitions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Existing training dataset parquet to augment.")
    ap.add_argument("--definitions", required=True, help="accepted_definitions.json from evaluate_discovered_factors.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-finite-ratio", type=float, default=0.3,
                    help="Abort if any new factor's finite ratio is below this (guards a broken expr).")
    args = ap.parse_args()

    defs = load_definitions(args.definitions)
    if not defs:
        print(f"no definitions in {args.definitions}; nothing to do")
        return 1
    names = [d.name for d in defs]
    print(f"materializing {len(defs)} factors: {names}")

    df = pd.read_parquet(args.dataset)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # Time-series DSL ops require chronological order within each symbol.
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    pre_cols = set(df.columns)
    collide = [n for n in names if n in pre_cols]
    if collide:
        raise SystemExit(f"factor names already present in dataset (would overwrite): {collide}")

    long = compute_synthesized_factors(df, args.definitions)
    if long.empty:
        raise SystemExit("compute_synthesized_factors returned empty — exprs failed to evaluate")
    wide = long.pivot_table(index=["symbol", "trade_date"], columns="factor_name",
                            values="factor_value", aggfunc="last").reset_index()
    wide.columns.name = None

    # Finite-ratio guard: a near-empty column means a broken/incompatible expr.
    report = {}
    for n in names:
        if n not in wide.columns:
            raise SystemExit(f"factor {n} missing from computed output")
        fr = float(wide[n].replace([np.inf, -np.inf], np.nan).notna().mean())
        report[n] = round(fr, 4)
        if fr < args.min_finite_ratio:
            raise SystemExit(f"factor {n} finite ratio {fr:.3f} < {args.min_finite_ratio}; aborting")

    out = df.merge(wide, on=["symbol", "trade_date"], how="left")
    assert len(out) == len(df), f"row count changed {len(df)} -> {len(out)}"
    added = [c for c in out.columns if c not in pre_cols]
    assert set(added) == set(names), f"unexpected added cols: {added}"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": args.output,
        "rows": len(out),
        "cols_before": len(pre_cols),
        "cols_after": len(out.columns),
        "added_factors": added,
        "finite_ratios": report,
        "date_range": [str(out['trade_date'].min().date()), str(out['trade_date'].max().date())],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
