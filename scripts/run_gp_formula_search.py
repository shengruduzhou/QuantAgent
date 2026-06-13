#!/usr/bin/env python3
"""Run the symbolic-GA formula-alpha search on real full-universe data.

Anti-overfit protocol:
- The GA only ever sees dates <= --train-end (default 2024-07-31, the v8
  deep-model train cutoff), so everything after stays a clean OOS window
  for the discovered factors.
- Inside that window the GA itself holds out the last validation_fraction
  of dates chronologically.
- Survivors must clear validation RankIC, finite-ratio, mutual-correlation
  and existing-factor-correlation gates.

Run several seeds (different contiguous fitness blocks) and merge with
``scripts/merge_factor_definitions.py`` before OOS evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


DEFAULT_PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
DEFAULT_LABELS = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
DEFAULT_REFERENCES = "alpha016,alpha015,alpha050,alpha044,alpha040,alpha161,alpha163,alpha088,alpha145"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market-panel", default=DEFAULT_PANEL)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-column", default="forward_return_5d")
    ap.add_argument("--train-end", default="2024-07-31", help="GA never sees dates after this.")
    ap.add_argument("--population", type=int, default=240)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--fitness-sample-dates", type=int, default=350)
    ap.add_argument("--fitness-sample-symbols", type=int, default=600)
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--min-validation-rank-ic", type=float, default=0.01)
    ap.add_argument("--max-correlation", type=float, default=0.7)
    ap.add_argument("--reference-columns", default=DEFAULT_REFERENCES)
    ap.add_argument("--max-reference-correlation", type=float, default=0.6)
    ap.add_argument("--warm-start-fraction", type=float, default=0.4)
    ap.add_argument("--icir-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    from quantagent.factors.factor_synthesis import SymbolicGAConfig, save_result, synthesize_factors

    references = tuple(c.strip() for c in args.reference_columns.split(",") if c.strip())
    panel = pd.read_parquet(
        args.market_panel,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"],
    )
    labels = pd.read_parquet(
        args.labels,
        columns=["symbol", "trade_date", args.label_column, *references],
    )
    train_end = pd.Timestamp(args.train_end)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
    panel = panel[panel["trade_date"] <= train_end]
    labels = labels[labels["trade_date"] <= train_end]
    if panel.empty or labels.empty:
        raise SystemExit(f"no data on or before --train-end {args.train_end}")

    config = SymbolicGAConfig(
        population=args.population,
        generations=args.generations,
        max_depth=args.max_depth,
        top_k=args.top_k,
        label_column=args.label_column,
        validation_fraction=args.validation_fraction,
        min_validation_rank_ic=args.min_validation_rank_ic,
        max_correlation=args.max_correlation,
        fitness_sample_dates=args.fitness_sample_dates,
        fitness_sample_symbols=args.fitness_sample_symbols,
        seed=args.seed,
        warm_start_fraction=args.warm_start_fraction,
        icir_weight=args.icir_weight,
        reference_columns=references,
        max_reference_correlation=args.max_reference_correlation,
    )
    result = synthesize_factors(panel, labels=labels, config=config)
    out_dir = Path(args.output_dir)
    paths = save_result(result, out_dir)
    summary = {
        "status": "passed",
        "selected": len(result.definitions),
        "train_end": args.train_end,
        "config": asdict(config),
        **paths,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
