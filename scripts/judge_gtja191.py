#!/usr/bin/env python3
"""Run GTJA-191 tranche-1 factors through the SAME unified judgment protocol.

Computes the factors on a symbol-sample of the market panel and judges them
with the engine from factor_full_judgment (eligible pool, 5/20/60d horizons,
per-year 2022-2026, per-regime, capacity). Appends results to the master
judgment table with family="gtja191".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_full_judgment import DailyICEngine, HORIZONS, _judge_factor, _regime_from_bench  # noqa: E402

from quantagent.factors.gtja191 import compute_gtja191_factors, gtja191_names  # noqa: E402

DATASET = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--gate-years", default=None)
    ap.add_argument("--symbols-sample", type=int, default=1200)
    ap.add_argument("--output-dir", default="runtime/reports/v8/factor_full_judgment")
    ap.add_argument("--no-merge-master", action="store_true",
                    help="write gtja191_judgment.csv only; don't touch factor_judgment_table.csv")
    args = ap.parse_args()

    if args.gate_years:
        import factor_full_judgment as fj
        gates = tuple(int(y) for y in args.gate_years.split(",") if y.strip())
        fj.set_gate_years(tuple(sorted(set(gates) | set(fj.YEARS))), gates)

    base = pd.read_parquet(DATASET, columns=[
        "symbol", "trade_date", "return_1d", "amount", "is_st", "is_suspended", "is_limit_up",
        "forward_return_5d", "forward_return_20d", "forward_return_60d"])
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base = base[base["trade_date"] >= pd.Timestamp(args.start)].reset_index(drop=True)
    if args.end:
        base = base[base["trade_date"] <= pd.Timestamp(args.end)].reset_index(drop=True)
    bench = base.groupby("trade_date")["return_1d"].mean()
    regime_by_date = _regime_from_bench(bench)

    rng = np.random.default_rng(7)
    syms = sorted(base["symbol"].unique())
    keep = set(rng.choice(syms, size=min(args.symbols_sample, len(syms)), replace=False))
    sub = base[base["symbol"].isin(keep)].reset_index(drop=True)
    eligible = ~(
        sub["is_st"].fillna(False).astype(bool)
        | sub["is_suspended"].fillna(False).astype(bool)
        | sub["is_limit_up"].fillna(False).astype(bool)
    ).to_numpy()
    engine = DailyICEngine(sub["trade_date"], {h: sub[f"forward_return_{h}d"] for h in HORIZONS}, eligible)
    amount = pd.to_numeric(sub["amount"], errors="coerce")

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["symbol"].isin(keep))
                  & (panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=200))]
    if args.end:
        panel = panel[panel["trade_date"] <= pd.Timestamp(args.end)]
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    print(f"computing {len(gtja191_names())} GTJA factors on {panel['symbol'].nunique()} symbols ...")
    wide = compute_gtja191_factors(panel, wide=True)
    wide["trade_date"] = pd.to_datetime(wide["trade_date"])

    rows = []
    for name in gtja191_names():
        merged = sub[["symbol", "trade_date"]].merge(
            wide[["symbol", "trade_date", name]], on=["symbol", "trade_date"], how="left")
        row = {"factor": name, "family": "gtja191", "source": "tranche1"}
        row.update(_judge_factor(engine, merged[name], regime_by_date, amount, eligible, sub["trade_date"]))
        rows.append(row)
        print(f"{name:10} {row.get('verdict','?'):18} best={row.get('best_horizon','-'):4} "
              f"ic5={row.get('ic_5d')} ic60={row.get('ic_60d')}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "gtja191_judgment.csv", index=False)
    master = out_dir / "factor_judgment_table.csv"
    if master.exists() and not args.no_merge_master:
        existing = pd.read_csv(master)
        existing = existing[existing["family"] != "gtja191"]
        pd.concat([existing, table], ignore_index=True).to_csv(master, index=False)
    elif args.no_merge_master is False and not master.exists():
        table.to_csv(master, index=False)
    print(json.dumps({"n": len(table), "verdicts": table["verdict"].value_counts().to_dict()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
