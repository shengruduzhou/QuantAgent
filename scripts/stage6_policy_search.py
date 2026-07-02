#!/usr/bin/env python3
"""Stage 6/7 portfolio policy search over full-universe OOS predictions.

Objective = after-cost absolute annualised return (CAGR). Searches portfolio
*construction* policies (NOT re-training the model) over the fixed OOS
predictions, ranks by CAGR while always showing max-drawdown, turnover, Sharpe
and win-rate (so high-CAGR/high-turnover overfit is visible), and benchmarks
against the same tradable-universe equal-weight basket and eqw-all-A.

Decision (item 6): if every policy underperforms the universe basket, stop
packaging this model and move to the feature/model fix.

Usage:
    AI_quant_venv/bin/python3 scripts/stage6_policy_search.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from quantagent.portfolio.policy_search import (
    PolicyConfig,
    annualised_metrics,
    prepare_working_frame,
    search_policies,
    universe_benchmark,
)

PREDS = "runtime/stage6_full_walkforward/wf/walkforward_predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
OUT = Path("runtime/stage6_policy_search")
# Stored reference: the v8.9 classical/ensemble book (different model, same evaluator).
V89_REFERENCE = {"name": "v8.9_classical_ensemble (stored ref)", "cagr": 0.173,
                 "max_drawdown": -0.109, "sharpe": None, "note": "from baseline_protocol variant-C, different model"}


def build_grid(horizons: list[int] | None = None) -> list[PolicyConfig]:
    horizons = horizons or [1, 5, 20]
    top_ks = [20, 50, 100, 200, 500]
    rebalances = [1, 5, 20]
    sides = ["long_only", "long_short"]
    neutralizes = ["none", "industry"]
    liquidity = ["none", "ex_bottom_30pct"]
    cfgs = []
    for h, k, rb, side, neu, liq in itertools.product(horizons, top_ks, rebalances, sides, neutralizes, liquidity):
        transform = "csrank" if neu != "none" else "raw"
        cfgs.append(PolicyConfig(horizon=h, top_k=k, rebalance_days=rb, side=side,
                                 transform=transform, neutralize=neu, liquidity_filter=liq))
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps-per-turnover", type=float, default=13.0)
    ap.add_argument("--predictions", default=PREDS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--score-column", default="",
                    help="Single composite score column (e.g. composite_score) used for ALL horizons; "
                         "when set the horizon grid collapses to one value (the score is horizon-agnostic).")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(args.predictions)
    horizons_grid = [1, 5, 20]
    if args.score_column:
        if args.score_column not in preds.columns:
            raise KeyError(f"--score-column '{args.score_column}' not in {list(preds.columns)}")
        preds = preds.rename(columns={args.score_column: "alpha_5d"})
        preds["alpha_1d"] = preds["alpha_5d"]
        preds["alpha_20d"] = preds["alpha_5d"]
        horizons_grid = [5]   # one horizon — score is the same across all
    pcols = ["symbol", "trade_date", "close", "amount", "is_st", "is_suspended", "is_limit_up"]
    panel = pd.read_parquet(PANEL, columns=pcols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    sector = pd.read_parquet(SECTOR)
    work = prepare_working_frame(preds, panel, sector)
    start, end = work["trade_date"].min(), work["trade_date"].max()
    print(f"OOS window {start.date()}..{end.date()} | dates {work['trade_date'].nunique()} | symbols {work['symbol'].nunique()}", flush=True)

    # Benchmarks over the same window.
    uni = universe_benchmark(work)
    allA = panel[(panel["trade_date"] >= start) & (panel["trade_date"] <= end)].copy()
    allA["ret"] = allA.sort_values(["symbol", "trade_date"]).groupby("symbol")["close"].pct_change()
    allA = allA[~allA["is_st"].fillna(False).astype(bool) & ~allA["is_suspended"].fillna(False).astype(bool)]
    allA_bm = annualised_metrics(allA.groupby("trade_date")["ret"].mean())
    benchmarks = {
        "tradable_universe_eqw": uni,
        "eqw_all_A": allA_bm,
        "v8.9_reference": V89_REFERENCE,
    }
    print("BENCHMARKS:")
    print(f"  tradable_universe_eqw : CAGR {uni['cagr']:+.2%}  maxDD {uni['max_drawdown']:+.2%}  Sharpe {uni['sharpe']:.2f}")
    print(f"  eqw_all_A             : CAGR {allA_bm['cagr']:+.2%}  maxDD {allA_bm['max_drawdown']:+.2%}")
    print(f"  v8.9_reference (stored): CAGR {V89_REFERENCE['cagr']:+.2%}  maxDD {V89_REFERENCE['max_drawdown']:+.2%}", flush=True)

    grid = build_grid(horizons_grid)
    for c in grid:
        object.__setattr__(c, "cost_bps_per_turnover", args.cost_bps_per_turnover)
    print(f"\nSearching {len(grid)} policies (cost {args.cost_bps_per_turnover} bps/turnover)...", flush=True)
    lb = search_policies(work, grid, progress=True)
    lb["calmar"] = lb["cagr"] / lb["max_drawdown"].abs()
    lb["beats_universe"] = lb["cagr"] > uni["cagr"]
    cols = ["policy", "horizon", "top_k", "rebalance_days", "side", "neutralize", "liquidity_filter",
            "cagr", "max_drawdown", "calmar", "sharpe", "annual_turnover", "win_rate_daily", "beats_universe", "n_days"]
    lb = lb[[c for c in cols if c in lb.columns]]
    lb.to_csv(out_dir / "strategy_leaderboard.csv", index=False)
    (out_dir / "benchmarks.json").write_text(json.dumps(benchmarks, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    n_beat = int(lb["beats_universe"].sum())
    best = lb.iloc[0]
    print("\n=== TOP 15 POLICIES by after-cost CAGR ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(lb.head(15).to_string(index=False))
    print(f"\nbest after-cost CAGR = {best['cagr']:+.2%} (maxDD {best['max_drawdown']:+.2%}, "
          f"calmar {best['calmar']:.2f}, ann.turnover {best['annual_turnover']:.1f}x) — policy {best['policy']}")
    print(f"policies beating tradable-universe basket ({uni['cagr']:+.2%}): {n_beat}/{len(lb)}")

    decision = ("PROCEED: at least one policy beats the universe basket — refine/confirm via strict simulator."
                if n_beat > 0 else
                "STOP PACKAGING THIS MLP: NO policy beats the universe basket → move to feature/model fix "
                "(cross-sectional per-day rank/zscore features, factor neutralisation, LightGBM/Ridge/ensemble, "
                "top-bucket/ranking objective, or the v8.9 classical book).")
    print("\nDECISION:", decision)
    summary = {"window": [str(start.date()), str(end.date())], "n_policies": len(lb),
               "benchmarks": benchmarks, "best_policy": best.to_dict(),
               "n_beat_universe": n_beat, "decision": decision}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
