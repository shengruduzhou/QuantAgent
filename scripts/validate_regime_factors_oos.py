#!/usr/bin/env python3
"""OOS forward-validation of regime factor experts (anti-overfit, backtest-expert).

Takes the IS-selected per-regime factors (from regime_factor_experts_summary.json),
applies them UNCHANGED to an out-of-sample period, and measures per-regime
equal-weight all-A excess at baseline AND stressed (1.5-2x) slippage. The IS
factors are NOT re-fit on OOS — this is a true forward test.

Verdict per regime: PASS if OOS excess > 0 at stressed slippage (edge survives);
the methodology cares about "breaks the least", so we report IS vs OOS side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.ensemble.strict_factor_search import (
    StrictFactorSearchConfig,
    evaluate_strict_factor_subset,
)
from quantagent.ensemble.strict_policy_search import (
    StrictPolicySearchConfig,
    equal_weight_benchmark,
    prepare_decision_chain_panel,
)
from quantagent.risk.regime_family import compute_regime_family

FRAME = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"


def _excess(tw, panel, sector_map, slippage):
    if tw is None or tw.empty:
        return None
    start, end = pd.to_datetime(tw.index.min()), pd.to_datetime(tw.index.max())
    bt_panel = panel[(panel["trade_date"] >= start) & (panel["trade_date"] <= end)
                     & (panel["symbol"].isin(tw.columns.astype(str)))].reset_index(drop=True)
    bt = run_strict_backtest_v8(tw.fillna(0.0), bt_panel, sector_map=sector_map,
                                config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=slippage))
    bench = equal_weight_benchmark(panel, start, end)
    ann = bt.metrics.annualized_return
    bench_ann = bench.get("ann", float("nan"))
    return {"ann": round(ann, 4), "bench_ann": round(bench_ann, 4),
            "excess_ann": round(ann - bench_ann, 4), "max_dd": round(bt.metrics.max_drawdown, 4),
            "sharpe": round(bt.metrics.sharpe, 3), "n_days": int(len(tw))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--is-summary", default="runtime/reports/v8/regime_factor_IS_2021_2023/regime_factor_experts_summary.json")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--oos-end", default="2025-12-31")
    ap.add_argument("--slippage-base", type=float, default=8.0)
    ap.add_argument("--slippage-stress", type=float, default=16.0)
    ap.add_argument("--out", default="runtime/reports/v8/regime_factor_OOS_validation.json")
    args = ap.parse_args()

    summ = json.load(open(args.is_summary))
    is_regimes = {r: v for r, v in summ.get("regimes", {}).items() if isinstance(v, dict) and v.get("status") == "passed"}
    if not is_regimes:
        print("no passed IS regimes in summary"); return 1

    ff = pd.read_parquet(FRAME)
    ff["trade_date"] = pd.to_datetime(ff["trade_date"], errors="coerce"); ff["symbol"] = ff["symbol"].astype(str)
    ff = ff[(ff["trade_date"] >= pd.Timestamp(args.oos_start)) & (ff["trade_date"] <= pd.Timestamp(args.oos_end))].reset_index(drop=True)
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce"); panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[(panel["trade_date"] >= pd.Timestamp(args.oos_start) - pd.Timedelta(days=260)) & (panel["trade_date"] <= pd.Timestamp(args.oos_end))]
    panel = panel[panel["symbol"].isin(set(ff["symbol"].unique()))].reset_index(drop=True)
    sector_map = pd.read_parquet(SECTOR)

    dcfg = StrictPolicySearchConfig(top_k=30, candidate_pool_size=60, min_avg_amount_yuan=5e7,
                                    liquidity_window=60, slippage_bps=args.slippage_base, initial_cash=1_000_000.0,
                                    return_weight=0.5, excess_weight=2.0, drawdown_penalty=0.35,
                                    turnover_penalty=0.02, cost_penalty=0.5)
    prepared = prepare_decision_chain_panel(panel, dcfg, sector_map=sector_map)
    regime_by_date = compute_regime_family(prepared)

    out = {"oos_window": f"{args.oos_start}..{args.oos_end}",
           "oos_regime_days": {str(k): int(v) for k, v in regime_by_date.value_counts().to_dict().items()}, "regimes": {}}
    for regime, info in is_regimes.items():
        factors = info["best_factors"]; top_k = info["best_top_k"]; signs = info.get("factor_signs")
        cfg = StrictFactorSearchConfig(regime_filter=regime, top_k_values=(top_k,), prefix_sizes=(len(factors),),
                                       return_weight=0.5, excess_weight=2.0, drawdown_penalty=0.35,
                                       turnover_penalty=0.02, cost_penalty=0.5, decision=dcfg)
        regime_dates = set(regime_by_date[regime_by_date == regime].index)
        filtered = ff[ff["trade_date"].isin(regime_dates)].reset_index(drop=True)
        if filtered.empty:
            out["regimes"][regime] = {"status": "no_oos_days"}; continue
        try:
            ev = evaluate_strict_factor_subset(factor_frame=filtered, factors=factors, top_k=top_k,
                market_panel=prepared, sector_map=sector_map, sector_pool=None, config=cfg,
                factor_signs=signs, write_backtest=False)
        except Exception as exc:
            out["regimes"][regime] = {"status": f"eval_error:{exc}"}; continue
        base = _excess(ev.target_weights, prepared, sector_map, args.slippage_base)
        stress = _excess(ev.target_weights, prepared, sector_map, args.slippage_stress)
        is_excess = info.get("best_metrics", {}).get("excess_return_ann")
        verdict = "PASS" if (stress and stress["excess_ann"] > 0) else "FAIL"
        out["regimes"][regime] = {"factors": factors, "top_k": top_k,
            "is_excess_ann": round(float(is_excess), 4) if is_excess is not None else None,
            "oos_excess_base": base, "oos_excess_stress": stress, "verdict": verdict}
        print(f"[{regime}] IS_excess={is_excess} | OOS_base={base['excess_ann'] if base else None} "
              f"| OOS_stress({args.slippage_stress}bps)={stress['excess_ann'] if stress else None} -> {verdict}")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
