#!/usr/bin/env python3
"""Deterministic value test for accepted discovered factors (no GPU needed).

Builds a cross-sectional z-score composite of the accepted formulas, then
runs the SAME strict protocol (flags + eligibility filter, variant B) for:

  composite alone                      — do the discovered factors carry
                                         executable OOS alpha by themselves?
  z(model alpha) + lambda * z(composite) — do they ADD to the v8 model sleeve
                                         before any retrain?

against the pure model sleeve and both benchmarks (paper + executable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_protocol import _bench_daily, _regime_excess, _target_weights  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402
from quantagent.factors.factor_synthesis import load_definitions  # noqa: E402

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
ANN = 244


def _zscore_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = values.groupby(dates, sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return (values - mean) / std


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--definitions", required=True, help="accepted_definitions.json")
    ap.add_argument("--predictions", default="runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet")
    ap.add_argument("--start", default="2024-08-28")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--lambdas", default="0.25,0.5,1.0")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--lookback-days", type=int, default=180, help="extra history loaded for rolling windows")
    ap.add_argument("--output", default="runtime/reports/v8/discovery/composite_test.json")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    if args.end:
        panel = panel[panel["trade_date"] <= pd.Timestamp(args.end)]
    sector = pd.read_parquet(SECTOR)

    # Factor computation needs rolling history before the start date.
    hist = panel[panel["trade_date"] >= start - pd.Timedelta(days=args.lookback_days)]
    hist = hist.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    definitions = load_definitions(args.definitions)
    if not definitions:
        raise SystemExit(f"no definitions in {args.definitions}")
    print(f"computing {len(definitions)} accepted factors on {hist['symbol'].nunique()} symbols ...")
    zsum = pd.Series(0.0, index=hist.index)
    used = 0
    for definition in definitions:
        try:
            vals = pd.to_numeric(definition.expr.evaluate(hist), errors="coerce").replace([np.inf, -np.inf], np.nan)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {definition.name}: {exc}")
            continue
        z = _zscore_by_date(vals, hist["trade_date"]).clip(-5, 5)
        zsum = zsum.add(z.fillna(0.0))
        used += 1
    if not used:
        raise SystemExit("no factor evaluated successfully")
    hist["composite"] = zsum / used

    scores = hist.loc[hist["trade_date"] >= start,
                      ["symbol", "trade_date", "composite", "is_suspended", "is_st", "is_limit_up"]].copy()

    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= start]
    if args.end:
        preds = preds[preds["trade_date"] <= pd.Timestamp(args.end)]
    df = preds.merge(scores, on=["symbol", "trade_date"], how="inner")
    df["z_alpha"] = _zscore_by_date(pd.to_numeric(df["alpha_score"], errors="coerce"), df["trade_date"])
    df["z_comp"] = _zscore_by_date(df["composite"], df["trade_date"])

    sim_panel = panel[panel["trade_date"] >= start - pd.Timedelta(days=10)]
    trade_dates = sorted(sim_panel["trade_date"].unique())
    bench = _bench_daily(sim_panel, sorted(df["trade_date"].unique()))
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)

    def _run(name: str, score_col: str) -> dict:
        tw = _target_weights(df.assign(alpha_score=df[score_col]), "alpha_score", args.top_k,
                             eligible_only=True, delay_days=0, trade_dates=trade_dates)
        res = run_strict_backtest_v8(tw, sim_panel, sector_map=sector,
                                     config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0,
                                                                            slippage_bps=args.slippage_bps))
        m = res.metrics
        rec = {"ann": round(m.annualized_return, 4), "excess_ann": round(m.annualized_return - bench_ann, 4),
               "sharpe": round(m.sharpe, 3), "maxDD": round(m.max_drawdown, 4),
               "regime": _regime_excess(res.nav, bench)}
        print(f"{name:24} ann {m.annualized_return:+8.2%} | excess {m.annualized_return - bench_ann:+8.2%} | "
              f"sharpe {m.sharpe:5.2f} | maxDD {m.max_drawdown:6.2%}")
        return rec

    out = {"bench_ann": round(bench_ann, 4), "n_factors": used, "top_k": args.top_k,
           "start": args.start, "definitions": args.definitions, "runs": {}}
    out["runs"]["model_only"] = _run("model_only", "z_alpha")
    out["runs"]["composite_only"] = _run("composite_only", "z_comp")
    for lam in [float(x) for x in args.lambdas.split(",") if x.strip()]:
        df[f"blend_{lam}"] = df["z_alpha"].fillna(0.0) + lam * df["z_comp"].fillna(0.0)
        out["runs"][f"blend_lambda_{lam}"] = _run(f"blend λ={lam}", f"blend_{lam}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
