#!/usr/bin/env python3
"""Run the end-to-end cost-sensitive intraday Do-T EV closed loop on TickFlow.

This is the real-data driver for the EV stack: it builds causal per-minute
features + round-trip labels on the held book minute panel, trains calibrated
models on train+validation dates, then simulates ``decide_ev`` minute-by-minute
over a T+1 ledger on the *unseen* test dates with conservative next-bar fills,
and emits a deployment-gate verdict with permutation-null baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path

import pandas as pd

from quantagent.research.intraday_dot_ev_backtest import (
    EVBacktestConfig,
    build_book_minute_panel,
    build_feature_label_table,
    load_book_keys,
    run_ev_closed_loop,
)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minute-dir", default="runtime/data/v7/silver/minute_bars")
    ap.add_argument("--holdings-csv", default="runtime/paper/replay_2026/holdings_daily.csv")
    ap.add_argument("--market-panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--output-dir", default="runtime/reports/intraday_dot_ev_backtest")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-06-12")
    ap.add_argument("--train-end", default="2026-02-27")
    ap.add_argument("--validation-end", default="2026-04-15")
    ap.add_argument("--order-notional-yuan", type=float, default=100_000.0)
    ap.add_argument("--horizon-minutes", type=int, default=60)
    ap.add_argument("--backend", choices=["lightgbm", "xgboost", "catboost", "sklearn"], default="lightgbm")
    ap.add_argument("--edge-cost-multiple", type=float, default=2.0)
    ap.add_argument("--min-round-trips-enable", type=int, default=300)
    ap.add_argument("--maker-only", action="store_true",
                    help="Use a maker/limit execution channel (~10bps round-trip) instead of retail taker.")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--spread-bps", type=float, default=6.0)
    ap.add_argument("--commission-rate", type=float, default=0.0003)
    ap.add_argument("--max-symbols", type=int, default=0, help="0 = all held symbols")
    ap.add_argument("--cache-table", default="", help="Optional parquet path to cache the feature+label table")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slippage = 2.0 if args.maker_only else args.slippage_bps
    spread = 2.0 if args.maker_only else args.spread_bps
    commission = 0.0001 if args.maker_only else args.commission_rate
    cfg = EVBacktestConfig(
        start=args.start, end=args.end, train_end=args.train_end, validation_end=args.validation_end,
        order_notional_yuan=args.order_notional_yuan, horizon_minutes=args.horizon_minutes,
        backend=args.backend, edge_cost_multiple=args.edge_cost_multiple,
        min_round_trips_enable=args.min_round_trips_enable,
        slippage_bps=slippage, spread_bps=spread, commission_rate=commission,
    )

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    table = None
    cache_path = Path(args.cache_table) if args.cache_table else None
    if cache_path and cache_path.exists():
        print(f"=== reuse cached feature+label table: {cache_path} ===", flush=True)
        table = pd.read_parquet(cache_path)
    else:
        print("=== Phase 1: build held book minute panel ===", flush=True)
        book_keys = load_book_keys(args.holdings_csv, start, end)
        if args.max_symbols:
            keep = sorted(book_keys["symbol"].unique())[: args.max_symbols]
            book_keys = book_keys[book_keys["symbol"].isin(keep)]
        print(json.dumps({"book_symbol_days": int(len(book_keys)),
                          "book_symbols": int(book_keys["symbol"].nunique())}), flush=True)
        panel = build_book_minute_panel(minute_dir=args.minute_dir, book_keys=book_keys,
                                        panel_path=args.market_panel, start=start, end=end)
        print(f"minute panel rows: {len(panel):,}", flush=True)
        print("=== Phase 2: build causal features + round-trip labels ===", flush=True)
        table = build_feature_label_table(panel, cfg)
        table = table.merge(
            panel[["symbol", "trade_date", "trade_time", "open", "high", "low", "close",
                   "volume", "limit_up", "limit_down"]],
            on=["symbol", "trade_date", "trade_time"], how="left", suffixes=("", "_bar"))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(cache_path, index=False)
            print(f"cached table -> {cache_path}", flush=True)

    print("=== Phase 3: walk-forward train + closed-loop simulate test dates ===", flush=True)
    result = run_ev_closed_loop(minute_dir=args.minute_dir, holdings_csv=args.holdings_csv,
                                panel_path=args.market_panel, cfg=cfg, feature_label_table=table)

    if not result.trades.empty:
        result.trades.to_parquet(out / "ev_trades.parquet", index=False)
    if result.models is not None:
        try:
            from quantagent.training.do_t_models import save_models
            save_models(result.models, out / "do_t_models.joblib")
        except Exception as exc:  # noqa: BLE001
            print(f"model save skipped: {exc}", flush=True)
    payload = _json_ready({
        "verdict": result.verdict,
        "reason": result.reason,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "n_train_rows": result.n_train_rows,
        "n_test_symbol_days": result.n_test_symbol_days,
        "execution_channel": "maker_only_10bps_rt" if args.maker_only else "retail_taker",
    })
    (out / "ev_backtest_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
