#!/usr/bin/env python3
"""Edge-frontier diagnostic for the intraday Do-T EV model.

Answers the only question that matters before tuning anything: do the model's
positive-EV predictions correspond to positive *realized* out-of-sample net
edge?  Trains on train+validation dates, predicts on the unseen test dates, then
measures rank-IC of predicted vs realized edge and the realized net edge of the
top predicted-EV minutes across a range of execution-cost assumptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import CAUSAL_INTRADAY_FEATURE_COLUMNS
from quantagent.research.intraday_dot_ev_backtest import (
    EVBacktestConfig, build_book_minute_panel, build_feature_label_table, load_book_keys,
)
from quantagent.training.do_t_models import train_do_t_models, predict_model_signals


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 50 or a[m].nunique() < 3 or b[m].nunique() < 3:
        return float("nan")
    return float(a[m].rank().corr(b[m].rank()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute-dir", default="runtime/data/v7/silver/minute_bars")
    ap.add_argument("--holdings-csv", default="runtime/paper/replay_2026/holdings_daily.csv")
    ap.add_argument("--market-panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--cache-table", default="runtime/reports/intraday_dot_ev_full/feature_label_table.parquet")
    ap.add_argument("--max-symbols", type=int, default=120)
    ap.add_argument("--backend", default="sklearn")
    args = ap.parse_args()

    cfg = EVBacktestConfig(backend=args.backend)
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    cache = Path(args.cache_table)
    if cache.exists():
        table = pd.read_parquet(cache)
        if args.max_symbols:
            keep = sorted(table["symbol"].astype(str).unique())[: args.max_symbols]
            table = table[table["symbol"].astype(str).isin(keep)]
    else:
        bk = load_book_keys(args.holdings_csv, start, end)
        keep = sorted(bk["symbol"].unique())[: args.max_symbols] if args.max_symbols else None
        if keep is not None:
            bk = bk[bk["symbol"].isin(keep)]
        panel = build_book_minute_panel(minute_dir=args.minute_dir, book_keys=bk,
                                        panel_path=args.market_panel, start=start, end=end)
        table = build_feature_label_table(panel, cfg)

    table["trade_date"] = pd.to_datetime(table["trade_date"]).dt.normalize()
    feat = [c for c in CAUSAL_INTRADAY_FEATURE_COLUMNS if c in table.columns]
    val_end = pd.Timestamp(cfg.validation_end)
    tr = table[table["trade_date"] <= val_end].copy()
    te = table[table["trade_date"] > val_end].copy()
    print(json.dumps({"train_rows": len(tr), "test_rows": len(te),
                      "test_symbol_days": te.drop_duplicates(['symbol','trade_date']).shape[0]}))

    models = train_do_t_models(tr, feature_columns=feat, backend=args.backend, allow_sklearn_fallback=True)
    sigs = predict_model_signals(models, te)
    te = te.reset_index(drop=True)
    te["pred_sell_gross"] = [s.expected_sell_high_gain_bps for s in sigs]
    te["pred_sell_p"] = [s.p_sell_high_success for s in sigs]
    te["pred_buy_gross"] = [s.expected_buy_low_gain_bps for s in sigs]
    te["pred_buy_p"] = [s.p_buy_low_success for s in sigs]
    # model "raw EV" before any floor gate, for several round-trip cost assumptions
    out = {"rank_IC": {}, "edge_curve": {}}
    out["rank_IC"]["sell_gross_vs_realized_gross"] = _ic(te["pred_sell_gross"], te["label_sell_high_gross_edge_bps"])
    out["rank_IC"]["sell_p_vs_realized_success"] = _ic(te["pred_sell_p"], te["label_sell_high_success"])
    out["rank_IC"]["buy_gross_vs_realized_gross"] = _ic(te["pred_buy_gross"], te["label_buy_low_gross_edge_bps"])
    out["rank_IC"]["buy_p_vs_realized_success"] = _ic(te["pred_buy_p"], te["label_buy_low_success"])

    # expected EV (model) = p*gross - cost ; check realized NET edge of the top predicted minutes
    for cost in (10.0, 20.0, 39.2, 60.0):
        te["ev_sell"] = te["pred_sell_p"] * te["pred_sell_gross"] - cost
        te["ev_buy"] = te["pred_buy_p"] * te["pred_buy_gross"] - cost
        te["best_ev"] = te[["ev_sell", "ev_buy"]].max(axis=1)
        te["best_side"] = np.where(te["ev_sell"] >= te["ev_buy"], "sell", "buy")
        te["realized_net"] = np.where(
            te["best_side"] == "sell",
            te["label_sell_high_gross_edge_bps"] - cost,
            te["label_buy_low_gross_edge_bps"] - cost,
        )
        rec = {}
        for label, mask in (
            ("ev>0", te["best_ev"] > 0),
            ("ev>8", te["best_ev"] > 8),
            ("top1pct", te["best_ev"] >= te["best_ev"].quantile(0.99)),
            ("top5pct", te["best_ev"] >= te["best_ev"].quantile(0.95)),
        ):
            sub = te[mask & te["realized_net"].notna()]
            rec[label] = {
                "n": int(len(sub)),
                "mean_realized_net_bps": round(float(sub["realized_net"].mean()), 2) if len(sub) else None,
                "median_realized_net_bps": round(float(sub["realized_net"].median()), 2) if len(sub) else None,
                "hit_rate": round(float((sub["realized_net"] > 0).mean()), 3) if len(sub) else None,
            }
        out["edge_curve"][f"cost_{cost}bps"] = rec
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
