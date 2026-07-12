#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): 做T on 1-min OHLCV: no realizable edge (stage3b/4 REJECT).
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Regime-conditional do-T edge analysis.

Tests the hypothesis that intraday do-T edge is *conditional on day type*:
positive on 震荡 / reversal days (低开高走、高开低走) and negative on trend days
(高开高走、低开低走) and in bear tape -- so a pooled average washes it out.

Reuses the cached feature+label table + a saved EV model.  Classifies each held
symbol-day by gap + intraday trajectory + efficiency ratio, then reports the
realized net edge of the model's top-predicted minutes WITHIN each day-type, at
maker and retail cost.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import CAUSAL_INTRADAY_FEATURE_COLUMNS
from quantagent.training.do_t_models import load_models, train_do_t_models, predict_model_signals


def classify_day(g: pd.DataFrame) -> dict:
    o = pd.to_numeric(g["open"], errors="coerce").to_numpy()
    c = pd.to_numeric(g["close"], errors="coerce").to_numpy()
    h = pd.to_numeric(g["high"], errors="coerce").to_numpy()
    l = pd.to_numeric(g["low"], errors="coerce").to_numpy()
    day_open = float(o[0]); day_close = float(c[-1])
    gap = float(g["gap_open"].iloc[0]) if "gap_open" in g.columns and np.isfinite(g["gap_open"].iloc[0]) else 0.0
    day_ret = day_close / day_open - 1.0 if day_open > 0 else 0.0
    rng = float(np.nanmax(h) - np.nanmin(l))
    eff = abs(day_close - day_open) / rng if rng > 1e-9 else 0.0  # 0=choppy, 1=trending
    g_th, r_th = 0.003, 0.003
    if abs(day_ret) < r_th or eff < 0.30:
        dtype = "震荡choppy"
    elif gap > g_th and day_ret > r_th:
        dtype = "高开高走trendUp"
    elif gap > g_th and day_ret < -r_th:
        dtype = "高开低走revDown"
    elif gap < -g_th and day_ret > r_th:
        dtype = "低开高走revUp"
    elif gap < -g_th and day_ret < -r_th:
        dtype = "低开低走trendDn"
    elif day_ret > r_th:
        dtype = "平开高走"
    else:
        dtype = "平开低走"
    return {"day_type": dtype, "gap": gap, "day_ret": day_ret, "efficiency": eff}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-table", default="runtime/reports/intraday_dot_ev_full/feature_label_table.parquet")
    ap.add_argument("--models", default="runtime/reports/intraday_dot_ev_full/do_t_models.joblib")
    ap.add_argument("--validation-end", default="2026-04-15")
    ap.add_argument("--backend", default="lightgbm")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    all_cols = set(pq.ParquetFile(args.cache_table).schema.names)
    feat = [c for c in CAUSAL_INTRADAY_FEATURE_COLUMNS if c in all_cols]
    need = feat + [c for c in ["symbol", "trade_date", "trade_time", "open", "high", "low", "close",
                               "gap_open", "label_sell_high_gross_edge_bps", "label_buy_low_gross_edge_bps"]
                   if c in all_cols]
    table = pd.read_parquet(args.cache_table, columns=sorted(set(need)))
    table["trade_date"] = pd.to_datetime(table["trade_date"]).dt.normalize()
    table["symbol"] = table["symbol"].astype(str)
    val_end = pd.Timestamp(args.validation_end)

    try:
        models = load_models(args.models)
    except Exception:
        train = table[table["trade_date"] <= val_end]
        models = train_do_t_models(train, feature_columns=feat, backend=args.backend, allow_sklearn_fallback=True)

    # classify only the TEST symbol-days (cheap)
    test = table[table["trade_date"] > val_end].copy()
    del table
    import gc; gc.collect()
    meta_rows = []
    for (sym, day), g in test.groupby(["symbol", "trade_date"], sort=False):
        meta_rows.append({"symbol": sym, "trade_date": day, **classify_day(g)})
    daymeta = pd.DataFrame(meta_rows)
    test = test.merge(daymeta, on=["symbol", "trade_date"], how="left")

    sigs = predict_model_signals(models, test)
    test = test.reset_index(drop=True)
    test["pred_sell_gross"] = [s.expected_sell_high_gain_bps for s in sigs]
    test["pred_sell_p"] = [s.p_sell_high_success for s in sigs]
    test["pred_buy_gross"] = [s.expected_buy_low_gain_bps for s in sigs]
    test["pred_buy_p"] = [s.p_buy_low_success for s in sigs]

    out = {"test_symbol_days": int(test.drop_duplicates(["symbol", "trade_date"]).shape[0]), "by_day_type": {}}
    for cost in (10.0, 39.2):
        test["ev_sell"] = test["pred_sell_p"] * test["pred_sell_gross"] - cost
        test["ev_buy"] = test["pred_buy_p"] * test["pred_buy_gross"] - cost
        test["best_ev"] = test[["ev_sell", "ev_buy"]].max(axis=1)
        test[f"realized_net_{cost}"] = np.where(
            test["ev_sell"] >= test["ev_buy"],
            pd.to_numeric(test["label_sell_high_gross_edge_bps"], errors="coerce") - cost,
            pd.to_numeric(test["label_buy_low_gross_edge_bps"], errors="coerce") - cost,
        )

    for dtype, g in test.groupby("day_type", sort=False):
        days = int(g.drop_duplicates(["symbol", "trade_date"]).shape[0])
        rec = {"symbol_days": days, "minutes": int(len(g))}
        for cost in (10.0, 39.2):
            col = f"realized_net_{cost}"
            thr = g["best_ev"].quantile(0.95)
            top = g[(g["best_ev"] >= thr) & g[col].notna()]
            rec[f"cost{cost}"] = {
                "top5pct_n": int(len(top)),
                "top5pct_mean_net_bps": round(float(top[col].mean()), 2) if len(top) else None,
                "top5pct_hit": round(float((top[col] > 0).mean()), 3) if len(top) else None,
                "all_achievable_mean_net_bps": round(float(np.nanmean(np.maximum(
                    pd.to_numeric(g["label_sell_high_gross_edge_bps"], errors="coerce"),
                    pd.to_numeric(g["label_buy_low_gross_edge_bps"], errors="coerce")) - cost)), 2),
            }
        out["by_day_type"][dtype] = rec
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
