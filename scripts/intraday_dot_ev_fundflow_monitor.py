#!/usr/bin/env python3
"""Autonomous monitor: accumulate 东财 minute fund-flow, auto-retrain when ready.

Run daily (wired into run_forward_daily.sh step 7). Cheap on most days -- it just
records how much forward fund-flow has accumulated. Once >= MIN_DAYS trading days
exist (and grown by RETRAIN_STEP since the last train), it AUTOMATICALLY:

  1. builds the causal feature+label table on the fund-flow-covered held panel,
     merging the order-flow features;
  2. trains the EV models WITH vs WITHOUT the fund-flow features on the earlier
     dates and evaluates the unseen later dates;
  3. records whether order flow lifts rank-IC and the realized net edge of the
     top-predicted minutes at maker cost -- i.e. whether do-T finally has a
     deployable edge.

Status  -> runtime/reports/intraday_dot_ev_fundflow/status.json
Verdict -> runtime/reports/intraday_dot_ev_fundflow/verdict_{days}d.json

This is the "持续监控直到找到最大超额" loop: it surfaces the max-excess result
the moment the data is sufficient, without manual intervention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import (
    CAUSAL_INTRADAY_FEATURE_COLUMNS, FUNDFLOW_FEATURE_COLUMNS,
)
from quantagent.research.intraday_dot_ev_backtest import (
    EVBacktestConfig, build_book_minute_panel, build_feature_label_table,
)
from quantagent.training.do_t_models import train_do_t_models, predict_model_signals

FF_DIR = Path("runtime/data/v7/silver/fundflow_minute")
MINUTE_DIR = "runtime/data/v7/silver/minute_bars"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
OUT = Path("runtime/reports/intraday_dot_ev_fundflow")
MIN_DAYS = 25          # forward trading days before a first train is meaningful
RETRAIN_STEP = 5       # retrain only after this many new days
TEST_FRACTION = 0.30   # last 30% of dates = unseen test


def _load_fundflow() -> pd.DataFrame:
    files = sorted(FF_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["symbol"] = df["symbol"].astype(str)
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df.get("trade_date", df["trade_time"].dt.normalize())).dt.normalize()
    return df.dropna(subset=["symbol", "trade_time"])


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 50 or a[m].nunique() < 3 or b[m].nunique() < 3:
        return float("nan")
    return float(a[m].rank().corr(b[m].rank()))


def _eval(test: pd.DataFrame, feat: list[str], train: pd.DataFrame, cost: float = 10.0) -> dict:
    models = train_do_t_models(train, feature_columns=feat, backend="lightgbm", allow_sklearn_fallback=True)
    sigs = predict_model_signals(models, test)
    te = test.reset_index(drop=True)
    ps = np.array([s.p_sell_high_success for s in sigs]); gs = np.array([s.expected_sell_high_gain_bps for s in sigs])
    pb = np.array([s.p_buy_low_success for s in sigs]); gb = np.array([s.expected_buy_low_gain_bps for s in sigs])
    ev_sell = ps * gs - cost; ev_buy = pb * gb - cost
    best_ev = np.maximum(ev_sell, ev_buy)
    realized = np.where(ev_sell >= ev_buy,
                        pd.to_numeric(te["label_sell_high_gross_edge_bps"], errors="coerce") - cost,
                        pd.to_numeric(te["label_buy_low_gross_edge_bps"], errors="coerce") - cost)
    realized = pd.Series(realized)
    thr = np.nanquantile(best_ev, 0.95)
    top = realized[(best_ev >= thr) & realized.notna()]
    return {
        "rank_IC_sell": _ic(pd.Series(gs), pd.to_numeric(te["label_sell_high_gross_edge_bps"], errors="coerce")),
        "rank_IC_buy": _ic(pd.Series(gb), pd.to_numeric(te["label_buy_low_gross_edge_bps"], errors="coerce")),
        "top5pct_n": int(len(top)),
        "top5pct_mean_net_bps@maker": round(float(top.mean()), 2) if len(top) else None,
        "top5pct_hit": round(float((top > 0).mean()), 3) if len(top) else None,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ff = _load_fundflow()
    if ff.empty:
        days = 0; status = {"fundflow_days": 0, "ready": False, "min_days": MIN_DAYS,
                            "note": "no fund-flow collected yet; run collect_eastmoney_fundflow_minute.py daily after close"}
        (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(json.dumps(status, ensure_ascii=False)); return 0
    dates = sorted(ff["trade_date"].dropna().unique())
    days = len(dates)
    status = {"fundflow_days": days, "ready": days >= MIN_DAYS, "min_days": MIN_DAYS,
              "date_range": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())],
              "symbols": int(ff["symbol"].nunique()),
              "symbol_days": int(ff.drop_duplicates(["symbol", "trade_date"]).shape[0])}
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
    if days < MIN_DAYS:
        status["note"] = f"accumulating: {days}/{MIN_DAYS} forward days"
        print(json.dumps(status, ensure_ascii=False)); return 0

    ckpt = OUT / "last_trained_days.txt"
    last = int(ckpt.read_text()) if ckpt.exists() else 0
    if last > 0 and days - last < RETRAIN_STEP:
        print(json.dumps({**status, "note": f"no retrain (need +{RETRAIN_STEP} days since {last})"}, ensure_ascii=False))
        return 0

    # ---- heavy path: build fund-flow-augmented table, compare WITH vs WITHOUT ----
    keys = ff.drop_duplicates(["symbol", "trade_date"])[["symbol", "trade_date"]].copy()
    keys["weight"] = 1.0
    panel = build_book_minute_panel(minute_dir=MINUTE_DIR, book_keys=keys, panel_path=PANEL,
                                    start=pd.Timestamp(dates[0]), end=pd.Timestamp(dates[-1]))
    if panel.empty:
        print(json.dumps({**status, "note": "no minute bars for fund-flow keys yet"}, ensure_ascii=False)); return 0
    cfg = EVBacktestConfig()
    table = build_feature_label_table(panel, cfg, fundflow_panel=ff)
    table["trade_date"] = pd.to_datetime(table["trade_date"]).dt.normalize()
    cut = dates[int(len(dates) * (1 - TEST_FRACTION))]
    train = table[table["trade_date"] < cut]; test = table[table["trade_date"] >= cut]
    base = [c for c in CAUSAL_INTRADAY_FEATURE_COLUMNS if c in table.columns]
    ff_cols = [c for c in FUNDFLOW_FEATURE_COLUMNS if c in table.columns and table[c].notna().any()]
    verdict = {**status, "test_cut": str(pd.Timestamp(cut).date()),
               "train_rows": int(len(train)), "test_rows": int(len(test)),
               "fundflow_features_active": ff_cols}
    if len(train) > 500 and len(test) > 200 and ff_cols:
        verdict["WITHOUT_fundflow"] = _eval(test, base, train)
        verdict["WITH_fundflow"] = _eval(test, base + ff_cols, train)
        w = verdict["WITH_fundflow"]["top5pct_mean_net_bps@maker"]
        wo = verdict["WITHOUT_fundflow"]["top5pct_mean_net_bps@maker"]
        verdict["fundflow_lifts_edge"] = bool(w is not None and wo is not None and w > wo and w > 0)
        verdict["headline"] = (f"order-flow {'LIFTS' if verdict['fundflow_lifts_edge'] else 'does NOT yet lift'} "
                               f"top-minute net @maker: {wo} -> {w} bps")
    else:
        verdict["note"] = "insufficient train/test rows or no active fund-flow features yet"
    (OUT / f"verdict_{days}d.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    ckpt.write_text(str(days))
    print(json.dumps({k: verdict.get(k) for k in ("fundflow_days", "headline", "fundflow_lifts_edge", "note")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
