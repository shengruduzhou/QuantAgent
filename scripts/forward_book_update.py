#!/usr/bin/env python3
"""Forward daily book update — hold-band targets + 反T signal prep.

Consumes the latest scores (training-run composite ∪ forward inference) and
maintains TWO parallel paper books (A/B test, 2026-06-12 finding):

  A  default band  enter rank≤30 / exit rank>150 / hold 50   (validated 2026)
  B  loose band    enter rank≤50 / exit rank>200 / hold 50   (wider = less
     churn; won H2/2026-chop +15.6% vs +6.1% in the strict replay)

For each book it emits the t+1 target weights, the orders delta vs the
previous forward day, and the 反T (spike_sell 做T) watchlist for tomorrow
morning: held names passing the selective gates (ATR / 5d-trend / regime),
with their per-name trigger levels precomputed from the panel context.

State lives under runtime/paper/forward/{A_default,B_loose}/.
Research ledger only — no orders are placed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.selective_dot import SelectiveDotParams, build_day_contexts, check_gates, DayContext
from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
BASE_PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
FWD_PREDS = "runtime/reports/v8/forward/ensemble_forward.parquet"
ROOT = Path("runtime/paper/forward")

BOOKS = {
    "A_default": HoldBandConfig(n_hold=50, entry_rank=30, exit_rank=150, delay_days=1),
    "B_loose": HoldBandConfig(n_hold=50, entry_rank=50, exit_rank=200, delay_days=1),
}
# 反T watchlist is RESEARCH-ONLY (no trade). Deep-history walk-forward
# (2026-06-12, runtime/reports/dot_selective_deep) killed the live 做T
# overlay: the spike_sell edge existed only in 2026Q1 (+0.12%/leg) and was
# negative in 2025H2 (−0.44%) AND 2026Q2 OOS (−0.51%). The watchlist stays
# emitted so the forward ledger keeps accumulating evidence for free.
DOT_PARAMS = SelectiveDotParams(mode="spike_sell", dip_atr_mult=0.45,
                                target_atr_mult=0.60, stop_atr_mult=0.60,
                                morning_deadline="10:30:00",
                                min_atr_pct=0.025, max_mom_5d=-0.02)


def _load_scores() -> pd.DataFrame:
    base = pd.read_parquet(BASE_PREDS, columns=["trade_date", "symbol", "composite_score"])
    frames = [base]
    if Path(FWD_PREDS).exists():
        fwd = pd.read_parquet(FWD_PREDS)
        score_col = "composite_score" if "composite_score" in fwd.columns else "alpha_score"
        frames.append(fwd[["trade_date", "symbol", score_col]].rename(
            columns={score_col: "composite_score"}))
    preds = pd.concat(frames, ignore_index=True)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    return preds.drop_duplicates(["trade_date", "symbol"], keep="last") \
                .rename(columns={"composite_score": "alpha_score"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup-start", default="2025-10-01")
    ap.add_argument("--as-of", default=None, help="signal date (default: latest scored date)")
    args = ap.parse_args()

    preds = _load_scores()
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close",
                  "is_st", "is_suspended", "is_limit_up"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    warmup = pd.Timestamp(args.warmup_start)
    panel = panel[panel["trade_date"] >= warmup - pd.Timedelta(days=130)]

    as_of = pd.Timestamp(args.as_of) if args.as_of else preds["trade_date"].max()
    preds = preds[(preds["trade_date"] >= warmup) & (preds["trade_date"] <= as_of)]
    if preds.empty:
        raise SystemExit("no scores in window")
    flags = panel[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel["trade_date"].unique())

    # PIT contexts for the 反T gates (uses data ≤ as_of only)
    ctx = build_day_contexts(panel[panel["trade_date"] <= as_of])
    ctx_today = ctx[ctx["trade_date"] == as_of].set_index("symbol")
    close_today = panel[panel["trade_date"] == as_of].set_index("symbol")["close"]

    report = {"as_of": str(as_of.date()), "books": {}}
    for name, cfg in BOOKS.items():
        out_dir = ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tw = build_hold_band_weights(preds, config=cfg, trade_dates=trade_dates)
        if tw.empty:
            print(f"[{name}] empty book"); continue
        target_date = tw.index.max()           # = first trading day after as_of
        w = tw.loc[target_date]
        held = w[w > 0].rename("weight").reset_index()
        held.columns = ["symbol", "weight"]

        prev_path = out_dir / "targets_latest.csv"
        orders = []
        if prev_path.exists():
            prev = pd.read_csv(prev_path)
            merged = held.merge(prev[["symbol", "weight"]], on="symbol", how="outer",
                                suffixes=("_new", "_old")).fillna(0.0)
            for _, r in merged.iterrows():
                delta = r["weight_new"] - r["weight_old"]
                if abs(delta) > 1e-6:
                    orders.append({"symbol": r["symbol"],
                                   "side": "buy" if delta > 0 else "sell",
                                   "delta_weight": round(float(delta), 5)})
        held.to_csv(prev_path, index=False)
        held.assign(target_date=str(pd.Timestamp(target_date).date())) \
            .to_csv(out_dir / f"targets_{pd.Timestamp(target_date).date()}.csv", index=False)
        pd.DataFrame(orders).to_csv(out_dir / f"orders_{pd.Timestamp(target_date).date()}.csv",
                                    index=False)

        # 反T watchlist: held names passing the selective gates today
        watch = []
        for sym in held["symbol"]:
            if sym not in ctx_today.index:
                continue
            c = ctx_today.loc[sym]
            ctx_obj = DayContext(atr_pct=float(c["atr_pct"]) if np.isfinite(c["atr_pct"]) else 0.0,
                                 mom_5d=float(c["mom_5d"]) if np.isfinite(c["mom_5d"]) else 0.0,
                                 gap_open=0.0, regime=str(c["regime"]))
            mode, reason = check_gates(ctx_obj, DOT_PARAMS)
            if mode is None:
                continue
            px = float(close_today.get(sym, np.nan))
            atr = ctx_obj.atr_pct
            watch.append({
                "symbol": sym, "mode": mode,
                "atr_pct": round(atr, 4), "mom_5d": round(ctx_obj.mom_5d, 4),
                "regime": ctx_obj.regime,
                "approx_sell_trigger": round(px * (1 + DOT_PARAMS.dip_atr_mult * atr), 3),
                "approx_buyback_target": round(px * (1 + DOT_PARAMS.dip_atr_mult * atr)
                                               * (1 - DOT_PARAMS.target_atr_mult * atr), 3),
                "note": "levels re-anchor to LIVE running VWAP intraday; these are close-based approximations",
            })
        pd.DataFrame(watch).to_csv(out_dir / f"dot_watchlist_{pd.Timestamp(target_date).date()}.csv",
                                   index=False)
        report["books"][name] = {
            "target_date": str(pd.Timestamp(target_date).date()),
            "n_held": int(len(held)), "n_orders": len(orders),
            "dot_watchlist": len(watch),
        }
        print(f"[{name}] target {pd.Timestamp(target_date).date()}: {len(held)} names, "
              f"{len(orders)} orders, 反T watch {len(watch)}", flush=True)

    (ROOT / "last_update.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
