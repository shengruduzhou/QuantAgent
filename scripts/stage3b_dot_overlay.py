#!/usr/bin/env python3
"""Stage 3B: true intraday 做T overlay on the daily base books (GATE before RL).

A-share T+1 reality: shares bought today CANNOT be sold today, so the only legal
INTRADAY 做T on existing inventory is HIGH-SELL-LOW-BUY (sell yesterday's shares
into an intraday spike, buy them back on an intraday dip the same day). Low-buy-
high-sell is inherently cross-day and is NOT an intraday overlay.

Causal triggers only (no future high/low/VWAP): sell when price is extended above
the running VWAP (price_vs_vwap_z >= sell_z) and intraday up; buy back when it
reverts (z <= buyback_z). If no reversion by close, forced buyback at close ->
FAILED high-sell. Every T-trade logs trigger/exec/buyback/cost/failure. Net 做T
PnL is added to the base book and compared to no-overlay, at 8/15/30/50/100 bps.

If net OOS contribution after cost is not positive on BOTH books, the gate FAILS
and we STOP before RL (per the spec).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/intraday_2026/intraday_panel_675.parquet"
S1 = "runtime/reports/v89_closed_loop/stage1"
OUT = Path("runtime/reports/v89_closed_loop/stage3b"); OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = {"non2026": ("2025-09-01", "2025-12-31"), "y2026": ("2026-01-02", "2026-05-13")}


def simulate_dot(panel: pd.DataFrame, positions: pd.DataFrame, *, sell_z=1.5, buyback_z=0.0,
                 sell_frac=0.5, cost_bps=8.0) -> pd.DataFrame:
    """Per held (symbol, day) high-sell-low-buy on sellable (prior-day) inventory."""
    # sellable inventory at date d for symbol s = its weight in the book as of d (held from <=d-1).
    pos = positions.sort_values(["symbol", "trade_date"]).copy()
    pos["prev_w"] = pos.groupby("symbol")["weight"].shift(1)
    sellable = pos.dropna(subset=["prev_w"])
    sellable = sellable[sellable["prev_w"] > 1e-9][["trade_date", "symbol", "prev_w"]]
    held = sellable.set_index(["trade_date", "symbol"])["prev_w"].to_dict()

    rt = cost_bps / 1e4  # one-way cost; round trip applied per leg
    rows = []
    pm = panel[["symbol", "trade_date", "trade_time", "close", "price_vs_vwap_z", "intraday_return"]].copy()
    pm = pm[pm.set_index(["trade_date", "symbol"]).index.isin(held.keys())]
    for (d, s), g in pm.groupby(["trade_date", "symbol"], sort=False):
        w = held.get((d, s))
        if w is None:
            continue
        g = g.sort_values("trade_time")
        z = g["price_vs_vwap_z"].to_numpy(); px = g["close"].to_numpy(); ir = g["intraday_return"].to_numpy()
        n = len(px)
        if n < 10:
            continue
        # find first sell trigger (extended above vwap + up), excluding last 10 min
        sell_i = None
        for i in range(5, n - 10):
            if z[i] >= sell_z and ir[i] > 0:
                sell_i = i; break
        if sell_i is None:
            continue
        sell_px = px[sell_i]
        # find buyback after sell: revert to/below buyback_z
        buy_i = None
        for j in range(sell_i + 1, n):
            if z[j] <= buyback_z:
                buy_i = j; break
        forced = buy_i is None
        buy_px = px[-1] if forced else px[buy_i]
        # 做T pnl on the sold fraction (per unit book weight): (sell-buyback)/sell - 2 legs cost
        gross = (sell_px - buy_px) / sell_px
        net = gross - 2 * rt
        pnl_contrib = sell_frac * w * net   # contribution to book daily return
        rows.append({"trade_date": d, "symbol": s, "sell_px": sell_px, "buy_px": buy_px,
                     "forced_buyback": forced, "gross": gross, "net": net,
                     "pnl_contrib": pnl_contrib, "failed_high_sell": net <= 0,
                     "weight": w})
    return pd.DataFrame(rows)


def book_daily_return(positions: pd.DataFrame, panel_daily: pd.DataFrame) -> pd.Series:
    """Base book daily return (close-to-close of held weights), for overlay addition."""
    p = positions.merge(panel_daily, on=["symbol", "trade_date"], how="left")
    p["contrib"] = p["weight"] * p["fwd1d"].fillna(0.0)
    return p.groupby("trade_date")["contrib"].sum()


def window_stats(dot: pd.DataFrame, start, end) -> dict:
    d = dot[(dot["trade_date"] >= pd.Timestamp(start)) & (dot["trade_date"] <= pd.Timestamp(end))]
    if d.empty:
        return {"n_T": 0}
    daily = d.groupby("trade_date")["pnl_contrib"].sum()
    days = len(daily)
    ann = (1 + daily).prod() ** (244 / max(1, days)) - 1
    return {"n_T": int(len(d)), "ann_overlay_contrib": round(float(ann), 4),
            "total_overlay_contrib": round(float(daily.sum()), 4),
            "win_rate": round(float((d["net"] > 0).mean()), 3),
            "avg_win": round(float(d.loc[d["net"] > 0, "net"].mean() if (d["net"] > 0).any() else 0), 5),
            "avg_loss": round(float(d.loc[d["net"] <= 0, "net"].mean() if (d["net"] <= 0).any() else 0), 5),
            "failed_high_sell_rate": round(float(d["failed_high_sell"].mean()), 3),
            "forced_buyback_rate": round(float(d["forced_buyback"].mean()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sell-z", type=float, default=1.5)
    ap.add_argument("--buyback-z", type=float, default=0.0)
    ap.add_argument("--sell-frac", type=float, default=0.5)
    args = ap.parse_args()
    if not Path(PANEL).exists():
        print(f"FATAL: {PANEL} not built yet (run Stage 2.5)."); return 1
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "trade_time", "close", "price_vs_vwap_z", "intraday_return"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    report = {}
    for book in ("w210_k10", "w111_k5"):
        pos = pd.read_parquet(f"{S1}/daily_{book}_positions.parquet")
        pos["trade_date"] = pd.to_datetime(pos["trade_date"])
        book_res = {}
        for cost in (8, 15, 30, 50, 100):
            dot = simulate_dot(panel, pos, sell_z=args.sell_z, buyback_z=args.buyback_z,
                               sell_frac=args.sell_frac, cost_bps=float(cost))
            if cost == 8:
                dot.to_parquet(OUT / f"dot_trades_{book}.parquet", index=False)
                # failure / success case dumps
                dd = dot.sort_values("net")
                dd.head(20).to_csv(OUT / f"failed_high_sell_top20_{book}.csv", index=False)
                dd.tail(20).to_csv(OUT / f"success_dot_top20_{book}.csv", index=False)
            book_res[f"{cost}bps"] = {w: window_stats(dot, *WINDOWS[w]) for w in WINDOWS}
        report[book] = book_res
        for cost in (8, 30):
            for w in WINDOWS:
                st = book_res[f"{cost}bps"][w]
                print(f"{book} {cost}bps {w}: nT={st.get('n_T')} ann_overlay {st.get('ann_overlay_contrib','-')} "
                      f"winrate {st.get('win_rate','-')} failrate {st.get('failed_high_sell_rate','-')}", flush=True)
    (OUT / "stage3b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # GATE verdict
    verdict = {}
    for book in report:
        passes = all(report[book]["30bps"][w].get("ann_overlay_contrib", -1) > 0 for w in WINDOWS)
        verdict[book] = "PASS (net positive both windows @30bps)" if passes else "FAIL (no net edge after cost)"
    (OUT / "GATE_VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\n=== STAGE 3B GATE ===")
    for b, v in verdict.items():
        print(f"  {b}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
