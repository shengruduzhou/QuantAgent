#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 7: is the LONG-SHORT factor spread robustly positive across regimes?

The factor edge lives in the long-short spread (long cheap/winners, short
expensive/losers). This is the market-neutral factor return — its CAGR per
regime window IS the alpha, independent of market beta. We test whether the
spread is CONSISTENTLY positive across bears and bulls (skill) or only in some
windows (fragile). After-cost (turnover charged); monthly rebalance for the
slow fundamental factors.

Executability caveat: single-stock shorting is restricted in A-share, so a real
market-neutral product hedges the short leg with index futures (captures spread
minus long-basket-vs-index deviation). This test is the standard factor-spread
diagnostic — a precondition for, not a guarantee of, an executable product.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.portfolio.policy_search import PolicyConfig, backtest_policy, prepare_working_frame

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
FUND = "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
LGBM = "runtime/stage6_classical_2018/wf/walkforward_predictions.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rebalance-days", type=int, default=20)
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--neutralize", default="industry")  # none | industry
    ap.add_argument("--output-dir", default="runtime/stage7_longshort_spread")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "amount",
                                            "is_st", "is_suspended", "is_limit_up"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[panel["trade_date"] >= args.start].reset_index(drop=True)
    sector = pd.read_parquet(SECTOR)

    fund = pd.read_parquet(FUND)
    fcols = ["eps_basic", "bps"]
    f = fund[["symbol", "available_at", *fcols]].copy()
    f["available_at"] = pd.to_datetime(f["available_at"], errors="coerce")
    f = f.dropna(subset=["available_at"]).sort_values("available_at")
    m = pd.merge_asof(panel.sort_values("trade_date"), f, left_on="trade_date",
                      right_on="available_at", by="symbol", direction="backward")
    close = pd.to_numeric(m["close"], errors="coerce")
    ey = pd.to_numeric(m["eps_basic"], errors="coerce") / close
    bp = pd.to_numeric(m["bps"], errors="coerce") / close
    base = m[["symbol", "trade_date"]].copy()

    factors: dict[str, pd.Series] = {
        "value_book_to_price": bp,
        "value_earnings_yield": ey,
        "value_composite": (bp.groupby(m["trade_date"]).rank(pct=True)
                            + ey.groupby(m["trade_date"]).rank(pct=True)),
    }
    # LightGBM momentum spread for comparison (if available).
    try:
        lg = pd.read_parquet(LGBM, columns=["symbol", "trade_date", "alpha_5d"])
        lg["trade_date"] = pd.to_datetime(lg["trade_date"])
        mm = base.merge(lg, on=["symbol", "trade_date"], how="left")
        factors["momentum_lgbm"] = pd.to_numeric(mm["alpha_5d"], errors="coerce")
    except Exception:
        pass

    dates = sorted(panel["trade_date"].dropna().unique())
    windows = [(dates[i], dates[min(i + args.window_days, len(dates)) - 1])
               for i in range(0, len(dates), args.window_days)]
    windows = [(s, e) for (s, e) in windows
               if pd.Index(dates).get_indexer([e])[0] - pd.Index(dates).get_indexer([s])[0] >= 40]

    rows = []
    for name, fac in factors.items():
        preds = base.copy()
        preds["alpha_5d"] = np.asarray(fac, dtype=float)
        preds["alpha_1d"] = preds["alpha_5d"]; preds["alpha_20d"] = preds["alpha_5d"]
        work = prepare_working_frame(preds, panel, sector)
        cfg = PolicyConfig(horizon=5, top_k=args.top_k, rebalance_days=args.rebalance_days,
                           side="long_short", transform="csrank", neutralize=args.neutralize,
                           liquidity_filter="ex_bottom_30pct")
        spreads, per_win = [], []
        for (ws, we) in windows:
            wk = work[(work["trade_date"] >= ws) & (work["trade_date"] <= we)]
            if wk["alpha_5d"].notna().sum() < 200:
                continue
            r = backtest_policy(wk, cfg).metrics
            c = r["cagr"]
            if np.isfinite(c):
                spreads.append(c)
                per_win.append({"window": f"{ws.date()}..{we.date()}", "spread_cagr": round(c, 4),
                                "maxDD": round(r["max_drawdown"], 4), "sharpe": round(r.get("sharpe", float('nan')), 3)})
        sp = np.array(spreads, dtype=float); n = len(sp)
        if n < 4:
            continue
        mean_s, med_s = float(np.mean(sp)), float(np.median(sp))
        std_s = float(np.std(sp, ddof=1)); ir = mean_s / std_s if std_s > 1e-9 else float("nan")
        rows.append({"factor": name, "n_windows": n,
                     "pct_pos_spread": round(float((sp > 0).mean()) * 100, 1),
                     "mean_spread": round(mean_s, 4), "median_spread": round(med_s, 4),
                     "spread_IR": round(ir, 3), "worst": round(float(np.min(sp)), 4),
                     "best": round(float(np.max(sp)), 4), "per_window": per_win})
        print(f"  {name:22} n={n} %pos {rows[-1]['pct_pos_spread']:>5}  median_spread {med_s:+.2%}  IR {ir:+.2f}  worst {np.min(sp):+.2%}", flush=True)

    lb = pd.DataFrame([{k: v for k, v in r.items() if k != "per_window"} for r in rows]).sort_values(
        ["pct_pos_spread", "spread_IR"], ascending=False) if rows else pd.DataFrame()
    if not lb.empty:
        lb.to_csv(out / "longshort_spread_stability.csv", index=False)
        print("\n=== LONG-SHORT SPREAD STABILITY (industry-neutral, after-cost) ===")
        print(lb.to_string(index=False))
        robust = lb[(lb["pct_pos_spread"] >= 70) & (lb["spread_IR"] >= 0.7) & (lb["median_spread"] > 0)]
        verdict = (f"ROBUST spread alpha: {robust['factor'].tolist()}" if not robust.empty
                   else "NO factor spread is robustly positive across regimes (≥70% windows + IR≥0.7 + median>0)")
    else:
        verdict = "no factors evaluated"
    (out / "summary.json").write_text(json.dumps({"verdict": verdict, "neutralize": args.neutralize,
        "config": {"top_k": args.top_k, "rebalance_days": args.rebalance_days, "window_days": args.window_days,
                   "start": args.start}, "factors": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
