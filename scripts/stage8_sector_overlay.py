#!/usr/bin/env python3
"""Stage 8 step 5 — risk overlay on the laggard sector-reversal book.

Takes the validated laggard-reversal sector book and sweeps risk overlays
(market-trend gate, vol-target, both) to cut the 23-34% drawdown toward the
v8.9 Calmar (1.58) while keeping the excess + bull capture. The fast
sector-basket engine does the sweep; the WINNER is then rebuilt as a per-stock
book and run through the strict A-share engine (T+1, costs, limit-up/ST) so the
headline after-cost CAGR/Calmar is honest and comparable to v8.9 (+17.3% /
10.9% DD / Calmar 1.58).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402
from quantagent.strategy.sector_risk_overlay import (  # noqa: E402
    apply_overlay_to_nav, combined_overlay,
)
from quantagent.strategy.sector_rotation_book import build_rotation_book  # noqa: E402

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
FEAT = "runtime/stage8_sector_rotation/stock_features.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
RET = "ret_eqw"
BULL = {"rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
        "rally_2025_2026": ("2025-01-01", "2026-05-18"),
        "covid_2020": ("2020-03-23", "2021-02-10")}


def _ann(d: pd.Series) -> float:
    n = len(d)
    return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def _stats(nav: pd.Series, bench_daily: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    peak = nav.cummax()
    dd = float(abs((nav / peak - 1).min()))
    cagr = _ann(r)
    b = bench_daily.reindex(r.index)
    bw = {}
    for name, (a, z) in BULL.items():
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
        if m.sum() < 10:
            continue
        bw[name] = round(float((1 + r[m]).prod() - (1 + b[m]).prod()), 4)
    return {"cagr": round(cagr, 4), "maxdd": round(dd, 4),
            "calmar": round(cagr / dd, 3) if dd > 1e-9 else None,
            "exc_eqsector": round(cagr - _ann(b.dropna()), 4),
            "bull_2024_2025": bw.get("rally_2024H2_2025"),
            "bull_2025_2026": bw.get("rally_2025_2026"),
            "bull_covid": bw.get("covid_2020")}


def laggard_basket_daily(sw: pd.DataFrame, wide_ret: pd.DataFrame, *, top_n: int,
                         rebalance: int, cost_bps: float = 18.0) -> pd.Series:
    dates = list(sw.index)
    rebal = set(dates[::rebalance])
    w = pd.DataFrame(0.0, index=dates, columns=sw.columns)
    cur = pd.Series(0.0, index=sw.columns)
    for d in dates:
        if d in rebal:
            s = sw.loc[d].dropna()
            if len(s) >= top_n:
                picks = s.sort_values(ascending=True).head(top_n).index
                cur = pd.Series(0.0, index=sw.columns)
                cur[picks] = 1.0 / top_n
        w.loc[d] = cur.values
    gross = (w.shift(1).fillna(0.0) * wide_ret.reindex(w.index)).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(1).fillna(0.0) * (cost_bps / 1e4)
    return (gross - cost).fillna(0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="mom_60")  # best OOS-Calmar reversal
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--rebalance", type=int, default=20)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-05-18")
    ap.add_argument("--n-within", type=int, default=8)
    args = ap.parse_args()

    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[(sp.trade_date >= args.start) & (sp.trade_date <= args.end)]
    wide_ret = sp.pivot_table(index="trade_date", columns="sector_level_1", values=RET)
    sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=args.signal).reindex(wide_ret.index)
    eqsector = wide_ret.mean(axis=1)

    base_daily = laggard_basket_daily(sw, wide_ret, top_n=args.top_n, rebalance=args.rebalance)
    base_nav = (1 + base_daily).cumprod()
    print(f"[base] laggard {args.signal} top{args.top_n} rb{args.rebalance} (no overlay):")
    print("   ", _stats(base_nav, eqsector), "\n")

    # ---- overlay sweep (fast, sector-basket level) ----
    variants = {
        "trend_60_120": dict(use_trend=True, lookback=60, ma_window=120),
        "trend_40_100": dict(use_trend=True, lookback=40, ma_window=100),
        "trend_120_200": dict(use_trend=True, lookback=120, ma_window=200),
        "voltarget_25": dict(use_trend=False, use_voltarget=True, target_ann=0.25),
        "voltarget_20": dict(use_trend=False, use_voltarget=True, target_ann=0.20),
        "trend60_vt25": dict(use_trend=True, lookback=60, ma_window=120, use_voltarget=True, target_ann=0.25),
        "trend60_vt20": dict(use_trend=True, lookback=60, ma_window=120, use_voltarget=True, target_ann=0.20),
    }
    print("=== overlay sweep (fast sector-basket) ===")
    rows = [{"overlay": "none", **_stats(base_nav, eqsector)}]
    overlays: dict[str, pd.Series] = {}
    for name, kw in variants.items():
        gross = combined_overlay(eqsector, base_daily, **kw)
        overlays[name] = gross
        nav = apply_overlay_to_nav(base_daily, gross)
        rows.append({"overlay": name, **_stats(nav, eqsector)})
    swp = pd.DataFrame(rows)
    swp.to_csv(OUT_DIR / "overlay_sweep.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(swp.to_string(index=False))

    # pick winner: best Calmar among overlays that keep excess>0 and don't kill 2025 bull
    cand = swp[(swp.overlay != "none") & (swp.exc_eqsector > 0)]
    cand = cand[cand.bull_2025_2026.fillna(-1) > -0.02]
    winner = cand.sort_values("calmar", ascending=False).iloc[0]["overlay"] if not cand.empty \
        else swp[swp.overlay != "none"].sort_values("calmar", ascending=False).iloc[0]["overlay"]
    print(f"\n[winner] overlay = {winner}")

    # ---- strict A-share confirm: per-stock laggard book + winning overlay ----
    print("[strict] rebuilding per-stock laggard book + overlay, running A-share engine ...")
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= args.start) & (panel.trade_date <= args.end)]
    smap = pd.read_parquet(SECTOR)
    feat = pd.read_parquet(FEAT)
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    feat = feat[(feat.trade_date >= args.start) & (feat.trade_date <= args.end)]
    eval_dates = sorted(win["trade_date"].unique())
    gross = overlays[winner]

    out = {}
    for label, ov in [("no_overlay", None), (f"overlay_{winner}", gross)]:
        tw = build_rotation_book(
            sp, feat, signal=args.signal, top_n=args.top_n, rebalance_days=args.rebalance,
            sector_weighting="equal", within_sector="top_liquid", n_within=args.n_within,
            reverse=True, exposure=("overlay" if ov is not None else "full"),
            gross_overlay=ov, eval_dates=eval_dates,
        )
        arts = run_strict_backtest_v8(tw, win, sector_map=smap)
        m = arts.metrics
        st = _stats(arts.nav, eqsector)
        out[label] = {"strict_cagr": round(m.annualized_return, 4),
                      "strict_maxdd": round(m.max_drawdown, 4),
                      "strict_calmar": round(m.calmar, 4) if m.calmar else None,
                      "strict_sharpe": round(m.sharpe, 4) if m.sharpe else None,
                      "strict_turnover": round(m.turnover, 4),
                      "exc_eqsector": st["exc_eqsector"],
                      "bull_2024_2025": st["bull_2024_2025"],
                      "bull_2025_2026": st["bull_2025_2026"]}
        print(f"  STRICT {label}: CAGR={out[label]['strict_cagr']:+.1%} "
              f"DD={out[label]['strict_maxdd']:.1%} Calmar={out[label]['strict_calmar']} "
              f"excEqSec={out[label]['exc_eqsector']:+.1%} "
              f"bull25={out[label]['bull_2025_2026']} turn={out[label]['strict_turnover']}")

    (OUT_DIR / "overlay_strict_confirm.json").write_text(
        json.dumps({"winner": winner, "fast_sweep": rows, "strict": out},
                   ensure_ascii=False, indent=2, default=str))
    print(f"\n[write] {OUT_DIR/'overlay_sweep.csv'} , {OUT_DIR/'overlay_strict_confirm.json'}")
    print("\n[ref] v8.9 baseline = +17.3% CAGR / 10.9% MaxDD / Calmar 1.58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
