#!/usr/bin/env python3
"""Stage 12 Task 4 — risk-control backtest suite ("beat it to death").

Runs the full robustness battery on a score-driven strategy (default: v8.9
composite) through the strict A-share engine, per the backtest-expert
methodology (punish the strategy; seek plateaus not peaks; regime + capacity
realism):

  cost stress     slippage 8/15/30/50/100 bps  -> does the edge survive friction?
  topK plateau    20/30/50/100                 -> stable range vs a lucky peak?
  phase stability  5 rebalance offsets          -> not a timing artifact
  regime split     bull/bear/sideways CAGR+excess (all-A 60d trend)
  capacity curve   AUM 1e6..1e9, 10% ADV cap    -> where market impact erodes it
  beta decomp      beta/Jensen alpha vs all-A + CSI300/500/1000 (Stage 11)

Emits a per-test table + a Deploy/Refine/Abandon-style production verdict.
Strict T+1, ST/停牌/涨跌停 all enforced by the engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest import beta_decomposition as bd  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

SCORE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
OUT = Path("runtime/stage12")
PERIOD = 20


def build_book(stock_day, *, rebal_dates, eval_dates, size, col="composite_score"):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=[col]).sort_values(col, ascending=False).head(size)
        if sd.empty:
            continue
        rows[d] = {s: 1.0 / len(sd) for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def index_daily(label, dates):
    idx = pd.read_parquet(INDEX); idx = idx[idx["label"] == label].copy()
    idx["observation_date"] = pd.to_datetime(idx["observation_date"])
    return idx.set_index("observation_date")["close"].sort_index().pct_change().reindex(pd.DatetimeIndex(sorted(dates))).dropna()


def run(tw, win, smap, *, slippage=8.0, cash=1e6, part=0.10):
    cfg = AShareExecutionSimulationConfig(slippage_bps=slippage, initial_cash=cash, volume_participation_cap=part)
    arts = run_strict_backtest_v8(tw, win, sector_map=smap, config=cfg)
    return arts.nav, arts.metrics


def regime_label(all_a):
    cum = (1 + all_a).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")), index=all_a.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", "composite_score"]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = sc.trade_date.min(), sc.trade_date.max()
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    smap = pd.read_parquet(SECTOR)
    eval_dates = sorted(win.trade_date.unique())
    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = sc.merge(flags, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())
    rebal0 = dsorted[0::PERIOD]
    all_a = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None).mean(axis=1).dropna()
    benches = {"all_a": all_a, "csi300": index_daily("csi300", eval_dates),
               "csi500": index_daily("csi500", eval_dates), "csi1000": index_daily("csi1000", eval_dates)}
    print(f"[suite] v8.9 OOS {start.date()}..{end.date()} ({len(eval_dates)}d) | all-A ann={bd.ann_return(all_a):+.1%}")
    report = {}

    # base book (size 30)
    tw30 = build_book(stock_day, rebal_dates=rebal0, eval_dates=eval_dates, size=30)

    # ---- 1. cost stress ----
    print("\n=== 1. COST STRESS (size30, rb20) ===")
    cost_rows = []
    for slip in (8, 15, 30, 50, 100):
        nav, m = run(tw30, win, smap, slippage=slip)
        r = nav.pct_change().dropna()
        cost_rows.append({"slippage_bps": slip, "cagr": round(m.annualized_return, 4),
                          "maxdd": round(m.max_drawdown, 4), "calmar": round(m.calmar, 3),
                          "turnover": round(m.turnover, 4)})
        print(f"  {slip:>3}bps: CAGR={m.annualized_return:+.1%} DD={m.max_drawdown:.1%} Calmar={m.calmar:.2f} turn={m.turnover:.4f}")
    report["cost_stress"] = cost_rows
    surv = cost_rows[-1]["cagr"] / cost_rows[0]["cagr"] if cost_rows[0]["cagr"] else 0
    print(f"  -> survives 100bps: {cost_rows[-1]['cagr']:+.1%} ({surv:.0%} of 8bps CAGR)")

    # ---- 2. topK plateau ----
    print("\n=== 2. topK PLATEAU (8bps) ===")
    topk_rows = []
    for K in (20, 30, 50, 100):
        tw = build_book(stock_day, rebal_dates=rebal0, eval_dates=eval_dates, size=K)
        nav, m = run(tw, win, smap)
        topk_rows.append({"topK": K, "cagr": round(m.annualized_return, 4), "calmar": round(m.calmar, 3),
                          "turnover": round(m.turnover, 4)})
        print(f"  K={K:>3}: CAGR={m.annualized_return:+.1%} Calmar={m.calmar:.2f} turn={m.turnover:.4f}")
    report["topk"] = topk_rows

    # ---- 3. phase stability (size30) ----
    print("\n=== 3. PHASE STABILITY (size30, 8bps) ===")
    phase_cagr = []
    for ph in (0, 4, 8, 12, 16):
        tw = build_book(stock_day, rebal_dates=dsorted[ph::PERIOD], eval_dates=eval_dates, size=30)
        nav, m = run(tw, win, smap)
        phase_cagr.append(m.annualized_return)
    report["phase"] = {"mean": round(float(np.mean(phase_cagr)), 4), "std": round(float(np.std(phase_cagr)), 4),
                       "min": round(min(phase_cagr), 4), "max": round(max(phase_cagr), 4)}
    print(f"  CAGR across 5 phases: {np.mean(phase_cagr):+.1%} ± {np.std(phase_cagr):.1%} "
          f"[{min(phase_cagr):+.1%}, {max(phase_cagr):+.1%}]")

    # ---- 4. regime split + beta decomp (size30 base) ----
    nav30, m30 = run(tw30, win, smap)
    r30 = nav30.pct_change().dropna()
    reg = regime_label(all_a).reindex(r30.index)
    print("\n=== 4. REGIME SPLIT (size30, vs all-A) ===")
    reg_rows = {}
    for rg in ("bull", "sideways", "bear"):
        mask = reg == rg
        if mask.sum() < 10:
            continue
        s, b = r30[mask], all_a.reindex(r30.index)[mask]
        reg_rows[rg] = {"days": int(mask.sum()), "strat_ann": round(bd.ann_return(s), 4),
                        "bench_ann": round(bd.ann_return(b), 4), "excess": round(bd.ann_return(s) - bd.ann_return(b), 4)}
        print(f"  {rg:<9} ({int(mask.sum()):>3}d): strat={bd.ann_return(s):+.1%} allA={bd.ann_return(b):+.1%} excess={bd.ann_return(s)-bd.ann_return(b):+.1%}")
    report["regime"] = reg_rows
    panel30 = bd.full_panel(r30, nav30, benches, turnover=m30.turnover, primary="all_a")
    report["beta_decomp"] = panel30
    print("\n=== 5. BETA DECOMPOSITION (size30) ===")
    print(f"  vs all-A : beta={panel30['beta_all_a']} Jensen-alpha={panel30['alpha_all_a']:+.1%} | vs CSI300 alpha={panel30['alpha_csi300']:+.1%}")
    print(f"  up_capture={panel30['up_capture']} down_capture={panel30['down_capture']}")

    # ---- 6. capacity curve (size30) ----
    print("\n=== 6. CAPACITY CURVE (size30, 8bps, 10% ADV cap) ===")
    cap_rows = []
    base_cagr = None
    for cash in (1e6, 1e7, 5e7, 1e8, 5e8, 1e9):
        nav, m = run(tw30, win, smap, cash=cash)
        if base_cagr is None:
            base_cagr = m.annualized_return
        cap_rows.append({"aum_cny": cash, "cagr": round(m.annualized_return, 4),
                         "pct_of_base": round(m.annualized_return / base_cagr, 3) if base_cagr else None,
                         "n_fills": int(m.n_fills)})
        print(f"  AUM={cash/1e6:>6.0f}M: CAGR={m.annualized_return:+.1%} ({m.annualized_return/base_cagr:.0%} of base) fills={m.n_fills}")
    report["capacity"] = cap_rows
    cap_ok = [c for c in cap_rows if c["pct_of_base"] and c["pct_of_base"] >= 0.8]
    capacity = cap_ok[-1]["aum_cny"] if cap_ok else cap_rows[0]["aum_cny"]
    print(f"  -> capacity (>=80% of base CAGR): ~{capacity/1e6:.0f}M CNY")

    # ---- verdict ----
    a = panel30.get("alpha_all_a") or 0
    verdict = []
    verdict.append(("cost_survival", "PASS" if surv >= 0.6 else "WEAK"))
    verdict.append(("phase_stable", "PASS" if report["phase"]["std"] < abs(report["phase"]["mean"]) * 0.5 else "WEAK"))
    verdict.append(("positive_alpha", "PASS" if a > 0.03 else "WEAK"))
    verdict.append(("regime_breadth", "PASS" if sum(1 for v in reg_rows.values() if v["excess"] > 0) >= 2 else "WEAK"))
    verdict.append(("capacity_100M", "PASS" if capacity >= 1e8 else "LIMITED"))
    report["verdict"] = dict(verdict)
    print("\n" + "=" * 66 + "\n=== PRODUCTION VERDICT (v8.9 size30) ===")
    for k, v in verdict:
        print(f"  {k:<18}: {v}")
    n_pass = sum(1 for _, v in verdict if v == "PASS")
    overall = "DEPLOY" if n_pass >= 4 else "REFINE" if n_pass >= 3 else "CAUTION"
    report["overall"] = overall
    print(f"  {'OVERALL':<18}: {overall} ({n_pass}/5 pass)")
    print("  NOTE: OOS window is 2024-08..2026-05 (momentum bull); bear robustness under-sampled.")

    (OUT / "risk_control_v89.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[write] {OUT/'risk_control_v89.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
