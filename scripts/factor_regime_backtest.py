#!/usr/bin/env python3
"""严格风控回测 — (A) 因子核心跨 regime；(B) 并集(因子∪产业链) vs 因子 的叠加对照.

设计 (诚实约束): 因子预测是确定性的、覆盖数年 → 可做真正的跨 regime(牛/震荡/熊) 回测，
这是"跨 regime 最佳"的系统化核心。LLM 产业链池只在少数月末可得(LLM 慢/非确定/有 hindsight
风险) → 并集只能在近端少数月做"月度再平衡"叠加对照，作为前瞻 overlay (短窗口, 诚实标注)。

A. 因子核心: 日频 top-N 等权 → 严格模拟(base 8bps + stress 16bps) → 按基准 60 日趋势
   切 牛/震荡/熊, 报告每 regime 年化超额 (vs 等权全A)。两段: 牛市段(2024-08..2026-05)
   与 熊/震荡段(2022-02..2023-12, v8_bear_test 预测)。
B. 叠加: 月末再平衡, 因子-only(top n_factor) vs 并集(top n_factor ∪ 产业链 top n_chain),
   日填充持有 → 严格模拟 → 全程 + 分 regime 超额/回撤/sharpe 对照。

输出 runtime/reports/v8/factor_regime_backtest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
PRED_BULL = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
PRED_BEAR = "runtime/reports/v8/deep/v8_bear_test_20260602_230424/short_5d/predictions.parquet"
MON = Path("runtime/reports/monthly")
ANN = 244


def _code6(s):
    return str(s).split(".")[0].zfill(6)


def _load_panel(start):
    p = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "available_at"])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    return p[p["trade_date"] >= pd.Timestamp(start) - pd.Timedelta(days=10)].reset_index(drop=True)


def _bench_daily(panel, dates):
    px = panel[panel.trade_date.isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    # fill_method=None: halted stocks → NaN → excluded from the cross-sectional mean
    # (don't book a halted name as a 0% day, which would dilute the benchmark).
    return px.pct_change(fill_method=None).mean(axis=1).dropna()


def _regime_label(bench_daily):
    cum = (1 + bench_daily).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=bench_daily.index)


def _regime_excess(nav, bench_daily):
    strat = nav.pct_change().dropna()
    idx = strat.index.intersection(bench_daily.index)
    strat, bench = strat.reindex(idx), bench_daily.reindex(idx)
    regime = _regime_label(bench).reindex(idx)
    rows = {}
    for rg in ["all", "bull", "sideways", "bear"]:
        mask = pd.Series(True, index=idx) if rg == "all" else (regime == rg)
        n = int(mask.sum())
        if n < 3:
            continue
        s, b = strat[mask], bench[mask]
        ann_s = float((1 + s).prod() ** (ANN / n) - 1)
        ann_b = float((1 + b).prod() ** (ANN / n) - 1)
        rows[rg] = {"days": n, "strat_ann": round(ann_s, 4), "bench_ann": round(ann_b, 4),
                    "excess_ann": round(ann_s - ann_b, 4)}
    return rows


def _daily_topN(preds, n):
    out = {}
    for d, g in preds.groupby("trade_date"):
        top = g.nlargest(n, "alpha_score")
        if len(top):
            out[d] = pd.Series(1.0 / len(top), index=top["symbol"].values)
    tw = pd.DataFrame(out).T.fillna(0.0)
    tw.index.name = "trade_date"
    return tw.sort_index()


def _sim(tw, panel, sector, slip):
    return run_strict_backtest_v8(tw.fillna(0.0), panel, sector_map=sector,
                                  config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=slip))


def _core_run(label, pred_path, n, panel_all, sector, report):
    preds = pd.read_parquet(pred_path); preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    tw = _daily_topN(preds, n)
    panel = panel_all[panel_all.trade_date >= tw.index.min() - pd.Timedelta(days=10)]
    bench = _bench_daily(panel, tw.index)
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)
    print(f"\n=== A. 因子核心 [{label}] {tw.index.min().date()}..{tw.index.max().date()} | top-{n} 日频 ===")
    rec = {"window": f"{tw.index.min().date()}..{tw.index.max().date()}", "n": n, "bench_ann": round(bench_ann, 4)}
    for slip in (8.0, 16.0):
        res = _sim(tw, panel, sector, slip)
        m = res.metrics
        tag = "base" if slip == 8 else "stress"
        rec[tag] = {"ann": round(m.annualized_return, 4), "excess": round(m.annualized_return - bench_ann, 4),
                    "maxDD": round(m.max_drawdown, 4), "sharpe": round(m.sharpe, 3), "slippage_bps": slip}
        print(f"  [{tag} {int(slip)}bps] ann {m.annualized_return*100:+.2f}% | 等权全A {bench_ann*100:+.2f}% | "
              f"超额 {(m.annualized_return-bench_ann)*100:+.2f}% | maxDD {m.max_drawdown*100:.1f}% | sharpe {m.sharpe:.2f}")
        if slip == 8.0:
            rec["regime"] = _regime_excess(res.nav, bench)
            for rg, v in rec["regime"].items():
                print(f"      regime[{rg:<8}] {v['days']:>4}d | 策略年化 {v['strat_ann']*100:+.1f}% | "
                      f"基准 {v['bench_ann']*100:+.1f}% | 超额 {v['excess_ann']*100:+.1f}%")
    report[f"factor_core_{label}"] = rec
    return rec


def _union_overlay(dates, n_factor, n_chain, panel_all, sector, report):
    """Part B: 月末再平衡, 因子-only vs 并集, 日填充持有."""
    preds = pd.read_parquet(PRED_BULL); preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    reb = [pd.Timestamp(d) for d in dates if not preds[preds.trade_date == pd.Timestamp(d)].empty]
    if not reb:
        print("\n=== B. 并集叠加: 无可用月末预测, 跳过 ==="); return
    start, end = min(reb), preds["trade_date"].max()
    tds = sorted(t for t in preds["trade_date"].unique() if start <= t <= end)
    pmap = {rd: preds[preds.trade_date == rd] for rd in reb}

    def factor_set(rd):
        top = pmap[rd].nlargest(n_factor, "alpha_score")["symbol"].tolist()
        return top

    def chain_set(rd):
        cp = MON / f"chain_pool_{rd.date()}.parquet"
        if not cp.exists():
            return []
        c = pd.read_parquet(cp)
        ch = c[c.get("source", "") == "LLM产业链"] if "source" in c else c
        return ch.symbol.tolist()[:n_chain]

    def build(setfn):
        rows = {}
        sets = {rd: setfn(rd) for rd in reb}
        for t in tds:
            active = [rd for rd in sorted(reb) if rd <= t]
            syms = sets[active[-1]]
            if syms:
                rows[t] = pd.Series(1.0 / len(syms), index=syms)
        tw = pd.DataFrame(rows).T.fillna(0.0); tw.index.name = "trade_date"
        return tw.sort_index()

    tw_fac = build(factor_set)
    tw_uni = build(lambda rd: list(dict.fromkeys(factor_set(rd) + chain_set(rd))))
    panel = panel_all[panel_all.trade_date >= start - pd.Timedelta(days=10)]
    bench = _bench_daily(panel, tw_fac.index)
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)
    print(f"\n=== B. 并集叠加 vs 因子 [{start.date()}..{end.date()}] 月末再平衡 | "
          f"因子{n_factor} ∪ 链{n_chain} | 等权全A {bench_ann*100:+.2f}% ===")
    rec = {"window": f"{start.date()}..{end.date()}", "n_factor": n_factor, "n_chain": n_chain,
           "reb_dates": [str(d.date()) for d in reb], "bench_ann": round(bench_ann, 4)}
    for name, tw in [("factor_monthly", tw_fac), ("union_monthly", tw_uni)]:
        sub = {}
        for slip in (8.0, 16.0):
            res = _sim(tw, panel, sector, slip)
            m = res.metrics; tag = "base" if slip == 8 else "stress"
            sub[tag] = {"ann": round(m.annualized_return, 4), "excess": round(m.annualized_return - bench_ann, 4),
                        "maxDD": round(m.max_drawdown, 4), "sharpe": round(m.sharpe, 3)}
            if slip == 8.0:
                sub["regime"] = _regime_excess(res.nav, bench)
            print(f"  {name:<16}[{tag:>6}] ann {m.annualized_return*100:+.2f}% | 超额 "
                  f"{(m.annualized_return-bench_ann)*100:+.2f}% | maxDD {m.max_drawdown*100:.1f}% | sharpe {m.sharpe:.2f}")
        rec[name] = sub
    # delta
    fb, ub = rec["factor_monthly"]["base"], rec["union_monthly"]["base"]
    print(f"  并集−因子: 超额 {(ub['excess']-fb['excess'])*100:+.2f}% | maxDD {(ub['maxDD']-fb['maxDD'])*100:+.2f}pp | "
          f"sharpe {ub['sharpe']-fb['sharpe']:+.2f}")
    report["union_overlay"] = rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="因子核心 top-N")
    ap.add_argument("--n-factor", type=int, default=12, help="并集中因子名额")
    ap.add_argument("--n-chain", type=int, default=8, help="并集中产业链名额")
    ap.add_argument("--dates", nargs="+", default=["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    ap.add_argument("--skip-bear", action="store_true")
    ap.add_argument("--skip-union", action="store_true")
    args = ap.parse_args()

    sector = pd.read_parquet(SECTOR)
    panel_all = _load_panel("2022-01-01")
    report = {}
    _core_run("bull_2024_2026", PRED_BULL, args.n, panel_all, sector, report)
    if not args.skip_bear:
        _core_run("bear_2022_2023", PRED_BEAR, args.n, panel_all, sector, report)
    if not args.skip_union:
        _union_overlay(args.dates, args.n_factor, args.n_chain, panel_all, sector, report)

    Path("runtime/reports/v8/factor_regime_backtest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote runtime/reports/v8/factor_regime_backtest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
