#!/usr/bin/env python3
"""Stage 1: rebuild + audit the daily base books (w210_k10, w111_k5).

Produces real artifacts (no fabrication) from the strict tradable backtest:
positions, orders (blotter), trade log (FIFO round trips), cost decomposition,
and a CAPACITY STRESS at 8/15/30/50/100 bps slippage. If returns collapse at
higher cost, the report says so plainly (turnover/liquidity-driven warning).

Strict windows: non-2026 OOS 2025-09..2025-12 ; 2026 quasi-live 2026-01..2026-05.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp
from regime_strategy_search import regime_target_weights
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

ENS = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
OUT = Path("runtime/reports/v89_closed_loop/stage1"); OUT.mkdir(parents=True, exist_ok=True)
STRATS = {"w210_k10": {"w": (2, 1, 0), "k": 10}, "w111_k5": {"w": (1, 1, 1), "k": 5}}
WINDOWS = {"non2026": ("2025-09-01", "2025-12-31"), "y2026": ("2026-01-02", "2026-05-13")}
SLIPPAGE = [8, 15, 30, 50, 100]


def load():
    ens = pd.read_parquet(ENS); ens["trade_date"] = pd.to_datetime(ens["trade_date"])
    pc = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
          "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(bp.PANEL, columns=pc); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp("2024-06-01")]
    sector = pd.read_parquet(bp.SECTOR)
    tds = sorted(panel["trade_date"].unique())
    bench = bp._bench_daily(panel, tds); regime = bp._regime_label(bench)
    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]]
    ens = ens.merge(flags, on=["symbol", "trade_date"], how="left")
    R = ens[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]].copy()
    for c in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score"):
        R[c] = ens.groupby("trade_date")[c].rank(pct=True)
    return panel, sector, tds, bench, regime, R


def comp_frame(R, w):
    ws, wm, wl = w
    f = R[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]].copy()
    f["composite_score"] = (ws * R["short_5d_score"] + wm * R["mid_5d_30d_score"] + wl * R["long_30d_120d_score"]).to_numpy()
    return f


def run(tw, panel, sector, start, end, slip):
    twin = tw[(tw.index >= pd.Timestamp(start)) & (tw.index <= pd.Timestamp(end))]
    if twin.empty:
        return None
    sim = panel[(panel["trade_date"] >= pd.Timestamp(start) - pd.Timedelta(days=10))
                & (panel["trade_date"] <= pd.Timestamp(end) + pd.Timedelta(days=10))]
    return run_strict_backtest_v8(twin, sim, sector_map=sector,
                                  config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=float(slip)))


def main() -> int:
    panel, sector, tds, bench, regime, R = load()
    print("loaded; regime:", regime.value_counts().to_dict(), flush=True)
    summary = {}
    cap_rows = []
    for name, cfg in STRATS.items():
        comp = comp_frame(R, cfg["w"])
        kb = {"bull": cfg["k"], "sideways": cfg["k"], "bear": cfg["k"]}
        gb = {"bull": 1.0, "sideways": 1.0, "bear": 1.0}
        tw = regime_target_weights(comp, regime, kb, gb, tds)
        # save target positions (intended daily book) — long format
        twp = tw.copy(); twp.index.name = "trade_date"
        pos = twp.reset_index().melt(id_vars="trade_date", var_name="symbol", value_name="weight")
        pos = pos[pos["weight"] > 1e-9].sort_values(["trade_date", "weight"], ascending=[True, False])
        pos.to_parquet(OUT / f"daily_{name}_positions.parquet", index=False)
        # 2026 full run @8bps for orders/trades/cost
        res = run(tw, panel, sector, "2026-01-02", "2026-05-13", 8)
        res.trades.to_parquet(OUT / f"daily_{name}_orders.parquet", index=False)
        res.realized_trades.to_parquet(OUT / f"daily_{name}_trade_log.parquet", index=False)
        # blocked orders breakdown
        fo = res.failed_orders
        blk = {"rejected_orders": int(len(fo))}
        if not fo.empty:
            txt = (fo.get("last_message", pd.Series([""] * len(fo))).fillna("").astype(str)
                   + " " + fo.get("reason", pd.Series([""] * len(fo))).fillna("").astype(str)).str.lower()
            side = fo.get("side", pd.Series([""] * len(fo))).fillna("").astype(str).str.lower()
            blk["limit_up_blocked_buys"] = int(((txt.str.contains("limit_up|limit up|涨停")) & (side == "buy")).sum())
            blk["limit_down_blocked_sells"] = int(((txt.str.contains("limit_down|limit down|跌停")) & (side == "sell")).sum())
            blk["suspension_blocked"] = int(txt.str.contains("suspend|停牌").sum())
        m = res.metrics
        avg_names = float((tw[(tw.index >= '2026-01-02')] > 0).sum(axis=1).mean())
        max_w = float(tw[(tw.index >= '2026-01-02')].max().max())
        # cost decomposition (2026)
        notional_traded = float(res.trades.get("filled_quantity", pd.Series(dtype=float)).abs().mul(
            res.trades.get("avg_price", pd.Series(dtype=float)).fillna(0)).sum()) if not res.trades.empty else 0.0
        cost_decomp = {"strategy": name, "window": "2026", "total_cost_rmb": round(m.total_cost, 2),
                       "n_fills": int(m.n_fills), "n_round_trips": int(m.n_trades),
                       "turnover_daily": round(float(tw[(tw.index >= '2026-01-02')].diff().abs().sum(axis=1).mean() / 2), 4),
                       "notional_traded_rmb": round(notional_traded, 2),
                       "total_cost_pct_of_traded": round(m.total_cost / notional_traded, 5) if notional_traded > 0 else None}
        pd.DataFrame([cost_decomp]).to_csv(OUT / f"daily_{name}_cost_decomp.csv", index=False)
        summary[name] = {"weights": cfg["w"], "top_k": cfg["k"],
                         "y2026": {"total": round(m.total_return, 4), "cagr": round(m.annualized_return, 4),
                                   "maxDD": round(m.max_drawdown, 4), "sharpe": round(m.sharpe, 3),
                                   "calmar": round(m.calmar, 3), "turnover_daily": cost_decomp["turnover_daily"],
                                   "avg_names": round(avg_names, 1), "max_single_weight": round(max_w, 4),
                                   **blk}}
        # non2026
        rh = run(tw, panel, sector, *WINDOWS["non2026"], 8)
        summary[name]["non2026"] = {"cagr": round(rh.metrics.annualized_return, 4),
                                    "maxDD": round(rh.metrics.max_drawdown, 4),
                                    "total": round(rh.metrics.total_return, 4)}
        # CAPACITY STRESS
        for slip in SLIPPAGE:
            r26 = run(tw, panel, sector, *WINDOWS["y2026"], slip)
            cap_rows.append({"strategy": name, "slippage_bps": slip,
                             "y2026_cagr": round(r26.metrics.annualized_return, 4),
                             "y2026_total": round(r26.metrics.total_return, 4),
                             "y2026_maxDD": round(r26.metrics.max_drawdown, 4),
                             "y2026_sharpe": round(r26.metrics.sharpe, 3)})
            print(f"  {name} slip={slip}bps -> 2026 CAGR {r26.metrics.annualized_return:+.1%}", flush=True)
        print(f"[{name}] 2026 {summary[name]['y2026']['cagr']:+.1%} | non2026 {summary[name]['non2026']['cagr']:+.1%}", flush=True)

    cap = pd.DataFrame(cap_rows)
    cap.to_csv(OUT / "daily_w210_k10_capacity_stress.csv", index=False)
    cap.to_csv(OUT / "capacity_stress_all.csv", index=False)
    (OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # comparison report
    lines = ["# Stage 1 — Daily base book confirmation", ""]
    for name in STRATS:
        s = summary[name]
        lines += [f"## {name}  (sleeve {s['weights']}, top-k {s['top_k']})", "",
                  f"- 2026: total {s['y2026']['total']:+.1%}, CAGR {s['y2026']['cagr']:+.1%}, maxDD {s['y2026']['maxDD']:.1%}, "
                  f"Sharpe {s['y2026']['sharpe']}, Calmar {s['y2026']['calmar']}, turnover/day {s['y2026']['turnover_daily']}",
                  f"- non-2026 OOS: CAGR {s['non2026']['cagr']:+.1%}, maxDD {s['non2026']['maxDD']:.1%}",
                  f"- avg names {s['y2026']['avg_names']}, max single weight {s['y2026']['max_single_weight']:.1%}",
                  f"- blocked: limit-up buys {s['y2026'].get('limit_up_blocked_buys','?')}, "
                  f"limit-down sells {s['y2026'].get('limit_down_blocked_sells','?')}, "
                  f"suspension {s['y2026'].get('suspension_blocked','?')}, total rejected {s['y2026'].get('rejected_orders','?')}", ""]
    lines += ["## Capacity stress (2026 CAGR by slippage)", "", "| strategy | 8bps | 15bps | 30bps | 50bps | 100bps |", "|---|---|---|---|---|---|"]
    for name in STRATS:
        row = cap[cap["strategy"] == name].set_index("slippage_bps")["y2026_cagr"]
        lines.append(f"| {name} | " + " | ".join(f"{row.get(s, float('nan')):+.1%}" for s in SLIPPAGE) + " |")
    # collapse warning
    for name in STRATS:
        row = cap[cap["strategy"] == name].set_index("slippage_bps")["y2026_cagr"]
        c8, c30 = row.get(8, np.nan), row.get(30, np.nan)
        verdict = "TURNOVER/LIQUIDITY-DRIVEN — collapses by 30bps" if (c8 > 0 and c30 < 0.5 * c8) else "holds through 30bps"
        lines += ["", f"**{name}: {verdict}** (8bps {c8:+.1%} -> 30bps {c30:+.1%})"]
    (OUT / "daily_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote artifacts to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
