#!/usr/bin/env python3
"""Stage 5 — long-history STRICT validation of turnover-reduced daily books.

IMPORTANT (stated per spec): the model was trained on data <=2024-06 and its
predictions exist ONLY for 2024-08-09..2026-05-07. So 2018->2024 is NOT available
as clean OOS (it was in-sample / no pre-2024 predictions; would need walk-forward
retraining). The "long" window here is the FULL 21-month OOS span 2024-08..2026-05
(~418 days) — long enough to test cost-curve monotonicity that the ~90-day
sub-windows could not.

Strict backtest ONLY for returns (the fast proxy was broken, discarded). One strict
run per (config, cost) over the full OOS; sub-window metrics are sliced from its NAV.
Fixed pre-specified configs (NO search, NO 2026 tuning).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp
from stage4_cost_robust import load, build_book
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

OUT = Path("runtime/reports/v89_closed_loop/stage5"); OUT.mkdir(parents=True, exist_ok=True)
ANN = 244
FULL = ("2024-08-09", "2026-05-13")
SUBWIN = {"val": ("2024-08-28", "2025-08-31"), "non2026": ("2025-09-01", "2025-12-31"),
          "y2026": ("2026-01-02", "2026-05-13"), "full_oos": FULL}
SLIPS = [8, 15, 30, 50, 100]


def window_metrics(nav: pd.Series, a, b) -> dict:
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))].dropna()
    if len(s) < 5:
        return {"cagr": None, "total": None, "maxDD": None, "sharpe": None, "calmar": None}
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    n = len(s)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (ANN / n) - 1)
    dd = float((s / s.cummax() - 1).min())
    r = s.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    return {"cagr": round(cagr, 4), "total": round(total, 4), "maxDD": round(dd, 4),
            "sharpe": round(sharpe, 3), "calmar": round(cagr / abs(dd), 3) if dd < 0 else 0.0}


def mech_metrics(w: pd.DataFrame) -> dict:
    """Deterministic mechanical metrics from the weight matrix (no strict)."""
    wd = w[(w.index >= pd.Timestamp(FULL[0])) & (w.index <= pd.Timestamp(FULL[1]))]
    held = (wd > 1e-9)
    turn = float(wd.diff().abs().sum(axis=1).mean() / 2)  # one-way/day
    n_names = float(held.sum(axis=1).mean())
    maxw = float(wd.max().max())
    # avg holding period: mean run-length of held per name
    runs = []
    for col in wd.columns[(held.any())]:
        h = held[col].astype(int).values
        if h.sum() == 0:
            continue
        # count consecutive runs
        d = np.diff(np.concatenate([[0], h, [0]]))
        starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    avg_hold = float(np.mean(runs)) if runs else 0.0
    n_entries = int(len(runs))
    return {"turnover_oneway_day": round(turn, 4), "ann_turnover": round(turn * ANN, 1),
            "avg_names": round(n_names, 1), "max_single_weight": round(maxw, 4),
            "avg_holding_days": round(avg_hold, 1), "n_entries": n_entries}


def strict_full(w: pd.DataFrame, panel, sector, slip) -> dict:
    di = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    wf = w[(w.index >= pd.Timestamp(FULL[0])) & (w.index <= pd.Timestamp(FULL[1]))]
    wf = wf.loc[:, (wf != 0).any(axis=0)]
    pos = di.searchsorted(wf.index) + 1
    keep = pos < len(di); tw = wf.iloc[keep]; tw.index = di[pos[keep]]
    sim = panel[(panel["trade_date"] >= pd.Timestamp(FULL[0]) - pd.Timedelta(days=10))]
    res = run_strict_backtest_v8(tw, sim, sector_map=sector,
                                 config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=float(slip)))
    fo = res.failed_orders
    blocked = {"rejected": int(len(fo))}
    return {"nav": res.nav, "total_cost": float(res.metrics.total_cost),
            "n_fills": int(res.metrics.n_fills), "n_trades": int(res.metrics.n_trades), **blocked}


def main() -> int:
    panel, sector, px, fwd1d, adv20, vol20, elig, comp = load()
    print("loaded.", flush=True)

    def bk(book, k, buf, fr):
        return build_book(comp[book], elig, top_k=k, buffer_rank=buf, rebalance_every=fr)

    configs = {
        "w210_k10_daily": bk("w210", 10, 10, 1),
        "w111_k5_daily": bk("w111", 5, 5, 1),
        "w210_buffer20_freq2": bk("w210", 10, 20, 2), "w210_buffer20_freq3": bk("w210", 10, 20, 3),
        "w210_buffer25_freq2": bk("w210", 10, 25, 2), "w210_buffer25_freq3": bk("w210", 10, 25, 3),
        "w210_buffer30_freq2": bk("w210", 10, 30, 2), "w210_buffer30_freq3": bk("w210", 10, 30, 3),
        "w111_buffer20_freq2": bk("w111", 5, 20, 2), "w111_buffer20_freq3": bk("w111", 5, 20, 3),
        "w111_buffer30_freq2": bk("w111", 5, 30, 2), "w111_buffer30_freq3": bk("w111", 5, 30, 3),
    }
    # hybrids
    a210 = bk("w210", 10, 25, 3); a111 = bk("w111", 5, 20, 3)
    configs["hyb_70_30"] = 0.7 * a210 + 0.3 * a111
    configs["hyb_50_50"] = 0.5 * a210 + 0.5 * a111
    configs["hyb_30_70"] = 0.3 * a210 + 0.7 * a111

    mech = {name: mech_metrics(w) for name, w in configs.items()}
    pd.DataFrame(mech).T.reset_index().rename(columns={"index": "config"}).to_csv(OUT / "stage5_turnover_reduction_report.csv", index=False)

    rows, cost_curve, mono = [], [], []
    navs = {}
    for name, w in configs.items():
        for slip in SLIPS:
            sf = strict_full(w, panel, sector, slip)
            navs[(name, slip)] = sf["nav"]
            wm = {win: window_metrics(sf["nav"], *SUBWIN[win]) for win in SUBWIN}
            cost_curve.append({"config": name, "slippage_bps": slip,
                               "full_oos_cagr": wm["full_oos"]["cagr"], "val_cagr": wm["val"]["cagr"],
                               "non2026_cagr": wm["non2026"]["cagr"], "y2026_cagr": wm["y2026"]["cagr"],
                               "full_oos_maxDD": wm["full_oos"]["maxDD"], "full_oos_sharpe": wm["full_oos"]["sharpe"],
                               "total_cost": round(sf["total_cost"], 0), "n_fills": sf["n_fills"], "rejected": sf["rejected"]})
            row = {"config": name, "slippage_bps": slip, **mech[name]}
            for win in SUBWIN:
                for kk, vv in wm[win].items():
                    row[f"{win}_{kk}"] = vv
            rows.append(row)
        print(f"  {name}: full-OOS CAGR by slip "
              f"{[next(c['full_oos_cagr'] for c in cost_curve if c['config']==name and c['slippage_bps']==s) for s in SLIPS]}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "stage5_long_history_strict_results.csv", index=False)
    cc = pd.DataFrame(cost_curve); cc.to_csv(OUT / "stage5_cost_curve_by_config.csv", index=False)

    # monotonicity audit: full_oos cagr should be non-increasing in slippage
    for name in configs:
        sub = cc[cc["config"] == name].sort_values("slippage_bps")
        cur = sub["full_oos_cagr"].tolist()
        viol = sum(1 for i in range(1, len(cur)) if (cur[i] is not None and cur[i-1] is not None and cur[i] > cur[i-1] + 1e-6))
        mono.append({"config": name, "monotonic": viol == 0, "violations": viol,
                     "full_oos_curve": cur})
    pd.DataFrame(mono).to_csv(OUT / "stage5_monotonicity_audit.csv", index=False)

    # OOS leaderboard ranked by full-OOS 30bps CAGR (most stable, realistic cost)
    lb = cc[cc["slippage_bps"] == 30].copy().sort_values("full_oos_cagr", ascending=False)
    lb = lb.merge(pd.DataFrame(mech).T.reset_index().rename(columns={"index": "config"})[["config", "turnover_oneway_day", "avg_holding_days"]], on="config")
    lb.to_csv(OUT / "stage5_oos_only_leaderboard.csv", index=False)

    # delta vs base + robustness criteria (vs w210_k10_daily base)
    base = "w210_k10_daily"
    base30 = cc[(cc["config"] == base) & (cc["slippage_bps"] == 30)].iloc[0]
    base_turn = mech[base]["turnover_oneway_day"]
    robust = []
    for name in configs:
        if name == base:
            continue
        c30 = cc[(cc["config"] == name) & (cc["slippage_bps"] == 30)].iloc[0]
        c50 = cc[(cc["config"] == name) & (cc["slippage_bps"] == 50)].iloc[0]
        turn_red = 1 - mech[name]["turnover_oneway_day"] / base_turn if base_turn else 0
        mono_ok = bool([m for m in mono if m["config"] == name][0]["monotonic"])
        crit = {"config": name, "turnover_reduction": round(turn_red, 3),
                "non2026_30bps": c30["non2026_cagr"], "y2026_30bps": c30["y2026_cagr"],
                "full_oos_30bps": c30["full_oos_cagr"], "full_oos_50bps": c50["full_oos_cagr"],
                "delta_full_oos_30bps_vs_base": (round(c30["full_oos_cagr"] - base30["full_oos_cagr"], 4)
                                                 if c30["full_oos_cagr"] is not None else None),
                "monotonic": mono_ok,
                "cost_robust": bool(turn_red >= 0.30 and (c30["non2026_cagr"] or -1) > 0
                                    and (c30["y2026_cagr"] or -1) > 0 and (c50["full_oos_cagr"] or -1) > -0.1 and mono_ok)}
        robust.append(crit)
    pd.DataFrame(robust).to_csv(OUT / "stage5_robustness_criteria.csv", index=False)

    # best cost-robust config (by full_oos 30bps among those passing turnover>=30% + monotonic)
    cand = [r for r in robust if r["turnover_reduction"] >= 0.30 and r["monotonic"] and (r["full_oos_30bps"] is not None)]
    best = max(cand, key=lambda r: r["full_oos_30bps"])["config"] if cand else base
    bw = configs[best]
    bwo = bw[(bw.index >= pd.Timestamp(FULL[0]))].reset_index().rename(columns={"index": "trade_date"})
    bwo.melt(id_vars="trade_date", var_name="symbol", value_name="weight").query("weight>1e-9").to_parquet(OUT / "stage5_best_config_positions.parquet", index=False)

    # capacity curve (long history) for best config
    pos = bw[(bw.index >= pd.Timestamp(FULL[0]))]
    adv_w = adv20.reindex(index=pos.index, columns=pos.columns)
    cap = []
    for size in (1e6, 3e6, 5e6, 1e7, 3e7, 5e7, 1e8):
        traded = (size * pos.diff().abs()).div(adv_w).replace([np.inf, -np.inf], np.nan)
        avg_p = float(traded.stack().mean()) if traded.notna().any().any() else None
        p95 = float(traded.stack().quantile(0.95)) if traded.notna().any().any() else None
        impact = round(10.0 * np.sqrt(max(avg_p or 0, 0)) * 100, 1)
        cap.append({"size_rmb": size, "avg_participation": round(avg_p, 5) if avg_p else None,
                    "p95_participation": round(p95, 5) if p95 else None, "est_impact_bps": impact,
                    "eff_slippage_bps": round(8 + impact, 1)})
    pd.DataFrame(cap).to_csv(OUT / "stage5_capacity_curve_long_history.csv", index=False)

    # report
    L = ["# Stage 5 — Long-history strict validation (cost-robust daily book)", "",
         "**Data note:** model trained <=2024-06; predictions exist ONLY 2024-08..2026-05. "
         "2018->2024 is NOT clean OOS (in-sample, no pre-2024 predictions) — would need walk-forward retraining. "
         "'Long' window = full 21-month OOS 2024-08..2026-05 (~418d).", "",
         "## Mechanical (turnover reduction, deterministic)", "",
         "| config | turnover/day | ann turn | avg hold(d) | avg names | max wt |", "|---|---|---|---|---|---|"]
    for name in configs:
        m = mech[name]
        L.append(f"| {name} | {m['turnover_oneway_day']} | {m['ann_turnover']} | {m['avg_holding_days']} | {m['avg_names']} | {m['max_single_weight']} |")
    L += ["", "## OOS leaderboard (full-OOS net CAGR @30bps)", "", "| config | full-OOS 30bps | non2026 30bps | 2026 30bps | turnover/day |", "|---|---|---|---|---|"]
    for _, r in lb.iterrows():
        L.append(f"| {r['config']} | {r['full_oos_cagr']} | {r['non2026_cagr']} | {r['y2026_cagr']} | {r['turnover_oneway_day']} |")
    L += ["", "## Monotonicity audit (full-OOS CAGR vs slippage)", ""]
    nmono = [m['config'] for m in mono if not m['monotonic']]
    L.append(f"- monotonic configs: {sum(m['monotonic'] for m in mono)}/{len(mono)}; non-monotonic: {nmono or 'none'}")
    L += ["", "## Cost-robust candidates (turnover>=30% cut, +30bps both OOS, monotonic)", ""]
    passed = [r['config'] for r in robust if r['cost_robust']]
    L.append(f"- PASS: {passed or 'NONE'}")
    L += ["", f"## Best cost-robust config: **{best}**", "",
          "## Capacity curve (best config)", "", "| size | avg part | impact bps |", "|---|---|---|"]
    for c in cap:
        L.append(f"| {c['size_rmb']:.0e} | {c['avg_participation']} | {c['est_impact_bps']} |")
    (OUT / "stage5_long_history_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote artifacts to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
