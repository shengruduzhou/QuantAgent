#!/usr/bin/env python3
"""EXP-010 / H-010: hysteresis fixes for the R2 trend overlay (FINAL overlay
iteration this cycle — line stops after this run regardless of outcome).

Pre-registered (HYPOTHESIS_REGISTRY.md H-010, N=2):
  R2a_confirm5   switch 1.0 -> 0.5 only after 5 consecutive closes below MA60;
                 switch back only after 5 consecutive closes at/above MA60
  R2b_ema_gross  g_t = EMA(alpha=0.2) of the raw binary R2 gross

Carrier, folds, evaluation, gates: identical to EXP-009.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import FOLDS, TOP_K, build_candidates, sleeve_frame  # noqa: E402
from exp009_exposure_overlay import CARRIER, bench_series  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp010_hysteresis"
QUARANTINE_START = pd.Timestamp("2025-09-01")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def gross_series(bench: pd.Series, rule: str) -> pd.Series:
    nav = (1.0 + bench).cumprod()
    ma = nav.rolling(60, min_periods=40).mean()
    below = (nav < ma).to_numpy()
    if rule == "R2a_confirm5":
        g = np.ones(len(nav))
        state = 1.0
        run_below = run_above = 0
        for i in range(len(nav)):
            if below[i]:
                run_below += 1; run_above = 0
            else:
                run_above += 1; run_below = 0
            if state == 1.0 and run_below >= 5:
                state = 0.5
            elif state == 0.5 and run_above >= 5:
                state = 1.0
            g[i] = state
        out = pd.Series(g, index=nav.index)
    elif rule == "R2b_ema_gross":
        raw = pd.Series(np.where(below, 0.5, 1.0), index=nav.index)
        out = raw.ewm(alpha=0.2, adjust=False).mean()
    else:
        raise ValueError(rule)
    return out.shift(1).fillna(1.0)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)
    rules = ("R2a_confirm5", "R2b_ema_gross")
    results: dict[str, dict] = {r: {} for r in rules}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)[CARRIER]
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        p = carrier.merge(flags, on=["symbol", "trade_date"], how="left")
        tw = bp._target_weights(p, "alpha_score", TOP_K, eligible_only=True,
                                delay_days=1, trade_dates=trade_dates)
        bench = bench_series(oos_s, oos_e)
        for rule in rules:
            g = gross_series(bench, rule).reindex(tw.index).fillna(1.0)
            res = run_strict_backtest_v8(tw.mul(g, axis=0), panel, sector_map=sector, config=cfg)
            m = res.metrics
            results[rule][fold] = {
                "cagr": round(m.annualized_return, 4), "maxdd": round(m.max_drawdown, 4),
                "sharpe": round(m.sharpe, 3), "turnover": round(float(m.turnover), 4),
                "mean_gross": round(float(g.loc[g.index >= oos_s].mean()), 3),
            }
            print(f"{fold} {CARRIER}+{rule:14s} CAGR {m.annualized_return:+.1%} "
                  f"DD {m.max_drawdown:.1%} turn {m.turnover:.3f} "
                  f"mean_gross {results[rule][fold]['mean_gross']}", flush=True)

    base = {"worst_dd": 0.2503, "f2_cagr": -0.2986, "turn_cap": 0.309, "median_floor": 0.2802}
    summary: dict[str, object] = {"carrier": CARRIER, "baseline_frozen": base, "rules": {}}
    for rule in rules:
        cs = [results[rule][f]["cagr"] for f in FOLDS]
        dd = [results[rule][f]["maxdd"] for f in FOLDS]
        to = [results[rule][f]["turnover"] for f in FOLDS]
        med = float(np.median(cs))
        gates = {
            "worst_dd_reduced": max(dd) < base["worst_dd"],
            "f2_improved": results[rule]["F2"]["cagr"] > base["f2_cagr"],
            "turnover_ok": max(to) <= base["turn_cap"],
            "median_not_degraded": med >= base["median_floor"],
        }
        summary["rules"][rule] = {"fold_cagrs": cs, "median_cagr": round(med, 4),
                                  "worst_dd": round(max(dd), 4), "max_turnover": round(max(to), 4),
                                  "per_fold": results[rule], "gates": gates,
                                  "all_gates": all(gates.values())}
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({r: summary["rules"][r] for r in rules},
                     indent=2, default=str)[:1200])
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
