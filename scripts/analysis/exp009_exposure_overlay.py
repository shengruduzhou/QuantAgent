#!/usr/bin/env python3
"""EXP-009 / H-009: a-priori drawdown/regime gross-exposure overlays.

Pre-registered rules (HYPOTHESIS_REGISTRY.md, frozen BEFORE execution):
  R1_dd_tiers    bench 60d rolling-peak drawdown at t-1: <8% -> 1.0,
                 8-15% -> 0.5, >=15% -> 0.3
  R2_trend_ma60  bench close(t-1) >= MA60 -> 1.0 else 0.5
  R3_vol_tiers   bench 20d realized vol (annualized) at t-1: <25% -> 1.0,
                 25-40% -> 0.5, >=40% -> 0.3

Carrier book: frozen C3_ema0.7 target weights rebuilt exactly as EXP-008
(same folds, no retraining). gross in (0,1] -> scaled-out fraction is cash,
never leverage. Trigger uses ONLY the equal-weight all-A bench (panel closes)
observed through t-1, applied to weights executed at t (delay-1 parity).

Evaluation: strict variant C on the four H-008 folds. Outputs
wf_h008/exp009_overlay/{results.json, overlay_fold_metrics.csv}.
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
from exp008_walkforward_eval import (  # noqa: E402
    FOLDS, TOP_K, ANN, build_candidates, cagr, max_dd, sleeve_frame,
)

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp009_overlay"
CARRIER = "C3_ema0.7"
QUARANTINE_START = pd.Timestamp("2025-09-01")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def bench_series(oos_s: pd.Timestamp, oos_e: pd.Timestamp) -> pd.Series:
    """Equal-weight all-A daily returns with 150d warmup before the fold."""
    panel = pd.read_parquet(
        REPO / bp.PANEL, columns=["symbol", "trade_date", "close"],
        filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=160)),
                 ("trade_date", "<=", oos_e)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return bp._bench_daily(panel, sorted(panel["trade_date"].unique()))


def gross_series(bench: pd.Series, rule: str) -> pd.Series:
    nav = (1.0 + bench).cumprod()
    if rule == "R1_dd_tiers":
        dd = 1.0 - nav / nav.rolling(60, min_periods=20).max()
        g = pd.Series(1.0, index=nav.index)
        g[dd >= 0.08] = 0.5
        g[dd >= 0.15] = 0.3
    elif rule == "R2_trend_ma60":
        ma = nav.rolling(60, min_periods=40).mean()
        g = pd.Series(1.0, index=nav.index)
        g[nav < ma] = 0.5
    elif rule == "R3_vol_tiers":
        vol = bench.rolling(20, min_periods=15).std() * np.sqrt(ANN)
        g = pd.Series(1.0, index=nav.index)
        g[vol >= 0.25] = 0.5
        g[vol >= 0.40] = 0.3
    else:
        raise ValueError(rule)
    # observe at t-1, apply at t (no lookahead)
    return g.shift(1).fillna(1.0)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)
    rules = ("R1_dd_tiers", "R2_trend_ma60", "R3_vol_tiers")
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
            tw_scaled = tw.mul(g, axis=0)
            res = run_strict_backtest_v8(tw_scaled, panel, sector_map=sector, config=cfg)
            nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
            r = nav.pct_change().dropna()
            m = res.metrics
            results[rule][fold] = {
                "cagr": round(m.annualized_return, 4),
                "maxdd": round(m.max_drawdown, 4),
                "sharpe": round(m.sharpe, 3),
                "turnover": round(float(m.turnover), 4),
                "mean_gross": round(float(g.loc[(g.index >= oos_s)].mean()), 3),
                "days_derisked": int((g.loc[g.index >= oos_s] < 1.0).sum()),
            }
            (OUT / f"daily_{rule}_{fold}.csv").write_text(r.to_csv())
            print(f"{fold} {CARRIER}+{rule:14s} CAGR {m.annualized_return:+.1%} "
                  f"DD {m.max_drawdown:.1%} turn {m.turnover:.3f} "
                  f"mean_gross {results[rule][fold]['mean_gross']}", flush=True)

    # aggregates + registered gates vs frozen H-008 baseline numbers
    base = {"median_cagr": 0.3302, "worst_dd": 0.2503, "f2_cagr": -0.2986,
            "max_turnover": 0.2594, "median_floor": 0.2802}
    summary: dict[str, object] = {"carrier": CARRIER, "baseline_frozen": base, "rules": {}}
    for rule in rules:
        cs = [results[rule][f]["cagr"] for f in FOLDS]
        dd = [results[rule][f]["maxdd"] for f in FOLDS]
        to = [results[rule][f]["turnover"] for f in FOLDS]
        med = float(np.median(cs))
        gates = {
            "worst_dd_reduced": max(dd) < base["worst_dd"],
            "f2_improved": results[rule]["F2"]["cagr"] > base["f2_cagr"],
            "turnover_ok": max(to) <= min(base["max_turnover"] + 0.05, 0.35),
            "median_not_degraded": med >= base["median_floor"],
        }
        summary["rules"][rule] = {
            "fold_cagrs": cs, "median_cagr": round(med, 4), "min_cagr": round(min(cs), 4),
            "worst_dd": round(max(dd), 4), "max_turnover": round(max(to), 4),
            "per_fold": results[rule], "gates": gates, "all_gates": all(gates.values()),
        }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for rule in rules:
        for f in FOLDS:
            rows.append({"rule": rule, "fold": f, **results[rule][f]})
    pd.DataFrame(rows).to_csv(OUT / "overlay_fold_metrics.csv", index=False)
    print(json.dumps({r: summary["rules"][r]["gates"] for r in rules}, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
