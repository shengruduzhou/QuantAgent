#!/usr/bin/env python3
"""Regime-conditional hold-band parameter validation (strict protocol).

Motivation (2026 paper replay): fixed (entry30/exit150) ran target turnover
hot (33%/d one-sided) in the 2026 chop — band params should adapt to regime.

Protocol:
  * PIT regime label: benchmark 60d trailing return, lagged one day
    (bull > +5% / bear < −5% / sideways otherwise).
  * Grid over per-regime (entry, exit, n_hold) maps + fixed baselines,
    every combo through the SAME strict simulator as baseline_protocol.
  * Honest accounting: the grid winner on the full window is in-sample;
    adoption requires the candidate to beat the fixed baseline's excess in
    BOTH halves of the window (H1/H2 robustness, same rule as 2026-06-11
    hold-band validation).

Outputs <out>/{grid.csv, verdict.json}.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.portfolio.hold_band import (
    HoldBandConfig,
    build_hold_band_weights,
    build_regime_hold_band_weights,
    turnover_stats,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
ANN = 244

BAND = {
    "default_30_150": (30, 150, 50),
    "tight_20_100": (20, 100, 50),
    "wide_20_200": (20, 200, 50),
    "loose_50_200": (50, 200, 50),
    "defensive_10_60": (10, 60, 30),
}
GRID = {
    "bull": ["default_30_150", "loose_50_200"],
    "sideways": ["default_30_150", "tight_20_100", "wide_20_200"],
    "bear": ["tight_20_100", "defensive_10_60"],
}
FIXED_BASELINES = ["default_30_150", "tight_20_100", "wide_20_200", "loose_50_200"]


def _cfg(name: str) -> HoldBandConfig:
    e, x, n = BAND[name]
    return HoldBandConfig(n_hold=n, entry_rank=e, exit_rank=x, delay_days=1)


def _pit_regime(panel: pd.DataFrame) -> pd.Series:
    px = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    bench = px.pct_change(fill_method=None).mean(axis=1)
    cum = (1 + bench.fillna(0)).cumprod().shift(1)
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull",
                              np.where(trail < -0.05, "bear", "sideways")),
                     index=px.index)


def _half_metrics(nav: pd.Series, bench_daily: pd.Series, split: pd.Timestamp) -> dict:
    out = {}
    rets = nav.pct_change().dropna()
    for tag, mask in (("full", pd.Series(True, index=rets.index)),
                      ("H1", rets.index <= split), ("H2", rets.index > split)):
        r = rets[mask]
        if len(r) < 10:
            continue
        b = bench_daily.reindex(r.index).fillna(0.0)
        ann = float((1 + r).prod() ** (ANN / len(r)) - 1)
        bann = float((1 + b).prod() ** (ANN / len(b)) - 1)
        sharpe = float(r.mean() / (r.std(ddof=0) + 1e-12) * np.sqrt(ANN))
        navc = (1 + r).cumprod()
        maxdd = float(((navc.cummax() - navc) / navc.cummax()).max())
        out[tag] = {"ann": round(ann, 4), "excess_ann": round(ann - bann, 4),
                    "sharpe": round(sharpe, 3), "maxDD": round(maxdd, 4), "days": int(len(r))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=PREDS)
    ap.add_argument("--score-column", default="composite_score")
    ap.add_argument("--start", default="2024-08-28")
    ap.add_argument("--end", default="2026-05-07")
    ap.add_argument("--half-split", default="2025-07-01")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--output-dir", default="runtime/reports/v8/regime_band")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    split = pd.Timestamp(args.half_split)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    if args.score_column != "alpha_score":
        preds = preds.rename(columns={args.score_column: "alpha_score"})
    preds = preds[(preds["trade_date"] >= start) & (preds["trade_date"] <= end)]

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["trade_date"] >= start - pd.Timedelta(days=130))
                  & (panel["trade_date"] <= end + pd.Timedelta(days=10))]
    sector = pd.read_parquet(SECTOR)

    regime = _pit_regime(panel)
    flags = panel[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel.loc[panel["trade_date"] >= start, "trade_date"].unique())
    sim_panel = panel[panel["trade_date"] >= start - pd.Timedelta(days=10)]

    px = sim_panel.pivot_table(index="trade_date", columns="symbol", values="close")
    bench_daily = px.pct_change(fill_method=None).mean(axis=1).dropna()

    reg_counts = regime.reindex(pd.DatetimeIndex(trade_dates)).value_counts().to_dict()
    print(f"regime days in window: {reg_counts}", flush=True)

    def run(tw: pd.DataFrame) -> tuple[dict, dict]:
        res = run_strict_backtest_v8(
            tw, sim_panel, sector_map=sector,
            config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0,
                                                   slippage_bps=args.slippage_bps))
        return _half_metrics(res.nav, bench_daily, split), turnover_stats(tw)

    rows = []
    for name in FIXED_BASELINES:
        tw = build_hold_band_weights(preds, config=_cfg(name), trade_dates=trade_dates)
        m, ts = run(tw)
        rows.append({"config": f"fixed:{name}", "bull": name, "sideways": name, "bear": name,
                     **{f"{k}_{kk}": vv for k, v in m.items() for kk, vv in v.items()},
                     "mean_turnover": round(ts["mean_daily_turnover"], 4)})
        print(f"fixed:{name:18} full {m['full']['excess_ann']:+.2%} "
              f"H1 {m.get('H1', {}).get('excess_ann', float('nan')):+.2%} "
              f"H2 {m.get('H2', {}).get('excess_ann', float('nan')):+.2%} "
              f"turn {ts['mean_daily_turnover']:.2%}", flush=True)

    for bull, side, bear in itertools.product(GRID["bull"], GRID["sideways"], GRID["bear"]):
        if bull == side == bear:
            continue  # identical to a fixed baseline
        config_map = {"bull": _cfg(bull), "sideways": _cfg(side), "bear": _cfg(bear)}
        tw = build_regime_hold_band_weights(preds, config_map=config_map,
                                            regime_by_date=regime, trade_dates=trade_dates)
        m, ts = run(tw)
        rows.append({"config": f"bull={bull}|side={side}|bear={bear}",
                     "bull": bull, "sideways": side, "bear": bear,
                     **{f"{k}_{kk}": vv for k, v in m.items() for kk, vv in v.items()},
                     "mean_turnover": round(ts["mean_daily_turnover"], 4)})
        print(f"{bull[:12]}/{side[:12]}/{bear[:12]:14} full {m['full']['excess_ann']:+.2%} "
              f"H1 {m.get('H1', {}).get('excess_ann', float('nan')):+.2%} "
              f"H2 {m.get('H2', {}).get('excess_ann', float('nan')):+.2%} "
              f"turn {ts['mean_daily_turnover']:.2%}", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "grid.csv", index=False)

    base = grid[grid["config"] == "fixed:default_30_150"].iloc[0]
    cand = grid[~grid["config"].str.startswith("fixed:")].copy()
    cand["h1_impr"] = cand["H1_excess_ann"] - base["H1_excess_ann"]
    cand["h2_impr"] = cand["H2_excess_ann"] - base["H2_excess_ann"]
    cand["worst_half_impr"] = cand[["h1_impr", "h2_impr"]].min(axis=1)
    cand = cand.sort_values("worst_half_impr", ascending=False)
    best = cand.iloc[0]

    verdict = {
        "verdict": "ADOPT" if best["worst_half_impr"] > 0 else "KEEP_FIXED",
        "window": f"{args.start}..{args.end}", "half_split": args.half_split,
        "regime_days": {str(k): int(v) for k, v in reg_counts.items()},
        "fixed_baseline": base.to_dict(),
        "best_candidate": best.to_dict(),
        "top5_by_worst_half_improvement": cand.head(5)[
            ["config", "full_excess_ann", "H1_excess_ann", "H2_excess_ann",
             "h1_impr", "h2_impr", "worst_half_impr", "full_maxDD", "mean_turnover"]
        ].to_dict("records"),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2,
                                                     default=str), encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ("verdict", "regime_days")}, ensure_ascii=False))
    print(f"best: {best['config']} worst-half improvement {best['worst_half_impr']:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
