#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 9 step 2 — regime-aware sector wave state machine, head-to-head vs plain.

Uses the multi-regime walk-forward stock score (stage6_classical_2018,
alpha_20d, OOS 2020-2025 across COVID bull / 2022 bear / 2023 chop / 2024 crash
/ 2025 bull) as the stock selector — the only score panel that spans regimes.

Variants (all equal-weight, strict A-share engine, SAME dates / universe):
  plain          : top-N by alpha_20d across all-A, full gross (the baseline)
  static_laggard : top-N by alpha_20d inside fixed laggard (low mom_60) sectors
  static_leader  : ... inside fixed leader (high mom_60) sectors
  wave_full      : regime-aware sector selection, gross ALWAYS 1.0
                   (isolates dynamic SECTOR PICKING from market timing)
  wave_gross     : wave_full + the state machine's gross overlay (full system)

Rigor carried from Stage 8: SIZE-matched (same N), PHASE-matched (5 rebalance
offsets, averaged — kills timing luck), no concentration packaging. Reports
overall + per-regime (bull/bear/sideways) excess vs plain + bull-window capture
+ the wave mode distribution.
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
from quantagent.strategy.sector_wave_state import WaveConfig, select_wave  # noqa: E402

SCORE = "runtime/stage6_classical_2018/wf/walkforward_predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
WAVE_PANEL = "runtime/stage8_sector_rotation/sector_panel_wave.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
SCORE_COL = "alpha_20d"
SEC = "sector_level_1"
PHASES = [0, 4, 8, 12, 16]
PERIOD = 20
BULL = {"covid_2020": ("2020-03-23", "2021-02-10"),
        "rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
        "rally_2025_2026": ("2025-01-01", "2025-12-31")}


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def build_book(stock_day, sector_day, mkt_trend, *, rebal_dates, eval_dates, size,
               mode, n_sectors=0, cfg=WaveConfig()):
    """mode: 'plain' | 'laggard' | 'leader' | 'wave_full' | 'wave_gross'."""
    rows = {}
    modes_log = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=["score"])
        gross = 1.0
        if mode != "plain":
            secf = sector_day.get(d)
            if secf is None or secf.empty:
                continue
            if mode in ("laggard", "leader"):
                asc = (mode == "laggard")
                picks = list(secf.dropna(subset=["mom_60"]).sort_values("mom_60", ascending=asc)[SEC].head(n_sectors))
            else:
                tr = mkt_trend.get(d, 0.0)
                sel = select_wave(secf, tr, n_sectors=n_sectors, cfg=cfg)
                picks = sel["sectors"]
                modes_log[d] = sel["mode"]
                if mode == "wave_gross":
                    gross = sel["gross"]
            sd = sd[sd["sector"].isin(picks)]
        if sd.empty:
            continue
        sd = sd.sort_values("score", ascending=False).head(size)
        w = gross / len(sd)
        rows[d] = {s: w for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame(), modes_log
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    tw = tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")
    return tw, modes_log


def strict_nav(tw, win, smap_full):
    if tw.empty:
        return None
    return run_strict_backtest_v8(tw, win, sector_map=smap_full).nav


def regime_series(eqsector_daily: pd.Series) -> pd.Series:
    cum = (1 + eqsector_daily).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=eqsector_daily.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="30,50")
    ap.add_argument("--n-sectors", type=int, default=5)
    ap.add_argument("--start", default="2020-01-23")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    print("[load] multi-regime classical score + panels ...")
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", SCORE_COL]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    sc = sc[(sc.trade_date >= start) & (sc.trade_date <= end)]
    smap_full = pd.read_parquet(SECTOR)
    smap = smap_full[["symbol", SEC]].dropna().drop_duplicates("symbol")
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    wp = pd.read_parquet(WAVE_PANEL); wp["trade_date"] = pd.to_datetime(wp["trade_date"])
    wp = wp[(wp.trade_date >= start) & (wp.trade_date <= end)]
    eval_dates = sorted(win.trade_date.unique())
    print(f"  OOS {start.date()}..{end.date()} ({len(eval_dates)} days), score syms={sc.symbol.nunique()}")

    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = sc.merge(smap, on="symbol", how="left").merge(flags, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    df = df.rename(columns={SCORE_COL: "score", SEC: "sector"})
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    sector_day = {d: g for d, g in wp.groupby("trade_date")}

    # market 60d trend (equal-sector benchmark) for regime classification
    eqsector = wp.pivot_table(index="trade_date", columns=SEC, values="ret_eqw").mean(axis=1).dropna()
    cum = (1 + eqsector).cumprod()
    mkt_trend = (cum / cum.shift(60) - 1.0).to_dict()
    regime = regime_series(eqsector)
    print("  regime day mix:", regime.value_counts().to_dict())

    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())
    variants = [("plain", "plain"), ("static_laggard", "laggard"), ("static_leader", "leader"),
                ("wave_full", "wave_full"), ("wave_gross", "wave_gross")]

    all_rows = []
    mode_dist = {}
    for size in [int(x) for x in args.sizes.split(",")]:
        print(f"\n##### size={size} names, n_sectors={args.n_sectors}, 5 phases, strict #####")
        plain_navs = {}
        for label, mode in variants:
            navs = {}
            for ph in PHASES:
                rebal = dsorted[ph::PERIOD]
                tw, ml = build_book(stock_day, sector_day, mkt_trend, rebal_dates=rebal,
                                    eval_dates=eval_dates, size=size, mode=mode, n_sectors=args.n_sectors)
                nav = strict_nav(tw, win, smap_full)
                if nav is not None:
                    navs[ph] = nav
                if mode == "wave_full" and ph == 0 and ml:
                    mode_dist[size] = pd.Series(list(ml.values())).value_counts().to_dict()
            if mode == "plain":
                plain_navs = navs
            # aggregate across phases
            cagrs, dds, excs, regex = [], [], [], {r: [] for r in ["bull", "bear", "sideways"]}
            bullw = {k: [] for k in BULL}
            for ph, nav in navs.items():
                r = nav.pct_change().dropna()
                peak = nav.cummax(); dds.append(float(abs((nav/peak-1).min()))); cagrs.append(_ann(r))
                if ph in plain_navs:
                    pr = plain_navs[ph].pct_change().dropna()
                    idx = r.index.intersection(pr.index)
                    excs.append(_ann(r.reindex(idx)) - _ann(pr.reindex(idx)))
                    rg = regime.reindex(idx)
                    for rname in regex:
                        mm = rg == rname
                        if mm.sum() > 10:
                            regex[rname].append(_ann(r.reindex(idx)[mm]) - _ann(pr.reindex(idx)[mm]))
                base = eqsector.reindex(r.index)
                for k, (a, z) in BULL.items():
                    mm = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
                    if mm.sum() > 10:
                        bullw[k].append(float((1+r[mm]).prod() - (1+base[mm]).prod()))
            row = {"size": size, "variant": label,
                   "cagr": round(float(np.mean(cagrs)), 4), "cagr_std": round(float(np.std(cagrs)), 4),
                   "maxdd": round(float(np.mean(dds)), 4),
                   "calmar": round(float(np.mean(cagrs))/float(np.mean(dds)), 3) if np.mean(dds) > 1e-9 else None,
                   "exc_plain": round(float(np.mean(excs)), 4) if excs else None,
                   "exc_plain_std": round(float(np.std(excs)), 4) if excs else None,
                   "exc_win": round(float(np.mean([e > 0 for e in excs])), 2) if excs else None}
            for rname in regex:
                row[f"exc_{rname}"] = round(float(np.mean(regex[rname])), 4) if regex[rname] else None
            for k in BULL:
                row[f"bull_{k}"] = round(float(np.mean(bullw[k])), 4) if bullw[k] else None
            all_rows.append(row)
            ex = f"excPlain={row['exc_plain']:+.1%}±{row['exc_plain_std']:.1%}(win{row['exc_win']})" if row["exc_plain"] is not None else "(baseline)"
            print(f"  {label:<16} CAGR={row['cagr']:+.1%}±{row['cagr_std']:.1%} DD={row['maxdd']:.1%} "
                  f"Calmar={row['calmar']} {ex}  bull24={row.get('bull_rally_2024H2_2025')} bull25={row.get('bull_rally_2025_2026')}")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_DIR / "stage9_wave_results.csv", index=False)
    (OUT_DIR / "stage9_wave_results.json").write_text(
        json.dumps({"results": all_rows, "wave_mode_distribution": mode_dist,
                    "regime_day_mix": regime.value_counts().to_dict()}, indent=2, default=str))
    print(f"\n[write] {OUT_DIR/'stage9_wave_results.csv'}")
    print("wave_full mode distribution (phase0):", mode_dist)
    print("\nVERDICT: wave beats plain only if exc_plain>0 AND exc_plain>exc_plain_std AND exc_win>=0.6,")
    print("and ideally positive per-regime (esp bull capture).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
