#!/usr/bin/env python3
"""Stage 8 step 3 — clean sector-TIMING diagnostic (isolated from stock pick).

The 450-config book search showed concentrated within-sector baskets blow up.
That conflates two questions. This isolates the one that matters first:

  *Does ranking SW1 sectors by a signal and holding the top ones as whole
   equal-weight baskets beat holding ALL sectors equally?*

We treat each SW1 sector as a tradable "asset" whose return is the sector
equal-weight member return (`ret_eqw` from the sector panel). For every signal
(both momentum AND reversal direction) we measure:

  1. rank-IC of signal_t vs forward N-day sector return (predictive power,
     direction-agnostic — the cleanest evidence of sector-timing alpha);
  2. a top-N long basket NAV vs the equal-sector benchmark, after a turnover
     cost, with bull-window capture.

No broker sim needed — this is sector-index level, runs in seconds, and tells
us whether ANY sector-timing edge exists before we spend effort on books.

Honesty: sector returns here are equal-weight (carry the small-cap breadth
phantom + survivorship), so absolute levels are inflated — but excess vs the
equal-sector benchmark is apples-to-apples (same phantom on both sides), which
is exactly the quantity that answers the timing question.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
RET = "ret_eqw"

SIGNALS = ["mom_20", "mom_60", "mom_120", "rs_20", "rs_60", "rs_120",
           "rmom_60", "breadth_ma60", "breadth_high60", "amt_accel", "vol_60", "drawdown"]
BULL = {
    "covid_2020": ("2020-03-23", "2021-02-10"),
    "rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
    "rally_2025_2026": ("2025-01-01", "2026-05-18"),
}


def _ann(daily: pd.Series) -> float:
    n = len(daily)
    return float((1 + daily).prod() ** (ANN / n) - 1) if n else float("nan")


def forward_ret(wide_ret: pd.DataFrame, h: int) -> pd.DataFrame:
    """h-day forward sector return, indexed at signal date t."""
    logr = np.log1p(wide_ret.clip(lower=-0.99))
    fwd = logr.rolling(h).sum().shift(-h)
    return np.expm1(fwd)


def signal_ic(sig_wide: pd.DataFrame, fwd: pd.DataFrame) -> tuple[float, float]:
    """Mean cross-sectional rank-IC of signal vs forward return + t-stat."""
    ics = []
    for d in sig_wide.index:
        s = sig_wide.loc[d]
        f = fwd.loc[d] if d in fwd.index else None
        if f is None:
            continue
        m = s.notna() & f.notna()
        if m.sum() < 8:
            continue
        rho = spearmanr(s[m], f[m]).statistic
        if np.isfinite(rho):
            ics.append(rho)
    if len(ics) < 20:
        return float("nan"), float("nan")
    ics = np.array(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)) + 1e-12)
    return float(ics.mean()), float(t)


def topn_nav(sig_wide: pd.DataFrame, wide_ret: pd.DataFrame, *, top_n: int,
             rebalance: int, ascending: bool, cost_bps: float = 18.0) -> pd.Series:
    """Hold top-N sectors (by signal, dir=ascending) as eqw baskets; NAV after cost."""
    dates = list(sig_wide.index)
    rebal = set(dates[::rebalance])
    w = pd.DataFrame(0.0, index=dates, columns=sig_wide.columns)
    cur = pd.Series(0.0, index=sig_wide.columns)
    for d in dates:
        if d in rebal:
            s = sig_wide.loc[d].dropna()
            if len(s) >= top_n:
                picks = s.sort_values(ascending=ascending).head(top_n).index
                cur = pd.Series(0.0, index=sig_wide.columns)
                cur[picks] = 1.0 / top_n
        w.loc[d] = cur.values
    w_eff = w.shift(1).fillna(0.0)
    gross = (w_eff * wide_ret.reindex(w.index)).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(1).fillna(0.0) * (cost_bps / 1e4)
    return (1 + (gross - cost).fillna(0.0)).cumprod()


def main() -> int:
    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[sp.trade_date >= "2018-01-01"]
    wide_ret = sp.pivot_table(index="trade_date", columns="sector_level_1", values=RET)
    dates = wide_ret.index
    eqsector = wide_ret.mean(axis=1)  # equal-sector benchmark daily return
    print(f"[diag] sectors={wide_ret.shape[1]} dates={len(dates)} "
          f"{dates.min().date()}..{dates.max().date()}")
    print(f"[diag] equal-sector bench ann={_ann(eqsector.dropna()):+.2%}\n")

    # ---- 1. signal IC vs forward 20d sector return (both raw; sign tells direction) ----
    fwd20 = forward_ret(wide_ret, 20)
    print("=== sector-timing rank-IC (signal_t vs fwd-20d sector ret) ===")
    ic_rows = []
    for sig in SIGNALS:
        sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=sig).reindex(dates)
        ic, t = signal_ic(sw, fwd20)
        ic_rows.append({"signal": sig, "ic20": round(ic, 4), "t_stat": round(t, 2)})
        print(f"   {sig:<16} IC={ic:+.4f}  t={t:+.2f}")
    pd.DataFrame(ic_rows).to_csv(OUT_DIR / "sector_signal_ic.csv", index=False)

    # ---- 2. whole-sector top-N baskets vs equal-sector, both directions ----
    print("\n=== top-N whole-sector basket excess vs EQUAL-SECTOR bench (after 18bps cost) ===")
    rows = []
    bench_ann = _ann(eqsector.dropna())
    for sig in SIGNALS:
        sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=sig).reindex(dates)
        for direction, asc in [("high", False), ("low", True)]:
            for top_n in (3, 5, 8):
                for rb in (20,):
                    nav = topn_nav(sw, wide_ret, top_n=top_n, rebalance=rb, ascending=asc)
                    r = nav.pct_change().dropna()
                    peak = nav.cummax()
                    dd = float(abs((nav / peak - 1).min()))
                    cagr = _ann(r)
                    # bull capture
                    bw = {}
                    for name, (a, z) in BULL.items():
                        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
                        if m.sum() < 10:
                            continue
                        rr = r[m]
                        bb = eqsector.reindex(r.index)[m]
                        bw[name] = round(float((1 + rr).prod() - (1 + bb).prod()), 4)
                    rows.append({
                        "signal": sig, "dir": direction, "top_n": top_n, "rb": rb,
                        "cagr": round(cagr, 4), "maxdd": round(dd, 4),
                        "calmar": round(cagr / dd, 3) if dd > 1e-9 else None,
                        "exc_eqsector": round(cagr - bench_ann, 4),
                        "bull_2024_2025": bw.get("rally_2024H2_2025"),
                        "bull_2025_2026": bw.get("rally_2025_2026"),
                        "bull_covid": bw.get("covid_2020"),
                    })
    res = pd.DataFrame(rows).sort_values("exc_eqsector", ascending=False)
    res.to_csv(OUT_DIR / "sector_timing_baskets.csv", index=False)
    cols = ["signal", "dir", "top_n", "cagr", "maxdd", "calmar", "exc_eqsector",
            "bull_2024_2025", "bull_2025_2026", "bull_covid"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("TOP 15 by excess vs equal-sector:")
        print(res[cols].head(15).to_string(index=False))
        print("\nBOTTOM 5:")
        print(res[cols].tail(5).to_string(index=False))
    print(f"\n[write] {OUT_DIR/'sector_timing_baskets.csv'} , {OUT_DIR/'sector_signal_ic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
