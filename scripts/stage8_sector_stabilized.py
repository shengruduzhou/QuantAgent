#!/usr/bin/env python3
"""Stage 8 step 6 — 企稳 (stabilization) reversal + diversification.

The vanilla risk overlay failed (anti-correlated with reversal). The right DD
fix for a reversal book is to (a) buy laggards that are TURNING UP, not falling
knives, and (b) hold more sectors. We build a composite sector score:

    score = z(-mom_60)            # laggard (long-term weak = cheap)
          + w_stab * z(stab)      # but stabilizing / inflows turning up

where `stab` ∈ {mom_5-ish via mom_20-mom_60 accel, amt_accel, breadth_ma60
turning up}. Hold top-N. Compare Calmar / excess / bull-capture vs plain
laggard, sweeping top_n and the stabilization weight, at the fast sector-basket
level (excess vs equal-sector is phantom-free).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
RET = "ret_eqw"
BULL = {"rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
        "rally_2025_2026": ("2025-01-01", "2026-05-18"),
        "covid_2020": ("2020-03-23", "2021-02-10")}


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def _z(df):  # cross-sectional z per row (date)
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def basket_daily(score: pd.DataFrame, wide_ret, *, top_n, rebalance, cost_bps=18.0):
    dates = list(score.index); rebal = set(dates[::rebalance])
    w = pd.DataFrame(0.0, index=dates, columns=score.columns)
    cur = pd.Series(0.0, index=score.columns)
    for d in dates:
        if d in rebal:
            s = score.loc[d].dropna()
            if len(s) >= top_n:
                picks = s.sort_values(ascending=False).head(top_n).index  # high score = buy
                cur = pd.Series(0.0, index=score.columns); cur[picks] = 1.0 / top_n
        w.loc[d] = cur.values
    gross = (w.shift(1).fillna(0.0) * wide_ret.reindex(w.index)).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1) * 0.5
    return (gross - turn.shift(1).fillna(0.0) * cost_bps / 1e4).fillna(0.0)


def stats(daily, eqsector):
    nav = (1 + daily).cumprod(); r = daily
    dd = float(abs((nav / nav.cummax() - 1).min())); cagr = _ann(r)
    bw = {}
    for name, (a, z) in BULL.items():
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
        if m.sum() >= 10:
            bw[name] = round(float((1 + r[m]).prod() - (1 + eqsector.reindex(r.index)[m]).prod()), 4)
    return {"cagr": round(cagr, 4), "maxdd": round(dd, 4),
            "calmar": round(cagr / dd, 3) if dd > 1e-9 else None,
            "exc_eqsector": round(cagr - _ann(eqsector.reindex(r.index).dropna()), 4),
            "bull_2024_2025": bw.get("rally_2024H2_2025"), "bull_2025_2026": bw.get("rally_2025_2026"),
            "bull_covid": bw.get("covid_2020")}


def main():
    sp = pd.read_parquet(SECTOR_PANEL); sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[sp.trade_date >= "2018-01-01"]
    def wide(col): return sp.pivot_table(index="trade_date", columns="sector_level_1", values=col)
    wide_ret = wide(RET); eqsector = wide_ret.mean(axis=1)
    mom60, mom20 = wide("mom_60"), wide("mom_20")
    accel = mom20 - mom60                       # short-term turning up vs long-term weak
    amt_accel, breadth = wide("amt_accel"), wide("breadth_ma60")
    z_lag = _z(-mom60)                           # laggard
    z_stab = _z(accel).add(_z(amt_accel), fill_value=0).add(_z(breadth), fill_value=0) / 3.0

    rows = []
    # baselines
    for label, score in [("plain_laggard", z_lag),
                         ("plain_momentum", _z(mom60))]:
        for tn in (3, 5, 8):
            rows.append({"variant": label, "w_stab": "-", "top_n": tn,
                         **stats(basket_daily(score, wide_ret, top_n=tn, rebalance=20), eqsector)})
    # stabilized laggard
    for wst in (0.5, 1.0, 1.5, 2.0):
        score = z_lag + wst * z_stab
        for tn in (3, 5, 8):
            rows.append({"variant": "stabilized_laggard", "w_stab": wst, "top_n": tn,
                         **stats(basket_daily(score, wide_ret, top_n=tn, rebalance=20), eqsector)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "stabilized_sweep.csv", index=False)
    cols = ["variant", "w_stab", "top_n", "cagr", "maxdd", "calmar", "exc_eqsector",
            "bull_2024_2025", "bull_2025_2026", "bull_covid"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== 企稳-stabilized laggard vs plain (fast sector-basket, excess phantom-free) ===")
        print(res.sort_values("calmar", ascending=False)[cols].to_string(index=False))
    print(f"\n[write] {OUT_DIR/'stabilized_sweep.csv'}")
    print("[ref] plain-laggard top3 ~ +20%/43%DD/Calmar0.47; v8.9 = +17.3%/10.9%DD/Calmar1.58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
