#!/usr/bin/env python3
"""Stage 8 step 4 — is the sector-reversal edge real out-of-sample?

The in-sample diagnostic said A-share SW1 sectors mean-revert (momentum IC
t<=-6) and a laggard (low-momentum) top-N basket beats the equal-sector
benchmark by ~+9%/yr. This project's history is full of in-sample phantoms, so
before believing it we check temporal stability:

  A. rank-IC of the reversal signal PER CALENDAR YEAR (sign + magnitude);
  B. laggard-basket excess vs equal-sector PER SUB-PERIOD + per-regime
     positive-excess ratio (the user's regime-consistency bar);
  C. a true walk-forward: at each rebalance pick the laggard sectors using only
     past data (the signal itself is already trailing), and report the OOS NAV
     over the back half — no parameter is fit on the future.

If the excess is positive in most years/regimes and survives the back-half-only
walk-forward, the edge is structural (mean-reversion is a known A-share trait),
not curve-fit.
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
KEY_SIGNALS = ["rs_20", "mom_20", "mom_60", "breadth_ma60"]  # reverse these (hold LOW)


def _ann(daily: pd.Series) -> float:
    n = len(daily)
    return float((1 + daily).prod() ** (ANN / n) - 1) if n else float("nan")


def forward_ret(wide_ret: pd.DataFrame, h: int) -> pd.DataFrame:
    logr = np.log1p(wide_ret.clip(lower=-0.99))
    return np.expm1(logr.rolling(h).sum().shift(-h))


def yearly_ic(sw: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    out = {}
    for yr, idx in sw.groupby(sw.index.year).groups.items():
        ics = []
        for d in idx:
            if d not in fwd.index:
                continue
            s, f = sw.loc[d], fwd.loc[d]
            m = s.notna() & f.notna()
            if m.sum() < 8:
                continue
            rho = spearmanr(s[m], f[m]).statistic
            if np.isfinite(rho):
                ics.append(rho)
        if len(ics) >= 10:
            out[int(yr)] = round(float(np.mean(ics)), 4)
    return out


def laggard_nav(sw: pd.DataFrame, wide_ret: pd.DataFrame, *, top_n: int,
                rebalance: int, cost_bps: float = 18.0) -> pd.Series:
    dates = list(sw.index)
    rebal = set(dates[::rebalance])
    w = pd.DataFrame(0.0, index=dates, columns=sw.columns)
    cur = pd.Series(0.0, index=sw.columns)
    for d in dates:
        if d in rebal:
            s = sw.loc[d].dropna()
            if len(s) >= top_n:
                picks = s.sort_values(ascending=True).head(top_n).index  # LOW = laggards
                cur = pd.Series(0.0, index=sw.columns)
                cur[picks] = 1.0 / top_n
        w.loc[d] = cur.values
    w_eff = w.shift(1).fillna(0.0)
    gross = (w_eff * wide_ret.reindex(w.index)).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(1).fillna(0.0) * (cost_bps / 1e4)
    return (1 + (gross - cost).fillna(0.0)).cumprod()


def regime_label(bench_daily: pd.Series) -> pd.Series:
    cum = (1 + bench_daily).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=bench_daily.index)


def main() -> int:
    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[sp.trade_date >= "2018-01-01"]
    wide_ret = sp.pivot_table(index="trade_date", columns="sector_level_1", values=RET)
    dates = wide_ret.index
    eqsector = wide_ret.mean(axis=1)
    fwd20 = forward_ret(wide_ret, 20)
    report = {}

    # ---- A. yearly IC ----
    print("=== A. reversal signal rank-IC per year (negative = mean-reversion edge) ===")
    print(f"{'signal':<14}" + "".join(f"{y:>8}" for y in range(2018, 2027)))
    for sig in KEY_SIGNALS:
        sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=sig).reindex(dates)
        yic = yearly_ic(sw, fwd20)
        report.setdefault("yearly_ic", {})[sig] = yic
        print(f"{sig:<14}" + "".join(f"{yic.get(y, float('nan')):>+8.3f}" for y in range(2018, 2027)))

    # ---- B. laggard-basket excess per sub-period + regime positive-excess ratio ----
    print("\n=== B. laggard top-3 (rb=20) excess vs equal-sector, by sub-period ===")
    subperiods = [("2018-2019", "2018-01-01", "2019-12-31"),
                  ("2020-2021", "2020-01-01", "2021-12-31"),
                  ("2022-2023", "2022-01-01", "2023-12-31"),
                  ("2024-2026", "2024-01-01", "2026-05-18")]
    reg = regime_label(eqsector)
    print(f"{'signal':<14}" + "".join(f"{n:>12}" for n, _, _ in subperiods) + f"{'pos-exc%':>10}")
    for sig in KEY_SIGNALS:
        sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=sig).reindex(dates)
        nav = laggard_nav(sw, wide_ret, top_n=3, rebalance=20)
        r = nav.pct_change().dropna()
        cells = []
        for name, a, z in subperiods:
            m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
            rr, bb = r[m], eqsector.reindex(r.index)[m]
            cells.append(_ann(rr) - _ann(bb))
        # daily positive-excess ratio
        exc_daily = (r - eqsector.reindex(r.index)).dropna()
        posrat = float((exc_daily > 0).mean())
        report.setdefault("subperiod_excess", {})[sig] = {
            n: round(c, 4) for (n, _, _), c in zip(subperiods, cells)}
        report["subperiod_excess"][sig]["pos_excess_ratio"] = round(posrat, 3)
        # per-regime
        regrows = {}
        for rg in ["bull", "sideways", "bear"]:
            mm = reg.reindex(r.index) == rg
            if mm.sum() < 10:
                continue
            rr, bb = r[mm], eqsector.reindex(r.index)[mm]
            ed = (rr - bb)
            regrows[rg] = {"excess_ann": round(_ann(rr) - _ann(bb), 4),
                           "pos_ratio": round(float((ed > 0).mean()), 3)}
        report["subperiod_excess"][sig]["regime"] = regrows
        print(f"{sig:<14}" + "".join(f"{c:>+12.3f}" for c in cells) + f"{posrat:>10.2f}")
        print(f"{'  regime:':<14}" + "  ".join(f"{rg}={d['excess_ann']:+.2f}/{d['pos_ratio']:.2f}"
                                               for rg, d in regrows.items()))

    # ---- C. back-half-only walk-forward (OOS) ----
    print("\n=== C. OOS: laggard top-3 over BACK HALF only (2022-06 .. 2026-05) ===")
    split = pd.Timestamp("2022-06-01")
    for sig in KEY_SIGNALS:
        sw = sp.pivot_table(index="trade_date", columns="sector_level_1", values=sig).reindex(dates)
        nav = laggard_nav(sw, wide_ret, top_n=3, rebalance=20)
        r = nav.pct_change().dropna()
        m = r.index >= split
        rr, bb = r[m], eqsector.reindex(r.index)[m]
        peak = (1 + rr).cumprod().cummax()
        dd = float(abs(((1 + rr).cumprod() / peak - 1).min()))
        cagr = _ann(rr)
        report.setdefault("oos_back_half", {})[sig] = {
            "cagr": round(cagr, 4), "bench_ann": round(_ann(bb), 4),
            "excess": round(cagr - _ann(bb), 4), "maxdd": round(dd, 4),
            "calmar": round(cagr / dd, 3) if dd > 1e-9 else None}
        d = report["oos_back_half"][sig]
        print(f"   {sig:<14} CAGR={d['cagr']:+.1%}  bench={d['bench_ann']:+.1%}  "
              f"excess={d['excess']:+.1%}  DD={d['maxdd']:.1%}  Calmar={d['calmar']}")

    (OUT_DIR / "sector_reversal_oos.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[write] {OUT_DIR/'sector_reversal_oos.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
