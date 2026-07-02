"""Stage 10.4 — forward paper-trade tracker with beta/alpha decomposition.

Builds a CONTINUOUS daily-rebalanced strategy NAV from the frozen daily concept
portfolios + PIT snapshots, plus four benchmarks (all-A eqw, same-tradable-
universe eqw, selected-concept stock eqw, concept-index eqw), and decomposes the
strategy into beta / Jensen alpha vs each — so from the first forward days we can
tell whether return is market beta or real alpha.

Forward-only: day t return uses day-(t-1)'s frozen portfolio marked with day-t
prices. Costs charged on daily rebalance turnover. Reuses
`quantagent.backtest.beta_decomposition` for the panel. Pure w.r.t. a snapshot
dir so it is unit-testable on synthetic snapshots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest import beta_decomposition as bd

COST_BPS = 18.0


def _spot(snaps: Path, date: str) -> pd.Series:
    sp = pd.read_parquet(snaps / date / "spot_all.parquet")
    sp["代码"] = sp["代码"].astype(str).str.zfill(6)
    px = pd.to_numeric(sp.set_index("代码")["最新价"].astype(str).replace({"-": np.nan}), errors="coerce")
    return px.dropna()


def _names(snaps: Path, date: str) -> pd.Series:
    sp = pd.read_parquet(snaps / date / "spot_all.parquet")
    sp["代码"] = sp["代码"].astype(str).str.zfill(6)
    return sp.set_index("代码")["名称"].astype(str)


def _same_universe(snaps: Path, date: str) -> list[str]:
    """Tradable universe proxy: exclude ST / *ST / price<2 (v8.9-style eligibility)."""
    nm = _names(snaps, date); px = _spot(snaps, date)
    codes = [c for c in px.index if c in nm.index and "ST" not in nm[c] and px[c] >= 2.0]
    return codes


def _concept_members(snaps: Path, date: str, concepts: list[str]) -> list[str]:
    consdir = snaps / date / "cons"
    codes: set[str] = set()
    for c in concepts:
        f = consdir / f"{c.replace('/', '_')}.parquet"
        if f.exists():
            codes |= set(pd.read_parquet(f)["代码"].astype(str).str.zfill(6))
    return sorted(codes)


def _board_index_ret(concepts: list[str], d_prev: str, d: str,
                     hist_dir: Path) -> float:
    """Equal-weight concept-INDEX return between two snapshot dates (板块指数)."""
    rets = []
    dp, dd = pd.Timestamp(d_prev), pd.Timestamp(d)
    for c in concepts:
        f = hist_dir / f"{c.replace('/', '_')}.parquet"
        if not f.exists():
            continue
        h = pd.read_parquet(f); h["日期"] = pd.to_datetime(h["日期"])
        h = h.set_index("日期")["收盘"].astype(float).sort_index()
        p0 = h.asof(dp); p1 = h.asof(dd)
        if p0 and p1 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    return float(np.mean(rets)) if rets else np.nan


def build_track(snaps: Path, pt: Path, hist_dir: Path) -> dict:
    dates = sorted(f.name.split("_")[1].split(".")[0]
                   for f in pt.glob("portfolio_*.csv"))
    if len(dates) < 2:
        return {"status": "awaiting_forward_days", "snapshot_days": len(dates),
                "need": ">=2 snapshot days for the first forward return"}

    strat, all_a, same_u, sel_c, cidx, turn, gross = {}, {}, {}, {}, {}, {}, {}
    for i in range(1, len(dates)):
        d0, d = dates[i - 1], dates[i]
        px0, pxd = _spot(snaps, d0), _spot(snaps, d)
        pf0 = pd.read_csv(pt / f"portfolio_{d0}.csv", dtype={"code": str})
        pf0["code"] = pf0["code"].str.zfill(6)
        codes, w = pf0["code"].values, pf0["weight"].values
        r = (pxd.reindex(codes) / px0.reindex(codes) - 1.0).values
        g = float(np.nansum(w))
        # daily rebalance cost vs today's target portfolio
        pfd = pd.read_csv(pt / f"portfolio_{d}.csv", dtype={"code": str}); pfd["code"] = pfd["code"].str.zfill(6)
        merged = pd.concat([pf0.set_index("code")["weight"].rename("w0"),
                            pfd.set_index("code")["weight"].rename("w1")], axis=1).fillna(0.0)
        tv = float((merged["w1"] - merged["w0"]).abs().sum() * 0.5)
        strat[d] = float(np.nansum(w * r)) - tv * COST_BPS / 1e4
        turn[d] = tv; gross[d] = g
        # benchmarks
        all_codes = px0.index.intersection(pxd.index)
        all_a[d] = float((pxd.reindex(all_codes) / px0.reindex(all_codes) - 1.0).mean())
        su = [c for c in _same_universe(snaps, d0) if c in pxd.index]
        same_u[d] = float((pxd.reindex(su) / px0.reindex(su) - 1.0).mean()) if su else np.nan
        sc = _concept_members(snaps, d0, list(pf0["concept"].unique()))
        sc = [c for c in sc if c in pxd.index and c in px0.index]
        sel_c[d] = float((pxd.reindex(sc) / px0.reindex(sc) - 1.0).mean()) if sc else np.nan
        cidx[d] = _board_index_ret(list(pf0["concept"].unique()), d0, d, hist_dir)

    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in strat])
    s = pd.Series(list(strat.values()), index=idx)
    benches = {
        "all_a": pd.Series(list(all_a.values()), index=idx),
        "same_universe": pd.Series(list(same_u.values()), index=idx),
        "selected_concept": pd.Series(list(sel_c.values()), index=idx),
        "concept_index": pd.Series(list(cidx.values()), index=idx),
    }
    nav = (1 + s).cumprod()
    panel = bd.full_panel(s, nav, benches, turnover=float(np.nanmean(list(turn.values()))), primary="all_a")

    # exposure + attribution from latest portfolio
    last = dates[-1]
    pf_last = pd.read_csv(pt / f"portfolio_{last}.csv", dtype={"code": str})
    concept_breakdown = pf_last.groupby("concept")["weight"].sum().round(4).to_dict()
    # top holding contribution (weight * cumulative fwd return of that name)
    px_first, px_last = _spot(snaps, dates[0]), _spot(snaps, last)
    pf0 = pd.read_csv(pt / f"portfolio_{dates[0]}.csv", dtype={"code": str}); pf0["code"] = pf0["code"].str.zfill(6)
    pf0["fwd_ret"] = (px_last.reindex(pf0["code"].values) / px_first.reindex(pf0["code"].values) - 1.0).values
    pf0["contrib"] = pf0["weight"] * pf0["fwd_ret"]
    top = pf0.sort_values("contrib", ascending=False).head(3)[["name", "concept", "weight", "fwd_ret", "contrib"]]

    daily = pd.DataFrame({"strat_ret": s, **{f"{k}_ret": v for k, v in benches.items()}}, index=idx)
    daily["gross"] = pd.Series(list(gross.values()), index=idx)
    daily["turnover"] = pd.Series(list(turn.values()), index=idx)
    daily["strat_cum"] = nav.values
    for k, v in benches.items():
        daily[f"{k}_cum"] = (1 + v.fillna(0)).cumprod().values

    return {"status": "ok", "days": len(s), "panel": panel,
            "gross_exposure": round(float(np.nanmean(list(gross.values()))), 3),
            "concept_breakdown": concept_breakdown,
            "top_contribution": top.round(4).to_dict("records"),
            "daily": daily}
