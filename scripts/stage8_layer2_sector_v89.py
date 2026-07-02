#!/usr/bin/env python3
"""Stage 8 step 7 — LAYER-2: v8.9 ensemble stock selection INSIDE laggard sectors.

Question (the only one that matters): does using the laggard-sector signal as a
first-level UNIVERSE TILT, then picking stocks with the proven v8.9 ensemble
`composite_score` inside those sectors, beat PLAIN v8.9 on after-cost CAGR /
excess / bull capture — on the SAME OOS dates, SAME tradable universe, SAME
strict A-share engine?

Design (everything held constant except the sector filter):
  * plain v8.9 control : top-K by composite_score over eligible all-A, weight W,
                         rebalance R.
  * layer-2           : same, but the universe is first restricted to the top-N
                         laggard SW1 sectors (mom_20/mom_60/rs_20, reverse), then
                         per-sector top-`s` by score, capped at top-K.
  Same K, W, R, dates, engine. Only difference = the sector tilt.

Two-phase: a per-stock vectorized engine ranks the grid (this is REAL per-stock
returns net of cost — NOT the inflated sector-basket), then the top configs and
their matched plain controls are confirmed through `run_strict_backtest_v8`
(T+1 / cost / limit-up-down / suspension / ST).

PIT note: sector_map is a current snapshot (2026-05-31) ~3wk after the window
end. Survivorship CANCELS in the head-to-head (both sides use the same score
panel universe); residual risk = SW1 reclassification over <21mo (rare). A
SW1-vs-SW2 sensitivity check is emitted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

SCORE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
FEAT = "runtime/stage8_sector_rotation/stock_features.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
SCORE_COL = "composite_score"
BULL = {"rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
        "rally_2025_2026": ("2025-01-01", "2026-05-07")}


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


# --------------------------- book construction -----------------------------
def _weights_from_pool(pool: pd.DataFrame, weighting: str) -> pd.Series:
    """pool has columns: symbol, sector, score, vol. Return symbol->weight (sum=1)."""
    if pool.empty:
        return pd.Series(dtype=float)
    p = pool.set_index("symbol")
    if weighting == "equal_stock":
        w = pd.Series(1.0, index=p.index)
    elif weighting == "equal_sector":
        secw = 1.0 / p["sector"].nunique()
        w = p.groupby("sector")["score"].transform(lambda s: secw / len(s))
        w = pd.Series(w.values, index=p.index)
    elif weighting == "score_weighted":
        s = p["score"]
        w = (s - s.min() + 1e-6)
    elif weighting == "vol_capped":
        iv = 1.0 / (p["vol"].abs() + 1e-6)
        w = iv.clip(upper=iv.quantile(0.9))
    else:
        raise ValueError(weighting)
    return w / w.sum()


def build_book(stock_day: dict, sector_day: dict, *, rebal_dates, eval_dates,
               topk, weighting, sector_signal=None, n_sectors=0, per_sector=0,
               reverse=True) -> pd.DataFrame:
    """Unified builder. sector_signal=None -> plain v8.9 (all-A); else layer-2."""
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=["score"])
        if sd.empty:
            continue
        if sector_signal is not None:
            secf = sector_day.get(d)
            if secf is None or secf.empty:
                continue
            sf = secf.dropna(subset=[sector_signal]).sort_values(sector_signal, ascending=reverse)
            picks_sec = list(sf["sector_level_1"].head(n_sectors))
            sd = sd[sd["sector"].isin(picks_sec)]
            if sd.empty:
                continue
            sd = sd.sort_values("score", ascending=False)
            sd = sd.groupby("sector", group_keys=False).head(per_sector)
        sd = sd.sort_values("score", ascending=False).head(topk)
        w = _weights_from_pool(sd[["symbol", "sector", "score", "vol"]], weighting)
        if not w.empty:
            rows[d] = w.to_dict()
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([d for d in eval_dates if d >= tw.index.min()])
    tw = tw.reindex(full).ffill().fillna(0.0)
    tw.index.name = "trade_date"
    return tw


def fast_nav(tw, ret_mat, *, cost_bps=18.0, delay=1):
    cols = tw.columns.intersection(ret_mat.columns)
    tw = tw[cols].reindex(ret_mat.index).fillna(0.0)
    R = ret_mat[cols].reindex(tw.index)
    gross = (tw.shift(delay).fillna(0.0) * R).sum(axis=1)
    turn = (tw - tw.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(delay).fillna(0.0) * (cost_bps / 1e4)
    return (1 + (gross - cost).fillna(0.0)).cumprod()


# ------------------------------ evaluation ---------------------------------
def stats(nav, benches: dict, plain_daily: pd.Series | None):
    r = nav.pct_change().dropna()
    peak = nav.cummax(); dd = float(abs((nav / peak - 1).min())); cagr = _ann(r)
    out = {"cagr": round(cagr, 4), "maxdd": round(dd, 4),
           "calmar": round(cagr / dd, 3) if dd > 1e-9 else None}
    for name, b in benches.items():
        bi = b.reindex(r.index).dropna()
        out[f"exc_{name}"] = round(cagr - _ann(bi), 4)
    if plain_daily is not None:
        idx = r.index.intersection(plain_daily.index)
        out["exc_plain_v89"] = round(_ann(r.reindex(idx)) - _ann(plain_daily.reindex(idx)), 4)
        # positive-excess window ratio: monthly buckets where layer2 > plain
        rl = (1 + r.reindex(idx)).resample("ME").prod() - 1
        rp = (1 + plain_daily.reindex(idx)).resample("ME").prod() - 1
        out["pos_excess_month_ratio"] = round(float((rl > rp).mean()), 3)
    for name, (a, z) in BULL.items():
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
        if m.sum() < 10:
            continue
        base = benches["all_a_eqw"].reindex(r.index)
        out[f"bull_{name}"] = round(float((1 + r[m]).prod() - (1 + base[m]).prod()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="mom_60,rs_20")
    ap.add_argument("--n-sectors", default="2,3,5,8")
    ap.add_argument("--per-sector", default="3,5,10,20")
    ap.add_argument("--topk", default="20,30,50,100")
    ap.add_argument("--rebalance", default="20,21")
    ap.add_argument("--weighting", default="equal_stock,equal_sector,score_weighted,vol_capped")
    ap.add_argument("--directions", default="laggard",
                    help="laggard (reverse) and/or leader (momentum) sector tilt")
    ap.add_argument("--confirm-topk", type=int, default=20)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] v8.9 score + panels ...")
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", SCORE_COL]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = sc.trade_date.min(), sc.trade_date.max()
    smap = pd.read_parquet(SECTOR)[["symbol", "sector_level_1"]].dropna().drop_duplicates("symbol")
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    feat = pd.read_parquet(FEAT, columns=["symbol", "trade_date", "vol60"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[(sp.trade_date >= start) & (sp.trade_date <= end)]

    eval_dates = sorted(win.trade_date.unique())
    print(f"  OOS {start.date()}..{end.date()} ({len(eval_dates)} days), score syms={sc.symbol.nunique()}")

    # merge flags + sector + vol into the score panel -> per-stock-day frame
    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = (sc.merge(smap, on="symbol", how="left")
            .merge(flags, on=["symbol", "trade_date"], how="left")
            .merge(feat, on=["symbol", "trade_date"], how="left"))
    df["bad"] = (df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1))
    df = df.rename(columns={SCORE_COL: "score", "sector_level_1": "sector", "vol60": "vol"})
    df["vol"] = df["vol"].fillna(df["vol"].median())
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    sector_day = {d: g for d, g in sp.groupby("trade_date")}

    # benchmarks
    ret_mat = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None).reindex(pd.DatetimeIndex(eval_dates))
    all_a = ret_mat.mean(axis=1).dropna()
    eqsector = sp.pivot_table(index="trade_date", columns="sector_level_1", values="ret_eqw").mean(axis=1).dropna()
    benches = {"all_a_eqw": all_a, "equal_sector": eqsector}

    def rebals(step):
        d = sorted(pd.DatetimeIndex(eval_dates).unique()); return list(d[::step])

    # plain v8.9 controls (per topk x weighting x rebalance) — built once, cached
    print("[plain] building v8.9 controls (fast) ...")
    topks = [int(x) for x in args.topk.split(",")]
    weightings = args.weighting.split(",")
    rbs = [int(x) for x in args.rebalance.split(",")]
    plain_daily = {}   # (topk,weighting,rb) -> daily returns (fast)
    plain_tw = {}
    for K, W, RB in itertools.product(topks, weightings, rbs):
        tw = build_book(stock_day, sector_day, rebal_dates=rebals(RB), eval_dates=eval_dates,
                        topk=K, weighting=W)
        if tw.empty:
            continue
        plain_tw[(K, W, RB)] = tw
        plain_daily[(K, W, RB)] = fast_nav(tw, ret_mat).pct_change().dropna()

    # ---- phase 1: fast rank the layer-2 grid ----
    sigs = args.signals.split(",")
    nss = [int(x) for x in args.n_sectors.split(",")]
    pss = [int(x) for x in args.per_sector.split(",")]
    dirs = args.directions.split(",")
    configs = list(itertools.product(sigs, nss, pss, topks, rbs, weightings, dirs))
    print(f"[grid] {len(configs)} layer-2 configs (phase 1 fast)")
    rows = []
    for i, (sig, ns, ps, K, RB, W, DR) in enumerate(configs):
        if ps * ns < 1:
            continue
        tw = build_book(stock_day, sector_day, rebal_dates=rebals(RB), eval_dates=eval_dates,
                        topk=K, weighting=W, sector_signal=sig, n_sectors=ns, per_sector=ps,
                        reverse=(DR == "laggard"))
        if tw.empty:
            continue
        nav = fast_nav(tw, ret_mat)
        st = stats(nav, benches, plain_daily.get((K, W, RB)))
        rows.append({"signal": sig, "dir": DR, "n_sectors": ns, "per_sector": ps, "topk": K,
                     "rebalance": RB, "weighting": W, **st})
        if (i + 1) % 100 == 0:
            print(f"  ...phase1 {i+1}/{len(configs)}")
    fast_df = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    fast_df.to_csv(OUT_DIR / "layer2_fast_leaderboard.csv", index=False)
    print(f"\n[phase1] {len(fast_df)} ranked -> layer2_fast_leaderboard.csv")
    fcols = ["signal", "dir", "n_sectors", "per_sector", "topk", "rebalance", "weighting",
             "cagr", "maxdd", "calmar", "exc_plain_v89", "bull_rally_2024H2_2025", "bull_rally_2025_2026"]
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== TOP 15 layer-2 by fast CAGR ===")
        print(fast_df[fcols].head(15).to_string(index=False))
        print("\n=== TOP 10 layer-2 by excess vs plain v8.9 ===")
        print(fast_df.sort_values("exc_plain_v89", ascending=False)[fcols].head(10).to_string(index=False))

    # ---- phase 2: strict confirm top-K layer-2 + their matched plain controls ----
    topc = fast_df.head(args.confirm_topk)
    print(f"\n[phase2] strict-confirming top {len(topc)} layer-2 + matched plain controls ...")
    smap_full = pd.read_parquet(SECTOR)
    strict_rows = []
    seen_plain = {}
    for _, r in topc.iterrows():
        sig, DR, ns, ps, K, RB, W = r["signal"], r["dir"], int(r["n_sectors"]), int(r["per_sector"]), int(r["topk"]), int(r["rebalance"]), r["weighting"]
        tw2 = build_book(stock_day, sector_day, rebal_dates=rebals(RB), eval_dates=eval_dates,
                         topk=K, weighting=W, sector_signal=sig, n_sectors=ns, per_sector=ps,
                         reverse=(DR == "laggard"))
        a2 = run_strict_backtest_v8(tw2, win, sector_map=smap_full)
        nav2 = a2.nav
        # matched plain control (strict) — cache by (K,W,RB)
        key = (K, W, RB)
        if key not in seen_plain:
            ap_ = run_strict_backtest_v8(plain_tw[key], win, sector_map=smap_full)
            seen_plain[key] = ap_.nav
        navp = seen_plain[key]
        st2 = stats(nav2, benches, navp.pct_change().dropna())
        m2 = a2.metrics
        strict_rows.append({
            "signal": sig, "dir": DR, "n_sectors": ns, "per_sector": ps, "topk": K, "rebalance": RB, "weighting": W,
            "l2_cagr": round(m2.annualized_return, 4), "l2_maxdd": round(m2.max_drawdown, 4),
            "l2_calmar": round(m2.calmar, 4) if m2.calmar else None, "l2_turnover": round(m2.turnover, 4),
            "exc_plain_v89": st2.get("exc_plain_v89"),
            "pos_excess_month_ratio": st2.get("pos_excess_month_ratio"),
            "bull_2024_2025": st2.get("bull_rally_2024H2_2025"),
            "bull_2025_2026": st2.get("bull_rally_2025_2026"),
        })
        print(f"  L2 {sig}/{DR} ns{ns} ps{ps} K{K} rb{RB} {W}: CAGR={m2.annualized_return:+.1%} "
              f"DD={m2.max_drawdown:.1%} Calmar={strict_rows[-1]['l2_calmar']} "
              f"excV89={st2.get('exc_plain_v89'):+.1%} bull25={st2.get('bull_rally_2025_2026')}")

    sr = pd.DataFrame(strict_rows).sort_values("l2_cagr", ascending=False)
    sr.to_csv(OUT_DIR / "layer2_strict_confirm.csv", index=False)
    # plain control strict metrics for reference
    plain_ref = {}
    for key, nav in seen_plain.items():
        m = run_strict_backtest_v8(plain_tw[key], win, sector_map=smap_full).metrics if False else None
        r = nav.pct_change().dropna(); peak = nav.cummax(); dd = float(abs((nav/peak-1).min())); c = _ann(r)
        plain_ref["_".join(map(str, key))] = {"cagr": round(c,4), "maxdd": round(dd,4),
                                              "calmar": round(c/dd,3) if dd>1e-9 else None}
    (OUT_DIR / "layer2_strict_confirm.json").write_text(
        json.dumps({"layer2": strict_rows, "plain_controls": plain_ref},
                   ensure_ascii=False, indent=2, default=str))
    print("\n=== STRICT layer-2 (vs matched plain v8.9 control) ===")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(sr.to_string(index=False))
    print("\n[plain controls, strict CAGR/DD/Calmar]:")
    for k, v in plain_ref.items():
        print(f"   {k}: {v}")
    print("\n[ref] plain v8.9 canonical ~ +17.3% / 10.9% DD / Calmar 1.58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
