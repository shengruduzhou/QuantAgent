#!/usr/bin/env python3
"""Stage 8 step 2 — sector-rotation portfolio search through the trusted engine.

Builds per-stock target weights for a grid of sector-rotation configs and runs
each through ``run_strict_backtest_v8`` (T+1, real A-share costs/tradability),
so after-cost CAGR / Calmar / turnover are honest and comparable to the v8.9
stock baseline (+17.3% CAGR / 10.9% MaxDD / Calmar 1.58).

Outputs, per config:
  after-cost CAGR, MaxDD, Calmar, turnover, Sharpe
  excess vs all-A eqw / equal-sector / CSI300 / CSI500 / CSI1000
  per-regime (bull/sideways/bear) excess ratio
  bull-window capture (2020 post-COVID, 2024-08→2025-08, 2025→2026)

Stock features (momentum/liquidity/vol per date) are precomputed once and
cached so the grid is fast.
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
from quantagent.strategy.sector_rotation_book import (  # noqa: E402
    SECTOR_COL, build_rotation_book,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244

BULL_WINDOWS = {
    "covid_2020": ("2020-03-23", "2021-02-10"),
    "rally_2024H2_2025": ("2024-08-28", "2025-08-31"),
    "rally_2025_2026": ("2025-01-01", "2026-05-18"),
}


# ----------------------------- stock features ------------------------------
def build_stock_features(panel: pd.DataFrame, smap: pd.DataFrame) -> pd.DataFrame:
    sm = smap[["symbol", SECTOR_COL]].dropna(subset=[SECTOR_COL]).drop_duplicates("symbol")
    keep = ["symbol", "trade_date", "close", "amount"]
    opt = [c for c in ("is_suspended", "is_st", "is_limit_up", "is_limit_down") if c in panel.columns]
    df = panel[keep + opt].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.merge(sm, on="symbol", how="inner").sort_values(["symbol", "trade_date"])
    g = df.groupby("symbol", sort=False)
    ret = g["close"].pct_change(fill_method=None)
    lr = np.log1p(ret.clip(lower=-0.99))
    df["mom20"] = np.expm1(lr.groupby(df["symbol"]).transform(lambda s: s.rolling(20, min_periods=10).sum()))
    df["mom60"] = np.expm1(lr.groupby(df["symbol"]).transform(lambda s: s.rolling(60, min_periods=30).sum()))
    df["vol60"] = ret.groupby(df["symbol"]).transform(lambda s: s.rolling(60, min_periods=30).std())
    df["amt20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    return df.dropna(subset=["amt20"])


# ------------------------------ benchmarks ---------------------------------
def fast_backtest(tw: pd.DataFrame, ret_mat: pd.DataFrame, *, cost_bps: float = 18.0,
                  delay: int = 1) -> pd.Series:
    """Vectorized long-only NAV for RANKING configs (no broker / tradability).

    Position weights `tw` (date x symbol, ffilled) established at close t are
    held; portfolio return on day t+1 = sum_i w_{i,t} * r_{i,t+1}. A `delay` of
    1 trading day approximates T+1 fill. Costs charged on rebalance turnover.
    Used only to screen the grid; survivors are re-run through the strict
    A-share engine for honest after-cost numbers.
    """
    cols = tw.columns.intersection(ret_mat.columns)
    tw = tw[cols].reindex(ret_mat.index).fillna(0.0)
    R = ret_mat[cols].reindex(tw.index)
    w_eff = tw.shift(delay).fillna(0.0)
    gross = (w_eff * R).sum(axis=1)
    turn = (tw - tw.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(delay).fillna(0.0) * (cost_bps / 1e4)
    net = (gross - cost).fillna(0.0)
    return (1.0 + net).cumprod()


def all_a_eqw_daily(panel: pd.DataFrame, dates) -> pd.Series:
    px = panel[panel["trade_date"].isin(dates)].pivot_table(
        index="trade_date", columns="symbol", values="close")
    return px.pct_change(fill_method=None).mean(axis=1).dropna()


def equal_sector_daily(sector_panel: pd.DataFrame, dates) -> pd.Series:
    sp = sector_panel[sector_panel["trade_date"].isin(dates)]
    return sp.groupby("trade_date")["ret_eqw"].mean().dropna()


def index_daily(label: str, dates) -> pd.Series:
    idx = pd.read_parquet(INDEX)
    idx = idx[idx["label"] == label].copy()
    idx["observation_date"] = pd.to_datetime(idx["observation_date"])
    s = idx.set_index("observation_date")["close"].sort_index().pct_change()
    return s.reindex(pd.DatetimeIndex(sorted(dates))).dropna()


def _ann(daily: pd.Series) -> float:
    n = len(daily)
    return float((1 + daily).prod() ** (ANN / n) - 1) if n else float("nan")


def _regime_label(bench_daily: pd.Series) -> pd.Series:
    cum = (1 + bench_daily).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=bench_daily.index)


def analyze(nav: pd.Series, benches: dict[str, pd.Series]) -> dict:
    strat = nav.pct_change().dropna()
    out: dict[str, object] = {}
    base = benches["all_a_eqw"]
    idx = strat.index.intersection(base.index)
    s = strat.reindex(idx)
    out["strat_cagr"] = round(_ann(s), 4)
    # excess vs each bench
    exc = {}
    for name, b in benches.items():
        bi = b.reindex(idx)
        exc[name] = round(_ann(s) - _ann(bi.dropna()), 4)
    out["excess_ann"] = exc
    # regime breakdown vs all-A
    regime = _regime_label(base).reindex(idx)
    reg_rows = {}
    for rg in ["bull", "sideways", "bear"]:
        mask = regime == rg
        n = int(mask.sum())
        if n < 5:
            continue
        ss, bb = s[mask], base.reindex(idx)[mask]
        # positive-excess ratio = share of days strat beats bench
        per_day_exc = (ss - bb)
        reg_rows[rg] = {
            "days": n,
            "strat_ann": round(_ann(ss), 4),
            "bench_ann": round(_ann(bb), 4),
            "excess_ann": round(_ann(ss) - _ann(bb), 4),
            "positive_excess_ratio": round(float((per_day_exc > 0).mean()), 3),
        }
    out["regime"] = reg_rows
    # bull windows
    bw = {}
    for name, (a, z) in BULL_WINDOWS.items():
        m = (s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(z))
        if m.sum() < 10:
            continue
        ss = s[m]
        bb = base.reindex(idx)[m]
        bw[name] = {
            "strat_ret": round(float((1 + ss).prod() - 1), 4),
            "all_a_ret": round(float((1 + bb).prod() - 1), 4),
            "excess": round(float((1 + ss).prod() - (1 + bb).prod()), 4),
        }
    out["bull_windows"] = bw
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-05-18")
    ap.add_argument("--signals", default="rs_60,mom_60,rmom_60,breadth_ma60")
    ap.add_argument("--top-n", default="1,2,3,5,8")
    ap.add_argument("--rebalance", default="5,20,21")
    ap.add_argument("--sector-weighting", default="equal,momentum,volparity")
    ap.add_argument("--within", default="top_liquid,top_momentum")
    ap.add_argument("--n-within", type=int, default=5)
    ap.add_argument("--confirm-topk", type=int, default=15,
                    help="re-run top-K fast configs through the strict A-share engine")
    ap.add_argument("--quick", action="store_true", help="small grid for smoke")
    ap.add_argument("--limit", type=int, default=0, help="cap number of configs")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] panels ...")
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    smap = pd.read_parquet(SECTOR)
    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)]
    eval_dates = sorted(win["trade_date"].unique())
    print(f"  eval {start.date()}..{end.date()} ({len(eval_dates)} trading days)")

    feat_cache = OUT_DIR / "stock_features.parquet"
    if feat_cache.exists():
        print(f"[load] cached stock features {feat_cache}")
        feat = pd.read_parquet(feat_cache)
        feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    else:
        print("[build] stock features (one-time) ...")
        feat = build_stock_features(panel, smap)
        feat.to_parquet(feat_cache, index=False)
    feat = feat[(feat.trade_date >= start) & (feat.trade_date <= end)]
    sp_win = sp[(sp.trade_date >= start) & (sp.trade_date <= end)]

    # benchmarks
    print("[bench] building benchmarks ...")
    benches = {
        "all_a_eqw": all_a_eqw_daily(win, eval_dates),
        "equal_sector": equal_sector_daily(sp_win, eval_dates),
        "csi300": index_daily("csi300", eval_dates),
        "csi500": index_daily("csi500", eval_dates),
        "csi1000": index_daily("csi1000", eval_dates),
    }
    for k, v in benches.items():
        print(f"   {k:<14} ann={_ann(v.dropna()):+.2%}  n={len(v)}")

    # grid
    if args.quick:
        signals, top_ns, rebals = ["rs_60"], [2, 3], [20]
        secw, within = ["equal"], ["top_liquid"]
    else:
        signals = args.signals.split(",")
        top_ns = [int(x) for x in args.top_n.split(",")]
        rebals = [int(x) for x in args.rebalance.split(",")]
        secw = args.sector_weighting.split(",")
        within = args.within.split(",")

    configs = list(itertools.product(signals, top_ns, rebals, secw, within))
    if args.limit:
        configs = configs[: args.limit]
    print(f"[grid] {len(configs)} configs (phase 1 = fast rank)")

    # forward return matrix for the fast engine (built once)
    ret_mat = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None)
    ret_mat = ret_mat.reindex(pd.DatetimeIndex(eval_dates))

    # ---- phase 1: fast vectorized rank over the whole grid ----
    fast_rows = []
    for i, (sig, tn, rb, sw, wi) in enumerate(configs):
        tw = build_rotation_book(
            sp_win, feat, signal=sig, top_n=tn, rebalance_days=rb,
            sector_weighting=sw, within_sector=wi, n_within=args.n_within,
            eval_dates=eval_dates,
        )
        if tw.empty:
            continue
        nav = fast_backtest(tw, ret_mat)
        ana = analyze(nav, benches)
        s = nav.pct_change().dropna()
        peak = nav.cummax()
        dd = float(abs((nav / peak - 1.0).min()))
        cagr = round(_ann(s), 4)
        turn = float((tw - tw.shift(1)).abs().sum(axis=1).mean() * 0.5)
        fast_rows.append({
            "signal": sig, "top_n": tn, "rebalance": rb,
            "sector_weighting": sw, "within": wi,
            "fast_cagr": cagr, "fast_maxdd": round(dd, 4),
            "fast_calmar": round(cagr / dd, 3) if dd > 1e-9 else None,
            "fast_turn_daily": round(turn, 4),
            "exc_all_a": ana["excess_ann"]["all_a_eqw"],
            "exc_eqsector": ana["excess_ann"]["equal_sector"],
            "exc_csi300": ana["excess_ann"]["csi300"],
            "bull_2024_2025": ana["bull_windows"].get("rally_2024H2_2025", {}).get("excess"),
            "bull_2025_2026": ana["bull_windows"].get("rally_2025_2026", {}).get("excess"),
            "_full": ana,
        })
        if (i + 1) % 50 == 0:
            print(f"  ...phase1 {i+1}/{len(configs)}")

    fast_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_full"} for r in fast_rows])
    fast_df = fast_df.sort_values("fast_cagr", ascending=False)
    fast_df.to_csv(OUT_DIR / "fast_leaderboard.csv", index=False)
    print(f"\n[phase1] {len(fast_df)} configs ranked -> {OUT_DIR/'fast_leaderboard.csv'}")
    print("=== TOP 15 (fast engine, by CAGR) ===")
    fcols = ["signal", "top_n", "rebalance", "sector_weighting", "within",
             "fast_cagr", "fast_maxdd", "fast_calmar", "exc_eqsector", "exc_csi300", "bull_2025_2026"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(fast_df[fcols].head(15).to_string(index=False))

    # ---- phase 2: confirm top-K through the trusted strict A-share engine ----
    topk = int(args.confirm_topk)
    top_configs = fast_df.head(topk)[["signal", "top_n", "rebalance", "sector_weighting", "within"]].values.tolist()
    print(f"\n[phase2] confirming top {len(top_configs)} through strict A-share engine ...")
    results = []
    for j, (sig, tn, rb, sw, wi) in enumerate(top_configs):
        tw = build_rotation_book(
            sp_win, feat, signal=sig, top_n=int(tn), rebalance_days=int(rb),
            sector_weighting=sw, within_sector=wi, n_within=args.n_within,
            eval_dates=eval_dates,
        )
        arts = run_strict_backtest_v8(tw, win, sector_map=smap)
        m = arts.metrics
        ana = analyze(arts.nav, benches)
        row = {
            "signal": sig, "top_n": int(tn), "rebalance": int(rb),
            "sector_weighting": sw, "within": wi,
            "cagr": round(m.annualized_return, 4),
            "maxdd": round(m.max_drawdown, 4),
            "calmar": round(m.calmar, 4) if m.calmar else None,
            "sharpe": round(m.sharpe, 4) if m.sharpe else None,
            "turnover": round(m.turnover, 4),
            "exc_all_a": ana["excess_ann"]["all_a_eqw"],
            "exc_eqsector": ana["excess_ann"]["equal_sector"],
            "exc_csi300": ana["excess_ann"]["csi300"],
            "exc_csi500": ana["excess_ann"]["csi500"],
            "bull_2024_2025": ana["bull_windows"].get("rally_2024H2_2025", {}).get("excess"),
            "bull_2025_2026": ana["bull_windows"].get("rally_2025_2026", {}).get("excess"),
            "_full": ana,
        }
        results.append(row)
        print(f"  [{j+1}/{len(top_configs)}] {sig} N={tn} rb={rb} {sw}/{wi}: "
              f"CAGR={row['cagr']:+.1%} DD={row['maxdd']:.1%} "
              f"Calmar={row['calmar']} excEqSec={row['exc_eqsector']:+.1%} "
              f"excCSI300={row['exc_csi300']:+.1%} bull25={row['bull_2025_2026']}")

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_full"} for r in results])
    res = res.sort_values("cagr", ascending=False)
    res.to_csv(OUT_DIR / "search_leaderboard.csv", index=False)
    (OUT_DIR / "search_full.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n[write] {OUT_DIR/'search_leaderboard.csv'}")
    print("\n=== STRICT-CONFIRMED, by after-cost CAGR (vs v8.9 baseline +17.3%/DD10.9%/Calmar1.58) ===")
    cols = ["signal", "top_n", "rebalance", "sector_weighting", "within",
            "cagr", "maxdd", "calmar", "turnover", "exc_eqsector", "exc_csi300", "bull_2025_2026"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res[cols].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
