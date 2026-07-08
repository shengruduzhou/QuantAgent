#!/usr/bin/env python3
"""H-020: build a PIT-safe valuation + fundamental feature block.

Reuses the existing PIT fundamentals panel
``silver/fundamentals/metrics_panel.parquet`` (announce_date + available_at,
quarterly eps_basic/ocfps which are YTD-CUMULATIVE, plus point-in-time bps/roe/
margins/growth/debt). Produces two artifacts:

  1. quarterly block  silver/valuation/val_fund_quarterly.parquet
     (symbol, period_end, available_at, eps_ttm, ocfps_ttm, bps, roe, ...)
     -- eps_ttm / ocfps_ttm are de-cumulated trailing-4-quarter sums.

  2. daily block       silver/valuation/val_fund_features.parquet   (--daily)
     as-of joined onto a keys frame (symbol, trade_date, close) with a strict
     backward merge_asof on available_at (same PIT pattern as
     enrich_panel_fundamentals.py), then price-based ratios + cross-sectional
     percentiles + composites. cheap == high for the *_yield / *_percentile.

No shares outstanding on disk -> PS / EV-EBITDA / market_cap / turnover_rate are
NOT built (would require fabrication). pe_ttm / pcf are NaN when the TTM
denominator <= 0 (negative-earnings multiples are meaningless).

Run:
  python3 scripts/build_valuation_fundamental_features.py --self-test
  python3 scripts/build_valuation_fundamental_features.py            # quarterly
  python3 scripts/build_valuation_fundamental_features.py --daily \
      --keys <training_dataset.parquet>                              # daily block
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
OUT_Q = REPO / "runtime/data/v7/silver/valuation/val_fund_quarterly.parquet"
OUT_DAILY = REPO / "runtime/data/v7/silver/valuation/val_fund_features.parquet"

# as-reported fundamentals carried straight through (already PIT in metrics_panel)
DIRECT = ["roe", "roe_diluted", "net_margin", "gross_margin", "revenue_yoy",
          "net_income_yoy", "debt_to_asset", "inventory_turnover",
          "operating_cash_to_revenue"]
# cumulative (YTD) statement fields that need TTM de-cumulation
CUMULATIVE = {"eps_basic": "eps_ttm", "ocfps": "ocfps_ttm"}


def ttm_from_ytd(q: pd.DataFrame, col: str) -> pd.Series:
    """Trailing-twelve-month value from a symbol's quarterly YTD-cumulative col.

    q: one symbol's rows, sorted by period_end, with a 'period_end' datetime.
    Method: de-cumulate to single-quarter (Q1 single = Q1 YTD; Qn single = Qn
    YTD - Q(n-1) YTD, only when the previous row is the immediately preceding
    quarter of the same fiscal year), then rolling 4-quarter sum requiring all
    four consecutive single quarters present. NaN otherwise (conservative).
    """
    pe = pd.to_datetime(q["period_end"])
    year = pe.dt.year.to_numpy()
    quarter = pe.dt.quarter.to_numpy()
    ytd = pd.to_numeric(q[col], errors="coerce").to_numpy(dtype=float)
    single = np.full(len(q), np.nan)
    for i in range(len(q)):
        if quarter[i] == 1:
            single[i] = ytd[i]
        elif i > 0 and year[i] == year[i - 1] and quarter[i] == quarter[i - 1] + 1:
            single[i] = ytd[i] - ytd[i - 1]
        # else: gap / non-consecutive -> leave NaN
    s = pd.Series(single, index=q.index)
    # rolling 4 consecutive quarters; require all present and each step +1 quarter
    step_ok = np.zeros(len(q), dtype=bool)
    for i in range(len(q)):
        if i >= 3 and not np.isnan(single[i - 3:i + 1]).any():
            # verify the 4 rows are 4 consecutive quarters
            qs = quarter[i - 3:i + 1]
            ys = year[i - 3:i + 1]
            seq = [(ys[j] * 4 + (qs[j] - 1)) for j in range(4)]
            if seq == list(range(seq[0], seq[0] + 4)):
                step_ok[i] = True
    ttm = s.rolling(4).sum()
    ttm[~step_ok] = np.nan
    return ttm


def build_quarterly(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["period_end"] = pd.to_datetime(m["period_end"], errors="coerce")
    m["available_at"] = pd.to_datetime(m["available_at"], errors="coerce")
    m = m.dropna(subset=["period_end", "available_at", "symbol"])
    m = m.sort_values(["symbol", "period_end"]).reset_index(drop=True)
    # rename to canonical fundamental names the trainer whitelists
    m = m.rename(columns={"debt_to_asset_ratio": "debt_to_asset"})
    out_cols = ["symbol", "period_end", "available_at", "bps"] + DIRECT + list(CUMULATIVE.values())
    parts = []
    for _, g in m.groupby("symbol", sort=False):
        g = g.copy()
        for src, dst in CUMULATIVE.items():
            g[dst] = ttm_from_ytd(g, src) if src in g else np.nan
        parts.append(g)
    q = pd.concat(parts, ignore_index=True)
    for c in out_cols:
        if c not in q.columns:
            q[c] = np.nan
    return q[out_cols]


def _cs_pct(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank within the caller's groupby (per date)."""
    return s.rank(pct=True)


def _z(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def _rolling_rank(df: pd.DataFrame, col: str, win: int, min_periods: int) -> np.ndarray:
    """Per-symbol trailing rolling percentile of the current value within its
    win-length window: fraction of window values <= current. Vectorized with
    sliding_window_view per symbol group. df must be sorted by [symbol, trade_date]."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(df), np.nan)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    # group boundaries (df already sorted by symbol)
    codes = pd.factorize(df["symbol"], sort=False)[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    bounds = np.r_[starts, len(df)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = vals[a:b]
        n = len(v)
        if n < min_periods:
            continue
        # expanding phase (min_periods..win): compare each pos to its prefix
        for i in range(min_periods - 1, min(win - 1, n)):
            w = v[: i + 1]
            m = np.isfinite(w).sum()
            if m >= min_periods:
                out[a + i] = np.nanmean((w <= v[i]).astype(float))
        if n >= win:
            sw = sliding_window_view(v, win)               # (n-win+1, win)
            last = v[win - 1:]
            rank = np.nanmean((sw <= last[:, None]).astype(float), axis=1)
            # positions win-1 .. n-1
            out[a + win - 1: b] = rank
    return out


def build_daily(quarterly: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """As-of merge the quarterly block onto keys(symbol, trade_date, close) with a
    strict backward merge_asof on available_at, then price ratios + percentiles."""
    keys = keys.copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    # merge_asof(by=symbol) requires BOTH frames globally sorted by the on-key
    keys = keys.dropna(subset=["symbol", "trade_date"]).sort_values(["trade_date", "symbol"])
    q = quarterly.copy()
    q["available_at"] = pd.to_datetime(q["available_at"])
    q["period_end"] = pd.to_datetime(q["period_end"])
    # deterministic as-of pick: 19% of (symbol, available_at) carry >1 statement
    # (e.g. two annuals disclosed the same day); the latest fiscal period must win
    # the tie. merge_asof(backward) keeps the LAST row among equal available_at,
    # so sort period_end ascending within each available_at.
    q = q.dropna(subset=["available_at"]).sort_values(["available_at", "period_end"])
    val_cols = ["bps"] + DIRECT + list(CUMULATIVE.values())
    merged = pd.merge_asof(
        keys, q[["symbol", "available_at"] + val_cols],
        left_on="trade_date", right_on="available_at", by="symbol",
        direction="backward", allow_exact_matches=True,
    )
    # PIT invariant: a fundamental visible at trade_date must have available_at <= trade_date
    assert (merged["available_at"].isna() | (merged["available_at"] <= merged["trade_date"])).all(), \
        "PIT violation: available_at > trade_date in merge_asof output"
    close = pd.to_numeric(merged["close"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["pb"] = np.where(merged["bps"] > 0, close / merged["bps"], np.nan)
        merged["pe_ttm"] = np.where(merged["eps_ttm"] > 0, close / merged["eps_ttm"], np.nan)
        merged["pcf"] = np.where(merged["ocfps_ttm"] > 0, close / merged["ocfps_ttm"], np.nan)
        merged["earnings_yield"] = merged["eps_ttm"] / close
        merged["ocf_yield"] = merged["ocfps_ttm"] / close
        merged["book_yield"] = np.where(merged["pb"] > 0, 1.0 / merged["pb"], np.nan)
    # cross-sectional percentiles per trade_date (cheap == high)
    g = merged.groupby("trade_date")
    ey_p = g["earnings_yield"].transform(_cs_pct)
    by_p = g["book_yield"].transform(_cs_pct)
    merged["valuation_percentile"] = (ey_p + by_p) / 2.0
    # own-history 2y (504 td) pb percentile -> re-rating / compression.
    # Vectorized per symbol (sliding_window_view) — a rolling.apply python lambda
    # over 6.78M rows is ~100x slower and gets rebuilt every retrain.
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    merged["pb_own_pctile_2y"] = _rolling_rank(merged, "pb", win=504, min_periods=120)
    # composites (cross-sectional z per date)
    def _comp(gdf: pd.DataFrame) -> pd.Series:
        qual = (_z(gdf["roe"]) + _z(gdf["net_margin"]) + _z(gdf["gross_margin"])
                + _z(gdf["operating_cash_to_revenue"]) - _z(gdf["debt_to_asset"]))
        return qual
    merged["quality_composite"] = merged.groupby("trade_date", group_keys=False).apply(_comp)
    merged["growth_composite"] = merged.groupby("trade_date", group_keys=False).apply(
        lambda gdf: _z(gdf["revenue_yoy"]) + _z(gdf["net_income_yoy"]))
    # honest missingness flags
    merged["missing_fundamentals"] = merged["roe"].isna().astype("int8")
    merged["missing_valuation"] = merged["pb"].isna().astype("int8")
    feat = (["symbol", "trade_date", "pb", "pe_ttm", "pcf", "earnings_yield", "ocf_yield",
             "book_yield", "valuation_percentile", "pb_own_pctile_2y",
             "quality_composite", "growth_composite", "eps_ttm", "ocfps_ttm",
             "missing_fundamentals", "missing_valuation"] + DIRECT)
    return merged[feat].reset_index(drop=True)


def self_test() -> int:
    """Hand-verified TTM: 000001.SZ 2025Q3 EPS TTM should be ~2.08."""
    m = pd.read_parquet(METRICS)
    g = m[m["symbol"] == "000001.SZ"].copy()
    g["period_end"] = pd.to_datetime(g["period_end"])
    g = g.sort_values("period_end").reset_index(drop=True)
    g["eps_ttm"] = ttm_from_ytd(g, "eps_basic")
    row = g[g["period_end"] == "2025-09-30"]
    val = float(row["eps_ttm"].iloc[0])
    print(f"000001.SZ 2025Q3 eps_ttm = {val:.4f} (expected 2.08)")
    assert abs(val - 2.08) < 0.02, f"TTM self-test FAILED: {val}"
    # de-cumulation sanity: a single 2024 -> reset in 2025Q1
    q1_2025 = g[g["period_end"] == "2025-03-31"]["eps_ttm"]
    print("de-cumulation + rolling-4 TTM self-test PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--daily", action="store_true", help="also build the daily block")
    ap.add_argument("--keys", default=None,
                    help="parquet with symbol,trade_date,close for the daily block")
    ap.add_argument("--metrics", default=str(METRICS))
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    metrics = pd.read_parquet(args.metrics)
    q = build_quarterly(metrics)
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT_Q, index=False)
    cov = q["eps_ttm"].notna().mean()
    print(f"wrote {OUT_Q.name}: {len(q):,} rows, {q['symbol'].nunique()} syms; "
          f"eps_ttm coverage {cov:.1%}", flush=True)

    if args.daily:
        assert args.keys, "--daily needs --keys <parquet with symbol,trade_date,close>"
        keys = pd.read_parquet(args.keys, columns=["symbol", "trade_date", "close"])
        daily = build_daily(q, keys)
        daily.to_parquet(OUT_DAILY, index=False)
        for c in ["pb", "roe", "pe_ttm", "valuation_percentile"]:
            print(f"  {c:22s} coverage {daily[c].notna().mean():.1%}")
        print(f"wrote {OUT_DAILY.name}: {len(daily):,} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
