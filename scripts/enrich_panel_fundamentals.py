#!/usr/bin/env python3
"""Add point-in-time fundamental columns to the silver market panel.

The silver market panel ships OHLCV (+ tradability flags) only, so the factor
DSL's valuation/quality ``OptionalColumn`` nodes (pb / roe / gross_margin /
debt_to_asset) evaluate to NaN and every fundamental factor is auto-rejected by
the finite-ratio gate. This script joins the PIT fundamentals panel
(``silver/fundamentals/metrics_panel.parquet``, which carries an ``available_at``
timestamp = when each statement became public) onto every (symbol, trade_date)
row with a strict backward ``merge_asof`` — so a fundamental is only visible on
trade dates at/after it was announced. No look-ahead.

Columns produced (mapped to expr.OptionalColumn names):
  pb            = close / bps           (price-to-book; bps = book value/share)
  roe           = roe                   (return on equity, as reported)
  gross_margin  = gross_margin
  debt_to_asset = debt_to_asset_ratio

Deliberately NOT produced: pe_ttm (needs trailing-4Q EPS de-cumulation, easy to
get wrong on A-share cumulative statements) and turnover_rate (needs shares
outstanding, absent here). Factors using those stay NaN -> rejected, which is
the safe default.

Output: a NEW file (default market_panel_fund.parquet) carrying every original
panel column (incl. the is_st/is_suspended/is_limit_up flags the OOS evaluator
needs) plus the four fundamental columns, so it can serve as the --market-panel
for BOTH discovery and evaluate_discovered_factors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
METRICS = "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
OUT = "runtime/data/v7/silver/market_panel/market_panel_fund.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--metrics", default=METRICS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str)

    metrics = pd.read_parquet(
        args.metrics,
        columns=["symbol", "available_at", "bps", "roe", "gross_margin", "debt_to_asset_ratio"],
    )
    # The silver panel already has its own ``available_at`` (bar PIT stamp), so
    # rename the fundamentals timestamp to avoid a merge collision.
    metrics["fund_available_at"] = pd.to_datetime(metrics["available_at"], errors="coerce")
    metrics = metrics.drop(columns=["available_at"])
    metrics["symbol"] = metrics["symbol"].astype(str)
    metrics = metrics.dropna(subset=["fund_available_at"]).rename(columns={"debt_to_asset_ratio": "debt_to_asset"})
    # One row per (symbol, fund_available_at); keep the latest-filed if duplicated.
    metrics = metrics.sort_values(["symbol", "fund_available_at"]).drop_duplicates(
        ["symbol", "fund_available_at"], keep="last"
    )

    # Strict PIT backward as-of join: each trade_date sees only fundamentals
    # whose fund_available_at <= trade_date.
    left = panel.sort_values("trade_date")
    right = metrics.sort_values("fund_available_at")
    joined = pd.merge_asof(
        left,
        right,
        left_on="trade_date",
        right_on="fund_available_at",
        by="symbol",
        direction="backward",
    )

    close = pd.to_numeric(joined["close"], errors="coerce")
    bps = pd.to_numeric(joined["bps"], errors="coerce")
    joined["pb"] = np.where(bps > 0, close / bps, np.nan).astype("float32")
    joined["roe"] = pd.to_numeric(joined["roe"], errors="coerce").astype("float32")
    joined["gross_margin"] = pd.to_numeric(joined["gross_margin"], errors="coerce").astype("float32")
    joined["debt_to_asset"] = pd.to_numeric(joined["debt_to_asset"], errors="coerce").astype("float32")
    joined = joined.drop(columns=["fund_available_at", "bps"])

    joined = joined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, index=False)

    cov = {c: round(float(joined[c].notna().mean()), 3) for c in ["pb", "roe", "gross_margin", "debt_to_asset"]}
    # Coverage on recent rows (post-2018, where discovery/eval actually operate).
    recent = joined[joined["trade_date"] >= "2018-01-01"]
    cov_recent = {c: round(float(recent[c].notna().mean()), 3) for c in ["pb", "roe", "gross_margin", "debt_to_asset"]}
    summary = {
        "out": str(out),
        "rows": int(len(joined)),
        "added_columns": ["pb", "roe", "gross_margin", "debt_to_asset"],
        "coverage_all": cov,
        "coverage_post_2018": cov_recent,
        "pb_describe": {k: round(float(v), 3) for k, v in recent["pb"].describe().to_dict().items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
