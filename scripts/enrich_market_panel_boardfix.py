#!/usr/bin/env python3
"""Recompute board-aware limit-up/down flags for the silver market panel.

The silver ``market_panel.parquet`` flags were materialised with a flat 10%
price-limit approximation (verified: ChiNext ``is_limit_up`` rows fire at a
median move of exactly +10.0%). ChiNext/STAR limits are 20% and BSE is 30%, so
the flat rule both false-flags non-main-board names at +10% (treating buyable
names as sealed) and misses real +20%/+30% seals (phantom tradability).

This script recomputes ``is_limit_up`` / ``is_limit_down`` with the canonical
board-aware engine (``quant_math.ashare``), **non-destructively**: it writes a
sidecar ``market_panel_boardfix.parquet`` + a manifest and prints the exact
flip statistics. The original panel is never overwritten.

Residual caveat: limits are nominal but the panel close is qfq-adjusted, so the
adjusted close/prev_close ratio differs from the raw ratio on ex-dividend days.
Board width is the first-order fix; an ex-div-exact recompute needs raw close
(a network re-fetch). The flag here is strictly more correct than flat-10%.

Usage:
  AI_quant_venv/bin/python3 scripts/enrich_market_panel_boardfix.py \
      [--panel PATH] [--out PATH] [--tolerance 0.005]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import AshareRuleEngine, board_price_limit_vector

DEFAULT_PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def recompute_flags(panel: pd.DataFrame, tolerance: float = 0.005) -> pd.DataFrame:
    """Return panel with board-aware is_limit_up / is_limit_down recomputed."""
    df = panel.sort_values(["symbol", "trade_date"]).copy()
    is_st = df["is_st"] if "is_st" in df.columns else False
    ratio = board_price_limit_vector(df["symbol"], is_st)
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    close_r = df["close"].round(2)
    cap_up = (prev_close * (1.0 + ratio)).round(2)
    cap_dn = (prev_close * (1.0 - ratio)).round(2)
    valid = prev_close > 0
    df["is_limit_up_boardfix"] = ((close_r - cap_up).abs() < tolerance).fillna(False) & valid
    df["is_limit_down_boardfix"] = ((close_r - cap_dn).abs() < tolerance).fillna(False) & valid
    df["board"] = df["symbol"].map(AshareRuleEngine().infer_board)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--out", default=None, help="defaults to <panel_dir>/market_panel_boardfix.parquet")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--report", default="reports/data/boardfix_flip_stats.json")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    out_path = Path(args.out) if args.out else panel_path.with_name("market_panel_boardfix.parquet")

    cols = ["symbol", "trade_date", "close", "is_st", "is_limit_up", "is_limit_down"]
    df = pd.read_parquet(panel_path, columns=cols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    fixed = recompute_flags(df, tolerance=args.tolerance)

    old_up = fixed["is_limit_up"].fillna(False).astype(bool)
    old_dn = fixed["is_limit_down"].fillna(False).astype(bool)
    new_up = fixed["is_limit_up_boardfix"].astype(bool)
    new_dn = fixed["is_limit_down_boardfix"].astype(bool)

    stats: dict = {"panel": str(panel_path), "rows": int(len(fixed)), "tolerance": args.tolerance, "by_board": {}}
    for board, g in fixed.groupby("board"):
        ou, nu = g["is_limit_up"].fillna(False).astype(bool), g["is_limit_up_boardfix"].astype(bool)
        od, nd = g["is_limit_down"].fillna(False).astype(bool), g["is_limit_down_boardfix"].astype(bool)
        stats["by_board"][str(board)] = {
            "rows": int(len(g)),
            "limit_up_old": int(ou.sum()), "limit_up_new": int(nu.sum()),
            "limit_up_cleared_false_positive": int((ou & ~nu).sum()),
            "limit_up_added_missed_seal": int((~ou & nu).sum()),
            "limit_down_old": int(od.sum()), "limit_down_new": int(nd.sum()),
            "limit_down_cleared_false_positive": int((od & ~nd).sum()),
            "limit_down_added_missed_seal": int((~od & nd).sum()),
        }
    stats["totals"] = {
        "limit_up_changed": int((old_up != new_up).sum()),
        "limit_down_changed": int((old_dn != new_dn).sum()),
    }

    # Write corrected sidecar (board-aware flags become the canonical columns).
    out_df = fixed[["symbol", "trade_date", "board", "is_limit_up_boardfix", "is_limit_down_boardfix"]].rename(
        columns={"is_limit_up_boardfix": "is_limit_up", "is_limit_down_boardfix": "is_limit_down"}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    manifest = {
        "source_panel": str(panel_path), "generated_from": "enrich_market_panel_boardfix.py",
        "rows": int(len(out_df)), "boards": sorted(out_df["board"].unique().tolist()),
        "method": f"board-aware AshareRuleEngine ratios; ST 5% override; tolerance {args.tolerance:.4f}",
        "caveat": "limits nominal but close is qfq-adjusted; ex-dividend days approximate. Strictly more correct than flat-10%.",
        "flip_stats": stats,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote corrected flags → {out_path}")
    print(f"wrote flip stats     → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
