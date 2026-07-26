#!/usr/bin/env python3
"""Forensic test: is a price panel raw or already adjusted?

The test replays the ex-rights factor series against the panel's own returns.
On an ex-rights session a RAW series drops by the entitlement, so the realised
return should match ``1/(1+factor_step) - 1``; an already-adjusted series shows
no such drop and the sign agreement collapses to a coin flip.

This is the check that exposed the defect this work fixes: the previous
full-universe panel declared ``adjustment_method = "none"`` for every row while
its frozen 3,872-symbol cohort was forward-adjusted and only the 235 backfilled
names were raw.

Output: runtime/data/u0/validation/adjustment_forensics.json

Usage: AI_quant_venv/bin/python3 scripts/u0_adjustment_forensics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FACTORS = REPO / "runtime/data/u0/pit/adjust_factors.parquet"
NEW_PANEL = REPO / "runtime/data/u0/panel/daily_bars_raw.parquet"
LEGACY_PANEL = REPO / "runtime/data/v7/full_universe/full_universe_market_panel.parquet"
OUT = REPO / "runtime/data/u0/validation"

#: Below this sign agreement the series cannot be raw.
RAW_THRESHOLD = 0.80


def ex_rights_events() -> pd.DataFrame:
    factors = pd.read_parquet(FACTORS)
    factors["effective_date"] = pd.to_datetime(factors["effective_date"])
    factors = factors.sort_values(["symbol", "effective_date"])
    factors["factor_step"] = factors.groupby("symbol")["hfq_factor"].pct_change()
    return factors[factors["factor_step"].abs() > 0.01]


def assess(panel: pd.DataFrame, events: pd.DataFrame, label: str) -> dict:
    panel = panel.dropna(subset=["close"]).sort_values(["symbol", "trade_date"]).copy()
    panel["prev_close"] = panel.groupby("symbol")["close"].shift(1)
    panel = panel[panel["prev_close"] > 0]
    panel["ret"] = panel["close"] / panel["prev_close"] - 1.0
    merged = events.merge(panel, left_on=["symbol", "effective_date"],
                          right_on=["symbol", "trade_date"], how="inner")
    merged["expected_ret"] = 1.0 / (1.0 + merged["factor_step"]) - 1.0
    merged = merged[merged["expected_ret"].abs() > 0.02]
    if merged.empty:
        return {"label": label, "events_tested": 0, "verdict": "INSUFFICIENT_OVERLAP"}
    agreement = float((np.sign(merged["ret"]) == np.sign(merged["expected_ret"])).mean())
    return {
        "label": label,
        "events_tested": int(len(merged)),
        "sign_agreement": round(agreement, 4),
        "median_abs_error": round(float((merged["ret"] - merged["expected_ret"]).abs().median()), 4),
        "verdict": "RAW" if agreement >= RAW_THRESHOLD else "ADJUSTED_OR_MIXED",
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not FACTORS.exists():
        print("missing adjust_factors.parquet — run u0_pit_intervals.py factors first")
        return 3
    events = ex_rights_events()
    results = []
    if NEW_PANEL.exists():
        panel = pd.read_parquet(NEW_PANEL, columns=["symbol", "trade_date", "close"])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        results.append(assess(panel, events, "u0/panel/daily_bars_raw.parquet"))
    if LEGACY_PANEL.exists():
        legacy = pd.read_parquet(LEGACY_PANEL,
                                 columns=["symbol", "trade_date", "close", "source_track"])
        legacy["trade_date"] = pd.to_datetime(legacy["trade_date"])
        legacy["symbol"] = legacy["symbol"].astype(str)
        results.append(assess(legacy, events, "legacy full_universe_market_panel.parquet (all)"))
        for track, subset in legacy.groupby("source_track"):
            results.append(assess(subset, events, f"legacy panel · source_track={track}"))

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": ("replay ex-rights factor steps against realised returns; a raw series drops by "
                   "1/(1+factor_step)-1 on the ex-date, an adjusted series does not"),
        "raw_sign_agreement_threshold": RAW_THRESHOLD,
        "results": results,
        "conclusion": ("A panel whose declared adjustment_method disagrees with its verdict here is "
                       "mislabelled; mixed verdicts across source tracks inside one panel mean the "
                       "panel silently blends adjustment conventions."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adjustment_forensics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
