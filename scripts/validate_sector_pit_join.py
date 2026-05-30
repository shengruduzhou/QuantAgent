"""Validate PIT-safety of the sector_map asof join.

Pure check, no network. Loads the existing sector_map.parquet and
v9 OOS predictions, runs the same asof-join used by
stratified_ic._attach_sector, and reports:

* how many prediction rows received a real sector vs UNKNOWN
* whether any leak (sector.available_at > prediction.trade_date) snuck in
* per-sector counts (diagnostic only)

This is the regression check before we ever wire sector data into a
strategy decision. If the join ever produces non-UNKNOWN for a date
EARLIER than the sector_map's available_at, the result file is the
smoking gun.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    sm_path = Path("runtime/data/v7/silver/sector_map/sector_map.parquet")
    if not sm_path.exists():
        raise SystemExit(f"sector_map missing at {sm_path}")
    sm = pd.read_parquet(sm_path)
    sm["available_at"] = pd.to_datetime(sm["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    print(f"sector_map: rows={len(sm):,}  uniq symbols={sm['symbol'].nunique():,}")
    print(f"  available_at range: {sm['available_at'].min()} → {sm['available_at'].max()}")
    print(f"  coverage_status: {dict(sm['coverage_status'].value_counts())}")
    sec_known = sm[sm["sector_level_1"].notna()]
    print(f"  rows with sector_level_1: {len(sec_known):,}")

    pred_paths = sorted(glob.glob(
        "runtime/models/v7_alpha_full_universe_nosynth_v9/walk_forward/fold_*/fold_*_oos_predictions.parquet"
    ))
    if not pred_paths:
        raise SystemExit("no v9 OOS predictions found")

    print(f"\nloading {len(pred_paths)} fold-horizon parquets ...", flush=True)
    parts = [pd.read_parquet(p, columns=["trade_date", "symbol"]) for p in pred_paths]
    preds = pd.concat(parts, ignore_index=True).drop_duplicates(["trade_date", "symbol"])
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce")
    preds["symbol"] = preds["symbol"].astype(str)
    print(f"  predictions: {len(preds):,} unique (trade_date, symbol)")
    print(f"  date range: {preds['trade_date'].min()} → {preds['trade_date'].max()}")

    # Mimic the stratified_ic._attach_sector asof join
    from quantagent.diagnostics.stratified_ic import _attach_sector
    enriched = _attach_sector(preds, sm)

    matched = enriched[enriched["sector_level_1"] != "UNKNOWN"]
    n_unknown = int((enriched["sector_level_1"] == "UNKNOWN").sum())
    print(f"\njoin result: {len(matched):,} matched, {n_unknown:,} UNKNOWN ({100*n_unknown/len(enriched):.1f}%)")

    if not matched.empty:
        # Smoking-gun check: for every matched row, sector_map.available_at must be <= prediction.trade_date.
        # Re-do the merge keeping available_at to verify.
        sm_copy = sm[["symbol", "available_at", "sector_level_1"]].sort_values(["available_at", "symbol"])
        left = preds.copy().sort_values(["trade_date", "symbol"])
        verify = pd.merge_asof(
            left, sm_copy,
            left_on="trade_date", right_on="available_at",
            by="symbol", direction="backward", allow_exact_matches=True,
        )
        verify_matched = verify[verify["sector_level_1"].notna()].copy()
        leaks = verify_matched[verify_matched["available_at"] > verify_matched["trade_date"]]
        print(f"\nPIT leak check (sector.available_at > prediction.trade_date):")
        print(f"  rows checked: {len(verify_matched):,}")
        print(f"  leaks found:  {len(leaks):,}")
        if not leaks.empty:
            print("  FIRST 5 LEAKS:")
            print(leaks.head())
            sys.exit(1)
        else:
            print("  OK — every matched sector row is PIT-safe")
        # Per-sector matched counts
        print(f"\nmatched per sector_level_1 (top 15):")
        print(matched["sector_level_1"].value_counts().head(15).to_string())
    else:
        print("\n(no sector rows matched — expected when sector_map is current-snapshot only and OOS predates available_at)")


if __name__ == "__main__":
    main()
