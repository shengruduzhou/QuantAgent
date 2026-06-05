"""Enrich silver/market_panel.parquet with tradability flags.

The qlib-derived silver panel ships OHLCV only. The v8 ``decision_chain``
gates (``is_st`` / ``is_limit_up`` / ``is_limit_down`` / ``is_suspended``)
need per-(symbol, trade_date) booleans to function. This script computes
them in-place once a real ``silver/st_flags/st_flags.parquet`` is on disk.

Derivation
----------
* ``is_st``         — broadcast current-snapshot ST status from
                       ``st_flags.parquet`` by symbol. All historical
                       rows of a currently-ST symbol get ``is_st=True``.
                       This is the documented "current snapshot
                       broadcast" semantics. An ``is_st_provenance``
                       column records the caveat for auditors.
* ``is_suspended``  — ``volume == 0`` (in A-share, non-trading days have
                       no volume entirely; we never see partial bars).
* ``is_limit_up``   — ``round(close, 2) ≈ round(prev_close × (1+cap), 2)``
* ``is_limit_down`` — ``round(close, 2) ≈ round(prev_close × (1−cap), 2)``
  where ``cap = 0.05 if is_st else 0.10``. The 0.10 default holds for
  non-ST main-board + ChiNext/STAR (the 0.20 cap on those boards
  technically applies post-2020, but mis-classifying a 20-cap into
  a 10-cap only over-counts limit-up events at the gate; it never
  silently leaks future information).

Run-once semantics
------------------
A pre-enrichment copy is written to
``market_panel.qlib_only_backup.parquet`` so we can roll back. The
script refuses to run if the panel already carries non-trivial flag
columns unless ``QA_FORCE=1`` is set.

Env vars:
  QA_DATA_ROOT  — silver layer root (default: runtime/data/v7)
  QA_FORCE      — set 1 to overwrite an already-enriched panel

Runtime: ~30 s on the existing 15 M-row × 14-col panel.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _round2(s: pd.Series) -> pd.Series:
    return s.round(2)


def main() -> int:
    data_root = Path(os.environ.get("QA_DATA_ROOT", "runtime/data/v7"))
    panel_path = data_root / "silver" / "market_panel" / "market_panel.parquet"
    st_path    = data_root / "silver" / "st_flags" / "st_flags.parquet"
    backup_path = panel_path.with_name("market_panel.qlib_only_backup.parquet")
    report_path = panel_path.with_name("enrichment_report.json")

    if not panel_path.exists():
        raise SystemExit(f"market_panel missing: {panel_path}")
    if not st_path.exists():
        raise SystemExit(f"st_flags missing: {st_path}")

    print(f"loading panel  : {panel_path}")
    t0 = time.time()
    panel = pd.read_parquet(panel_path)
    print(f"  rows={len(panel)}  cols={list(panel.columns)}  ({time.time()-t0:.1f}s)")
    if any(col in panel.columns for col in ("is_st", "is_limit_up", "is_limit_down")):
        if os.environ.get("QA_FORCE", "0") != "1":
            raise SystemExit(
                "panel already has flag columns — set QA_FORCE=1 to overwrite, "
                f"or restore from {backup_path}"
            )
        print("  WARN existing flag columns present, dropping (QA_FORCE=1)")
        panel = panel.drop(columns=[c for c in
                                    ("is_st", "is_limit_up", "is_limit_down", "is_suspended",
                                     "is_st_provenance", "prev_close")
                                    if c in panel.columns])

    print(f"loading st     : {st_path}")
    st = pd.read_parquet(st_path, columns=["symbol", "is_st", "st_known", "st_source"])
    st["symbol"] = st["symbol"].astype(str).str.strip()
    st_lookup = st[["symbol", "is_st", "st_source"]].rename(
        columns={"is_st": "is_st_current", "st_source": "is_st_source"}
    )

    # Back up the pre-enrichment panel
    if not backup_path.exists():
        print(f"backing up to  : {backup_path}")
        shutil.copy2(panel_path, backup_path)
    else:
        print(f"backup already present, not overwriting: {backup_path}")

    # Sort once for the per-symbol shift
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel["symbol"] = panel["symbol"].astype(str).str.strip()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")

    print("computing flags …")
    t0 = time.time()
    # is_st broadcast
    panel = panel.merge(st_lookup, on="symbol", how="left")
    panel["is_st"] = panel["is_st_current"].fillna(False).astype(bool)
    panel["is_st_provenance"] = panel["is_st_source"].fillna("missing").astype(str)
    panel = panel.drop(columns=["is_st_current", "is_st_source"])

    # is_suspended
    panel["is_suspended"] = (panel["volume"].fillna(0) == 0)

    # prev_close per symbol
    panel["prev_close"] = panel.groupby("symbol", sort=False)["close"].shift(1)

    # cap pct + limit thresholds
    cap = np.where(panel["is_st"].to_numpy(), 0.05, 0.10)
    prev = panel["prev_close"].to_numpy()
    close_r = _round2(panel["close"]).to_numpy()
    cap_up = np.round(prev * (1.0 + cap), 2)
    cap_dn = np.round(prev * (1.0 - cap), 2)
    panel["is_limit_up"]   = np.isfinite(prev) & (np.abs(close_r - cap_up) < 0.005)
    panel["is_limit_down"] = np.isfinite(prev) & (np.abs(close_r - cap_dn) < 0.005)

    # ensure dtypes
    for c in ("is_st", "is_suspended", "is_limit_up", "is_limit_down"):
        panel[c] = panel[c].fillna(False).astype(bool)
    print(f"  ({time.time()-t0:.1f}s)")

    # Drop intermediate prev_close to keep the panel lean (decision_chain
    # recomputes it from K-line)
    panel = panel.drop(columns=["prev_close"])

    print("writing back …")
    t0 = time.time()
    panel.to_parquet(panel_path, index=False)
    print(f"  ({time.time()-t0:.1f}s)")

    # Per-year report
    panel["year"] = panel["trade_date"].dt.year
    by_year = panel.groupby("year").agg(
        rows=("symbol", "size"),
        is_st_count=("is_st", "sum"),
        is_suspended_count=("is_suspended", "sum"),
        is_limit_up_count=("is_limit_up", "sum"),
        is_limit_down_count=("is_limit_down", "sum"),
    )
    by_year["is_st_rate"]         = by_year["is_st_count"]         / by_year["rows"]
    by_year["is_suspended_rate"]  = by_year["is_suspended_count"]  / by_year["rows"]
    by_year["is_limit_up_rate"]   = by_year["is_limit_up_count"]   / by_year["rows"]
    by_year["is_limit_down_rate"] = by_year["is_limit_down_count"] / by_year["rows"]

    print("\nper-year flag rates:")
    print(by_year[["rows", "is_st_rate", "is_suspended_rate",
                    "is_limit_up_rate", "is_limit_down_rate"]].round(4).to_string())

    max_lu_rate = float(by_year["is_limit_up_rate"].max())
    warn = max_lu_rate > 0.05
    if warn:
        print(f"\nWARN max yearly is_limit_up rate = {max_lu_rate:.4f} > 0.05 — "
              "check price precision or ST classification.")

    report = {
        "panel_path": str(panel_path),
        "backup_path": str(backup_path),
        "st_source": str(st_path),
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "broadcast_caveat": "is_st is a current-snapshot broadcast — historical accuracy approximate",
        "by_year": by_year.reset_index().to_dict(orient="records"),
        "limit_up_rate_warn": warn,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
