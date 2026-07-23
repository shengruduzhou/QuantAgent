#!/usr/bin/env python3
"""H-031 Track U0: provider coverage matrix (report-u0-provider-coverage).

Builds `runtime/data/u0/provider_coverage_matrix.{parquet,csv}` — one row per
security in the authoritative historical security master, recording which
provider (if any) actually carries usable bar history, how complete it is, and
why a security is blocked when no provider covers it.

Governing rules (H-031 §6):
  * exchange sources are authoritative for identity/listing/board/code mapping;
  * a market-data provider's EMPTY response is NOT proof a security has no
    history — it is recorded as EMPTY/NOT_PROBED, never silently as "no data";
  * two bar providers are never merged without recording the source boundary
    (this matrix records `selected_bar_provider` per symbol, one source only);
  * a missing mandatory source yields BLOCKED_BY_DATA, never a default false.

Bounded memory: reads only [symbol, trade_date, close] from the panel and
aggregates once. Side-effect-free apart from its own artifacts under
runtime/data/u0/. Never reads or reports candidate performance.

Usage: AI_quant_venv/bin/python3 scripts/u0_provider_coverage.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
SUPPLEMENTAL = REPO / "runtime/data/u0/master_supplemental_additions.parquet"


def _load_master() -> "pd.DataFrame":
    """H-028 master unioned with reconciliation-approved supplemental additions."""
    m = pd.read_parquet(MASTER)
    if SUPPLEMENTAL.exists():
        try:
            add = pd.read_parquet(SUPPLEMENTAL)
            shared = [c for c in m.columns if c in add.columns]
            add = add[shared]
            add = add[~add["symbol"].astype(str).isin(set(m["symbol"].astype(str)))]
            if len(add):
                m = pd.concat([m, add], ignore_index=True)
        except Exception:
            pass
    return m
FROZEN_PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
STAGING = REPO / "runtime/data/v7/full_universe/_staging"
FAILED_LEDGER = REPO / "runtime/data/v7/full_universe/failed_fetch_ledger.csv"
OUT = REPO / "runtime/data/u0"

# TickFlow is the entitled DAILY bar provider (see tickflow-tier-capabilities);
# akshare is a manually-invokable fallback; qlib is a local historical library;
# tushare carries PIT fundamentals/metadata, not bars.
ADJUSTMENT_METHOD = "none"        # raw as-of-day (matches FRESH_HOLDOUT manifest)
VOLUME_UNIT = "shares"            # lots x100 normalised at ingest
AMOUNT_UNIT = "CNY"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def _panel_coverage(panel_path: Path) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Per-symbol coverage aggregates + the trading calendar, bounded memory."""
    df = pd.read_parquet(panel_path, columns=["symbol", "trade_date", "close"])
    df["symbol"] = df["symbol"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    calendar = sorted(df["trade_date"].dt.normalize().unique())
    grp = df.groupby("symbol", sort=False)
    agg = grp["trade_date"].agg(coverage_start="min", coverage_end="max", bar_count="size")
    agg["actual_trading_days"] = grp["trade_date"].nunique()
    agg["null_close"] = grp["close"].apply(lambda s: int(s.isna().sum()))
    return agg.reset_index(), calendar


def _staging_coverage() -> pd.DataFrame:
    rows = []
    if STAGING.exists():
        for f in sorted(STAGING.glob("sym_*.parquet")):
            try:
                s = pd.read_parquet(f, columns=["symbol", "trade_date", "close"])
            except Exception:
                continue
            if not len(s):
                continue
            td = pd.to_datetime(s["trade_date"])
            rows.append({
                "symbol": str(s["symbol"].iloc[0]),
                "coverage_start": td.min(), "coverage_end": td.max(),
                "bar_count": int(len(s)), "actual_trading_days": int(td.nunique()),
                "null_close": int(s["close"].isna().sum()),
            })
    return pd.DataFrame(rows)


def _expected_trading_days(calendar: list[pd.Timestamp], start, end) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return int(sum(1 for d in calendar if start <= d <= end))


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    master = _load_master()
    master["symbol"] = master["symbol"].astype(str)
    for c in ("listing_date", "delisting_date"):
        master[c] = pd.to_datetime(master[c], errors="coerce")
    master_hash = _sha_file(MASTER)

    frozen_cov, calendar = _panel_coverage(FROZEN_PANEL)
    frozen_map = frozen_cov.set_index("symbol")
    stg_cov = _staging_coverage()
    stg_map = stg_cov.set_index("symbol") if len(stg_cov) else pd.DataFrame()
    failed = set()
    if FAILED_LEDGER.exists():
        failed = set(pd.read_csv(FAILED_LEDGER)["symbol"].astype(str))

    frozen_syms = set(frozen_map.index)
    staged_syms = set(stg_map.index) if len(stg_map) else set()
    panel_max = max(calendar) if calendar else pd.NaT
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # H-032A: a representative probe tells us whether an as-yet-unfetched board is
    # actually FETCHABLE (so NOT_PROBED means backlog, not "no data") vs unsupported.
    probe = REPO / "runtime/data/u0/star_bse_probe_report.json"
    fetchable_boards = set()
    if probe.exists():
        pr = json.loads(probe.read_text())
        for board, diag in pr.get("diagnosis", {}).items():
            if "FETCHABLE" in str(diag):
                fetchable_boards.add(board)

    records = []
    for _, m in master.iterrows():
        sym = m["symbol"]
        in_frozen = sym in frozen_syms
        in_staged = sym in staged_syms
        src = frozen_map if in_frozen else (stg_map if in_staged else None)

        cov_start = cov_end = pd.NaT
        bar_count = actual_td = 0
        if src is not None:
            r = src.loc[sym]
            cov_start, cov_end = r["coverage_start"], r["coverage_end"]
            bar_count = int(r["bar_count"])
            actual_td = int(r["actual_trading_days"])

        # provider statuses — strict vocabulary; EMPTY is distinct from NOT_PROBED
        # (an empty vendor response is never recorded as proof of "no history").
        covered = in_frozen or in_staged
        if in_frozen:
            tickflow_status, source_boundary = "COVERED_FROZEN_COHORT", "frozen_cohort<=2026-05-18+daily_topup"
        elif in_staged:
            tickflow_status, source_boundary = "COVERED_BACKFILL", "u0_backfill"
        elif sym in failed:
            tickflow_status, source_boundary = "EMPTY_PROVIDER_RESPONSE", ""
        else:
            tickflow_status, source_boundary = "NOT_PROBED", ""

        # disposition of the (not-yet-covered) securities using the strict vocabulary
        if covered:
            retry_class = "OK"
        elif sym in failed:
            retry_class = "NO_RELIABLE_HISTORY"       # 3x retried, still empty
        elif str(m["board"]) in fetchable_boards:
            retry_class = "FETCHABLE_NOT_PROBED"      # probe passed; pure backfill backlog
        else:
            retry_class = "NOT_PROBED"

        # fallback/other bar providers were not exercised for this cohort
        akshare_status = "NOT_PROBED"
        qlib_status = "NOT_PROBED"
        tushare_status = "METADATA_ONLY"        # PIT fundamentals, not bars
        exchange_metadata_status = "AUTHORITATIVE"  # identity/board/listing from exchange source

        selected_bar_provider = "tickflow" if covered else "NONE"
        selected_metadata_provider = f"exchange:{m['source']}"

        end_ref = m["delisting_date"] if pd.notna(m["delisting_date"]) else panel_max
        expected_td = _expected_trading_days(calendar, cov_start, cov_end) if covered else \
            _expected_trading_days(calendar, m["listing_date"], end_ref)
        missing_ratio = None
        if expected_td > 0:
            missing_ratio = round(max(0.0, 1.0 - actual_td / expected_td), 4)

        blocked = ""
        if not covered:
            if sym in failed:
                blocked = "BLOCKED_BY_DATA:no_reliable_history(3x_empty;retry_or_fallback_provider)"
            elif retry_class == "FETCHABLE_NOT_PROBED":
                blocked = "COVERAGE_BACKLOG:fetchable_not_probed(run_backfill)"
            else:
                blocked = "BLOCKED_BY_DATA:not_yet_backfilled"
        elif missing_ratio is not None and missing_ratio > 0.05:
            blocked = f"PARTIAL_COVERAGE:missing_ratio={missing_ratio}"

        records.append({
            "symbol": sym, "exchange": m["exchange"], "board": m["board"],
            "security_type": m["security_type"],
            "listing_date": m["listing_date"].date().isoformat() if pd.notna(m["listing_date"]) else None,
            "delisting_date": m["delisting_date"].date().isoformat() if pd.notna(m["delisting_date"]) else None,
            "current_status": m["status"],
            "tickflow_status": tickflow_status, "tushare_status": tushare_status,
            "akshare_status": akshare_status, "qlib_status": qlib_status,
            "exchange_metadata_status": exchange_metadata_status,
            "selected_bar_provider": selected_bar_provider,
            "selected_metadata_provider": selected_metadata_provider,
            "source_boundary": source_boundary,
            "provider_retry_class": retry_class,
            "coverage_start": cov_start.date().isoformat() if pd.notna(cov_start) else None,
            "coverage_end": cov_end.date().isoformat() if pd.notna(cov_end) else None,
            "bar_count": bar_count, "expected_trading_days": expected_td,
            "actual_trading_days": actual_td, "missing_ratio": missing_ratio,
            "adjustment_method": ADJUSTMENT_METHOD if covered else None,
            "volume_unit": VOLUME_UNIT if covered else None,
            "amount_unit": AMOUNT_UNIT if covered else None,
            "source_timestamp": now,
            "source_hash": master_hash,
            "blocked_reason": blocked,
        })

    matrix = pd.DataFrame.from_records(records)
    matrix.to_parquet(OUT / "provider_coverage_matrix.parquet", index=False)
    matrix.to_csv(OUT / "provider_coverage_matrix.csv", index=False)

    covered_mask = matrix["selected_bar_provider"] == "tickflow"
    summary = {
        "generated": now, "experiment": "H-031 Track U0 provider coverage",
        "master_securities": int(len(matrix)),
        "covered_bar_history": int(covered_mask.sum()),
        "covered_frozen_cohort": int((matrix["tickflow_status"] == "COVERED_FROZEN_COHORT").sum()),
        "covered_backfill": int((matrix["tickflow_status"] == "COVERED_BACKFILL").sum()),
        "tickflow_empty": int((matrix["tickflow_status"] == "EMPTY_PROVIDER_RESPONSE").sum()),
        "not_probed": int((matrix["tickflow_status"] == "NOT_PROBED").sum()),
        "blocked_by_data": int(matrix["blocked_reason"].str.startswith("BLOCKED_BY_DATA").sum()),
        "coverage_backlog_fetchable": int(matrix["blocked_reason"].str.startswith("COVERAGE_BACKLOG").sum()),
        "partial_coverage": int(matrix["blocked_reason"].str.startswith("PARTIAL_COVERAGE").sum()),
        "retry_class_counts": matrix["provider_retry_class"].value_counts().to_dict(),
        "by_board_total": matrix["board"].value_counts().to_dict(),
        "by_board_covered": matrix.loc[covered_mask, "board"].value_counts().to_dict(),
        "by_status_total": matrix["current_status"].value_counts().to_dict(),
        "by_status_covered": matrix.loc[covered_mask, "current_status"].value_counts().to_dict(),
        "adjustment_method": ADJUSTMENT_METHOD, "volume_unit": VOLUME_UNIT, "amount_unit": AMOUNT_UNIT,
        "master_source_hash": master_hash,
        "provider_entitlements": {
            "tickflow": "daily bars (entitled); no batch/minute/L2/financials",
            "akshare": "fallback daily bars (not exercised for this cohort)",
            "qlib": "local historical library (not exercised for this cohort)",
            "tushare": "PIT fundamentals/metadata only (not bars)",
        },
        "blinding": "no candidate performance included",
    }
    (OUT / "provider_coverage_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    s = build()
    print(json.dumps({k: s[k] for k in (
        "master_securities", "covered_bar_history", "covered_backfill",
        "tickflow_empty", "blocked_by_data", "by_board_covered")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
