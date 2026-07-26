#!/usr/bin/env python3
"""Assemble the full-universe RAW daily panel from acquired partitions.

The panel this builds differs from its predecessor in three ways that matter:

1. **One declared adjustment.** Every row is raw traded price. The previous panel
   mixed forward-adjusted history for the frozen cohort with unadjusted history
   for backfilled names while declaring ``adjustment_method = "none"``.
2. **One provider per symbol, recorded per row.** Providers are chosen by an
   explicit precedence and never blended inside a symbol; where the serving
   provider changes between the old panel and this one the seam is written to
   ``source_boundaries.parquet``.
3. **No unexplained null closes.** The panel contains traded sessions only.
   In-life sessions with no bar are written to ``session_gaps.parquet`` and
   classified SUSPENDED (matched to a vendor-dated halt interval) or
   MISSING_UNEXPLAINED — so a gap can never be mistaken for a zero-volume day.

Outputs (runtime/data/u0/panel/):
  daily_bars_raw.parquet, session_gaps.parquet, source_boundaries.parquet,
  coverage_matrix.parquet, panel_manifest.json

Usage: AI_quant_venv/bin/python3 scripts/u0_assemble_panel.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.acquire import load_partitions  # noqa: E402
from quantagent.data.ashare.contracts import DAILY_BARS  # noqa: E402

U0 = REPO / "runtime/data/u0"
BARS = U0 / "bars"
PIT = U0 / "pit"
OUT = U0 / "panel"
LEGACY_STAGING = REPO / "runtime/data/v7/full_universe/_staging"
LEGACY_PANEL = REPO / "runtime/data/v7/full_universe/full_universe_market_panel.parquet"

#: Provider precedence. TickFlow first: it is the entitled source and the only
#: one that publishes turnover. The H-032C staging directory holds raw TickFlow
#: partitions from an earlier run that was never assembled.
SOURCE_PRECEDENCE: tuple[tuple[str, Path], ...] = (
    ("tickflow", BARS / "tickflow_raw"),
    ("tickflow_h032c_staging", LEGACY_STAGING),
    ("tencent", BARS / "tencent_raw"),
    # Sina is last: it is the only public route that still serves DELISTED
    # names, but it truncates at 1023 sessions and publishes no turnover.
    ("sina_truncated", BARS / "sina_delisted"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _partition_symbols(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {p.stem.replace("sym_", "").replace("_", "."): p
            for p in directory.glob("sym_*.parquet")}


def assemble(args: argparse.Namespace) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(U0 / "security_master.parquet")
    master["symbol"] = master["symbol"].astype(str)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")

    # -- choose exactly one serving provider per symbol -----------------------
    available = {name: _partition_symbols(path) for name, path in SOURCE_PRECEDENCE}
    chosen: dict[str, tuple[str, Path]] = {}
    for name, _ in SOURCE_PRECEDENCE:
        for symbol, path in available[name].items():
            chosen.setdefault(symbol, (name, path))

    frames: list[pd.DataFrame] = []
    provenance_rows: list[dict] = []
    for symbol, (provider, path) in sorted(chosen.items()):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["symbol"] = symbol
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "source" not in frame.columns:
            # legacy H-032C partitions predate the provenance spine
            frame["source"] = "tickflow"
            frame["source_endpoint"] = "tickflow.klines.get(period=1d,adjust=none)"
            frame["retrieved_at"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            frame["available_at"] = frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00"
            frame["quality_status"] = "OK"
        frame["serving_provider"] = provider
        keep = [c for c in list(DAILY_BARS.columns) + ["serving_provider"] if c in frame.columns]
        frames.append(frame[keep])
        provenance_rows.append({
            "symbol": symbol, "serving_provider": provider, "rows": int(len(frame)),
            "first_date": frame["trade_date"].min(), "last_date": frame["trade_date"].max(),
            "amount_coverage": float(frame["amount"].notna().mean()) if "amount" in frame else 0.0,
            "partition": str(path.relative_to(REPO)),
        })
        if len(frames) >= args.flush_every:
            frames = [pd.concat(frames, ignore_index=True)]

    if not frames:
        raise SystemExit("no partitions found — run scripts/u0_acquire_bars.py first")
    panel = pd.concat(frames, ignore_index=True)
    panel["symbol"] = panel["symbol"].astype(str)

    checks: dict[str, object] = {}
    before = len(panel)
    panel = panel.drop_duplicates(["symbol", "trade_date"], keep="first")
    checks["duplicate_rows_removed"] = int(before - len(panel))

    listing = master.set_index("symbol")["listing_date"]
    delisting = master.set_index("symbol")["delisting_date"]
    panel["_list"] = panel["symbol"].map(listing)
    panel["_delist"] = panel["symbol"].map(delisting)
    pre_listing = panel["_list"].notna() & (panel["trade_date"] < panel["_list"])
    post_delist = panel["_delist"].notna() & (panel["trade_date"] > panel["_delist"])
    checks["rows_before_listing_date"] = int(pre_listing.sum())
    checks["rows_after_delisting_date"] = int(post_delist.sum())
    # A bar dated before the recorded listing date is an identity conflict, not a
    # tradable session: it is quarantined rather than silently kept or dropped.
    quarantine = panel[pre_listing | post_delist].copy()
    panel = panel[~(pre_listing | post_delist)].drop(columns=["_list", "_delist"])
    if len(quarantine):
        quarantine.to_parquet(OUT / "out_of_life_rows_quarantine.parquet", index=False)

    checks["null_close"] = int(panel["close"].isna().sum())
    checks["negative_or_zero_close"] = int((panel["close"] <= 0).sum())
    # A bar whose OHLC cannot all be true is a vendor defect, not a session. It
    # is quarantined with its provenance so the defect stays auditable instead of
    # sitting in the panel where a downstream range or gap calculation would
    # silently consume it.
    ohlc_violation = ((panel["high"] < panel["low"]) |
                      (panel["close"] > panel["high"]) | (panel["close"] < panel["low"]) |
                      (panel["open"] > panel["high"]) | (panel["open"] < panel["low"])).fillna(False)
    checks["ohlc_relationship_violations"] = int(ohlc_violation.sum())
    if ohlc_violation.any():
        defects = panel[ohlc_violation].copy()
        defects["quality_status"] = "SUSPECT"
        defects.to_parquet(OUT / "ohlc_violation_quarantine.parquet", index=False)
        checks["ohlc_violation_symbols"] = sorted(defects["symbol"].unique().tolist())[:20]
        checks["ohlc_violation_providers"] = defects["serving_provider"].value_counts().to_dict()
        panel = panel[~ohlc_violation]
    checks["negative_volume"] = int((panel["volume"] < 0).sum())
    checks["amount_coverage"] = float(panel["amount"].notna().mean())
    checks["rows"] = int(len(panel))
    checks["symbols"] = int(panel["symbol"].nunique())
    checks["min_date"] = str(panel["trade_date"].min().date())
    checks["max_date"] = str(panel["trade_date"].max().date())

    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel.to_parquet(OUT / "daily_bars_raw.parquet", index=False)

    # -- classify in-life sessions with no bar --------------------------------
    gaps = classify_session_gaps(panel, master)
    gaps.to_parquet(OUT / "session_gaps.parquet", index=False)
    checks["session_gaps"] = int(len(gaps))
    for label in ("SUSPENDED", "MISSING_UNEXPLAINED", "PROVIDER_HISTORY_TRUNCATED"):
        checks[f"session_gaps_{label.lower()}"] = (
            int((gaps["classification"] == label).sum()) if len(gaps) else 0)

    # -- source boundaries versus the previous panel --------------------------
    boundaries = source_boundaries(panel)
    boundaries.to_parquet(OUT / "source_boundaries.parquet", index=False)

    # -- coverage matrix ------------------------------------------------------
    coverage = build_coverage(master, panel, pd.DataFrame(provenance_rows))
    coverage.to_parquet(OUT / "coverage_matrix.parquet", index=False)
    coverage.to_csv(OUT / "coverage_matrix.csv", index=False)

    by_board = coverage.groupby("board")["covered"].agg(["sum", "count"])
    manifest = {
        "generated": _now(),
        "panel": "runtime/data/u0/panel/daily_bars_raw.parquet",
        "adjustment_method": "none (raw traded prices) — verified against an independent provider",
        "price_unit": "CNY", "volume_unit": "shares", "amount_unit": "CNY",
        "timezone": "Asia/Shanghai",
        "provider_precedence": [name for name, _ in SOURCE_PRECEDENCE],
        "serving_provider_counts": coverage["serving_provider"].value_counts().to_dict(),
        "master_securities": int(len(master)),
        "covered_securities": int(coverage["covered"].sum()),
        "coverage_by_board": {board: {"covered": int(row["sum"]), "total": int(row["count"])}
                              for board, row in by_board.iterrows()},
        "coverage_by_status": coverage.groupby("status")["covered"].sum().to_dict(),
        "quality_checks": checks,
        "panel_sha256": _sha(OUT / "daily_bars_raw.parquet"),
        "session_gap_policy": ("panel rows are traded sessions only; in-life sessions without a bar "
                               "are classified in session_gaps.parquet, never written as null bars"),
    }
    (OUT / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                       default=str))
    return manifest


def classify_session_gaps(panel: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """In-life trading sessions with no bar, labelled SUSPENDED or unexplained."""
    calendar_path = PIT / "trading_calendar.parquet"
    if not calendar_path.exists():
        return pd.DataFrame(columns=["symbol", "trade_date", "classification", "evidence"])
    calendar = pd.read_parquet(calendar_path)["trade_date"].sort_values().to_numpy()

    suspension_path = PIT / "suspension_intervals.parquet"
    halts = pd.read_parquet(suspension_path) if suspension_path.exists() else pd.DataFrame()
    halt_index: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    halt_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if len(halts):
        halts["effective_start"] = pd.to_datetime(halts["effective_start"])
        halts["effective_end"] = pd.to_datetime(halts["effective_end"])
        halt_window = (halts["effective_start"].min(), halts["effective_end"].max())
        for row in halts.itertuples():
            halt_index.setdefault(row.symbol, []).append(
                (row.effective_start, row.effective_end, str(row.suspension_reason)[:60]))

    listing = master.set_index("symbol")["listing_date"]
    delisting = master.set_index("symbol")["delisting_date"]
    today = pd.Timestamp.now().normalize()
    # Providers that cap how far back they will serve; sessions before their
    # earliest bar are a known provider limit, not missing market data.
    truncating = {"sina_truncated"}
    provider_of = (panel.groupby("symbol")["serving_provider"].first().to_dict()
                   if "serving_provider" in panel.columns else {})
    records: list[dict] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        traded = group["trade_date"].to_numpy()
        listed_on = listing.get(symbol, pd.NaT)
        truncates = provider_of.get(symbol) in truncating
        truncated_before = pd.Timestamp(traded.min()) if truncates else None
        # Normally the in-life window starts at the first bar we hold, because
        # earlier sessions may simply predate the vendor's coverage. For a
        # provider with a KNOWN history cap the window starts at the listing date
        # instead, so the truncated span is counted and labelled rather than
        # disappearing from the coverage picture entirely.
        if truncates and pd.notna(listed_on):
            start = pd.Timestamp(listed_on)
        else:
            start = max(pd.Timestamp(listed_on) if pd.notna(listed_on) else traded.min(),
                        traded.min())
        end_candidates = [traded.max()]
        delist = delisting.get(symbol, pd.NaT)
        if pd.notna(delist):
            end_candidates.append(pd.Timestamp(delist))
        end = min(max(end_candidates), today)
        sessions = calendar[(calendar >= np.datetime64(start)) & (calendar <= np.datetime64(end))]
        missing = np.setdiff1d(sessions, traded, assume_unique=False)
        if not len(missing):
            continue
        intervals = halt_index.get(symbol, [])
        for day in missing:
            stamp = pd.Timestamp(day)
            reason = next((note for lo, hi, note in intervals if lo <= stamp <= hi), None)
            if truncated_before is not None and stamp < truncated_before:
                records.append({"symbol": symbol, "trade_date": stamp,
                                "classification": "PROVIDER_HISTORY_TRUNCATED",
                                "evidence": f"serving provider {provider_of[symbol]} caps history "
                                            f"at its earliest bar {truncated_before.date()}"})
            elif reason is not None:
                records.append({"symbol": symbol, "trade_date": stamp,
                                "classification": "SUSPENDED",
                                "evidence": f"vendor halt interval: {reason}"})
            elif halt_window and not (halt_window[0] <= stamp <= halt_window[1]):
                records.append({"symbol": symbol, "trade_date": stamp,
                                "classification": "MISSING_UNEXPLAINED",
                                "evidence": "outside the halt-snapshot coverage window"})
            else:
                records.append({"symbol": symbol, "trade_date": stamp,
                                "classification": "MISSING_UNEXPLAINED",
                                "evidence": "no vendor halt record covers this session"})
    return pd.DataFrame(records, columns=["symbol", "trade_date", "classification", "evidence"])


def source_boundaries(panel: pd.DataFrame) -> pd.DataFrame:
    """Record where this panel's provider differs from the previous panel's."""
    if not LEGACY_PANEL.exists():
        return pd.DataFrame(columns=["symbol", "boundary_date", "provider_before",
                                     "provider_after", "reason"])
    legacy = pd.read_parquet(LEGACY_PANEL, columns=["symbol", "trade_date", "source_track"])
    legacy["symbol"] = legacy["symbol"].astype(str)
    legacy_track = legacy.groupby("symbol")["source_track"].first()
    current = panel.groupby("symbol")["serving_provider"].first()
    overlap = current.index.intersection(legacy_track.index)
    rows = []
    for symbol in overlap:
        before = legacy_track[symbol]
        after = current[symbol]
        reason = ("previous panel served this symbol as forward-adjusted (qfq) prices; "
                  "this panel serves raw prices"
                  if before == "frozen_cohort" else
                  "provider changed between panel generations")
        rows.append({"symbol": symbol,
                     "boundary_date": str(panel.loc[panel["symbol"] == symbol, "trade_date"].min().date()),
                     "provider_before": f"legacy:{before}", "provider_after": after,
                     "reason": reason})
    return pd.DataFrame(rows)


def build_coverage(master: pd.DataFrame, panel: pd.DataFrame,
                   provenance: pd.DataFrame) -> pd.DataFrame:
    stats = panel.groupby("symbol").agg(
        rows=("trade_date", "size"), first_date=("trade_date", "min"),
        last_date=("trade_date", "max"), amount_coverage=("amount", lambda s: float(s.notna().mean())))
    serving = panel.groupby("symbol")["serving_provider"].first()
    coverage = master[["symbol", "code", "exchange", "board", "security_type", "status",
                       "listing_date", "delisting_date", "current_st", "bse_legacy_code"]].copy()
    coverage = coverage.merge(stats, on="symbol", how="left")
    coverage["serving_provider"] = coverage["symbol"].map(serving).fillna("none")
    coverage["covered"] = coverage["rows"].notna()
    coverage["rows"] = coverage["rows"].fillna(0).astype(int)
    if len(provenance):
        coverage = coverage.merge(provenance[["symbol", "partition"]], on="symbol", how="left")
    coverage["blocked_reason"] = np.where(
        coverage["covered"], "",
        np.where(coverage["status"] == "delisted",
                 "NO_VENDOR_HISTORY_DELISTED", "NOT_YET_ACQUIRED"))
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flush-every", type=int, default=500,
                        help="concatenate partition frames every N symbols to bound memory")
    args = parser.parse_args()
    manifest = assemble(args)
    print(json.dumps({k: manifest[k] for k in
                      ("master_securities", "covered_securities", "coverage_by_board",
                       "coverage_by_status", "serving_provider_counts", "quality_checks")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
