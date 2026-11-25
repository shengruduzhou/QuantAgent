#!/usr/bin/env python3
"""Chunked, resumable PIT-safe market-panel catch-up.

TickFlow supplies historical OHLCV. Historical ST/risk-warning state must come
from a separately versioned dated PIT table; current instrument-name snapshots
are rejected. Staging is window-scoped and the final merge recomputes all
tradability flags for the catch-up window using the dated ST state.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import repair_fresh_window_20260704 as rep  # noqa: E402

from quantagent.data.providers.st_pit import (  # noqa: E402
    HistoricalSTCoverageError,
    attach_historical_st,
    default_historical_st_path,
    load_historical_st_evidence,
    require_symbol_coverage,
)
from quantagent.quant_math.ashare import board_price_limit_vector  # noqa: E402


PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
STAGING = PANEL.parent / "_staging_catchup"
CALENDAR = REPO / "runtime/data/u0/pit/trading_calendar.parquet"
MASTER = REPO / "runtime/data/u0/security_master.parquet"
DELISTINGS = REPO / "runtime/data/u0/pit/delisting_intervals.parquet"
SUSPENSION_DIRS = [REPO / "runtime/data/u0/pit", REPO / "runtime/data/u0/pit/suspension_fresh"]
# Frozen FRESH cohort: names printing in the 30 days before the freeze (3,653 =
# the freeze manifest's seed). Immutable pre-window rows, so it cannot ratchet.
COHORT_START = pd.Timestamp("2026-05-19")
COHORT_LOOKBACK_DAYS = 30
CHUNK = 250
# Extra bars beyond the window's session count (vendor returns the latest N bars).
FETCH_COUNT_MARGIN = 10
# Every session in the window must carry at least this share of the seed symbols
# (same threshold as scripts/coverage_guard.sh) before the panel is written.
MIN_DATE_COVERAGE = 0.93
# Seed names the vendor answered with no in-window bar and no dated halt covers
# (Cycle 32 review): recorded here at merge time and read by the FRESH integrity
# report; they do not block the merge and are re-asked every run (master seed).
VENDOR_EMPTY_NAME = "vendor_empty_no_evidence.json"
# Exit code for "fetch incomplete, retry": the supervisor re-enters instead of aborting.
EXIT_RETRY = 75


class WindowCoverageError(RuntimeError):
    """The staged catch-up rows do not cover the window; the panel must not be written."""


def window_sessions(calendar: pd.DataFrame, win_start, win_end) -> list[pd.Timestamp]:
    """Open sessions of the offline PIT trading calendar inside [win_start, win_end].

    Refuses a window that extends beyond the calendar's last date (an expired
    calendar must be refreshed, not silently treated as having no sessions)."""
    dates = pd.to_datetime(calendar["trade_date"])
    if "is_open" in calendar.columns:
        dates = dates[calendar["is_open"].astype(bool).to_numpy()]
    dates = dates.dt.normalize()
    if pd.Timestamp(win_end) > dates.max():
        raise WindowCoverageError(
            f"trading calendar ends {dates.max().date()} before window end "
            f"{pd.Timestamp(win_end).date()}; refresh runtime/data/u0/pit/trading_calendar.parquet"
        )
    mask = (dates >= pd.Timestamp(win_start)) & (dates <= pd.Timestamp(win_end))
    return sorted(pd.Timestamp(d) for d in dates[mask].unique())


def require_window_coverage(
    new: pd.DataFrame,
    sessions: list[pd.Timestamp],
    n_seed: int,
    min_ratio: float = MIN_DATE_COVERAGE,
) -> dict[str, int]:
    """Fail closed unless every calendar session in the window is present with at
    least ``min_ratio`` of the seed symbols and no row falls on a non-session date.

    Cycle 32: the catch-up fetched the latest 40 bars for a 73-session window, which
    would have appended 2026-09-22.. while leaving 2026-08-06..2026-09-21 empty; the
    supervisor's max(trade_date) check would still have reported CURRENT."""
    if not sessions:
        raise WindowCoverageError("window contains no calendar sessions")
    trade_dates = pd.to_datetime(new["trade_date"]).dt.normalize()
    per_date = new.assign(_d=trade_dates).groupby("_d")["symbol"].nunique()
    session_set = set(sessions)
    off_calendar = sorted(d for d in per_date.index if d not in session_set)
    if off_calendar:
        raise WindowCoverageError(
            f"{len(off_calendar)} staged dates are not calendar sessions: "
            f"{[str(d.date()) for d in off_calendar[:5]]}"
        )
    floor = int(np.ceil(min_ratio * n_seed))
    short = {
        str(d.date()): int(per_date.get(d, 0))
        for d in sessions
        if int(per_date.get(d, 0)) < floor
    }
    if short:
        first = list(short.items())[:5]
        raise WindowCoverageError(
            f"{len(short)}/{len(sessions)} sessions below {floor} symbols "
            f"({min_ratio:.0%} of {n_seed} seeds), e.g. {first}"
        )
    return {str(d.date()): int(per_date[d]) for d in sessions}


def tradability_flags(target: pd.DataFrame) -> pd.DataFrame:
    """Limit/suspension flags from raw close, prev_close and dated is_st.

    Shared by the catch-up merge and the trailing-symbol backfill so both write
    identical flag semantics."""
    target = target.copy()
    ratios = board_price_limit_vector(
        target["symbol"].astype(str),
        target["is_st"].astype(bool),
        trade_dates=target["trade_date"],
    )
    up_price = (target["prev_close"] * (1.0 + ratios)).round(2)
    down_price = (target["prev_close"] * (1.0 - ratios)).round(2)
    target["is_limit_up"] = (
        (target["close"].round(2) >= up_price - 0.005)
        & target["prev_close"].notna()
    )
    target["is_limit_down"] = (
        (target["close"].round(2) <= down_price + 0.005)
        & target["prev_close"].notna()
    )
    target["is_suspended"] = target["volume"].fillna(0) <= 0
    return target


def dated_delistings(master: pd.DataFrame, delistings: pd.DataFrame | None) -> pd.Series:
    """symbol -> latest dated delisting (master column + U0 delisting table)."""
    parts = [pd.to_datetime(master.set_index(master["symbol"].astype(str))["delisting_date"],
                            errors="coerce")]
    if delistings is not None and len(delistings):
        parts.append(pd.to_datetime(
            delistings.set_index(delistings["symbol"].astype(str))["effective_start"],
            errors="coerce"))
    dated = pd.concat(parts).dropna()
    return dated.groupby(level=0).max()


def seed_universe(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    delistings: pd.DataFrame | None,
    win_start,
    win_end,
    *,
    mode: str = "cohort",
) -> tuple[list[str], dict]:
    """Symbols the window must cover, from the security master -- never from the
    names that happened to print on the panel's last date (Cycle 32, audit F2:
    that seed ratcheted 3,584 -> 3,447 and dropped every name halted on a seed
    date for good).

    ``mode="cohort"``: master-listed names of the frozen FRESH cohort.
    ``mode="master"``: every master-listed SH/SZ name (adds names never in the
    FRESH universe -- a protocol change, opt-in only).
    Both exclude names whose dated delisting precedes the window.
    """
    win_start, win_end = pd.Timestamp(win_start), pd.Timestamp(win_end)
    uni = master[["symbol", "listing_date"]].copy()
    uni["symbol"] = uni["symbol"].astype(str)
    uni = uni[uni["symbol"].str[-2:].isin(["SH", "SZ"])]
    listing = pd.to_datetime(uni["listing_date"], errors="coerce")
    report: dict = {"mode": mode, "master_rows": int(len(uni))}
    listed = uni[listing.notna() & (listing <= win_end)]
    report["master_without_listing_date"] = int(listing.isna().sum())
    if mode == "cohort":
        dates = pd.to_datetime(panel["trade_date"])
        cohort = set(panel.loc[(dates < COHORT_START)
                               & (dates >= COHORT_START - pd.Timedelta(days=COHORT_LOOKBACK_DAYS)),
                               "symbol"].astype(str))
        if not cohort:
            raise SystemExit("refusing catch-up: FRESH cohort is empty (panel lacks pre-freeze rows)")
        report["cohort"] = len(cohort)
        report["cohort_not_in_master"] = sorted(cohort - set(uni["symbol"]))
        listed = listed[listed["symbol"].isin(cohort)]
    elif mode != "master":
        raise ValueError(f"unknown seed mode {mode!r}")
    dated = dated_delistings(master, delistings)
    gone = listed["symbol"].map(dated) < win_start
    report["excluded_dated_delisting"] = int(gone.sum())
    seed = sorted(listed.loc[~gone, "symbol"])
    report["seed"] = len(seed)
    return seed, report


def window_halt_covered(symbol: str, sessions, intervals: pd.DataFrame) -> bool:
    """True when dated halt intervals cover every window session for ``symbol``."""
    if intervals is None or not len(intervals) or not len(sessions):
        return False
    own = intervals[intervals["symbol"].astype(str) == symbol]
    if own.empty:
        return False
    starts = pd.to_datetime(own["effective_start"]).dt.normalize()
    ends = pd.to_datetime(own["effective_end"]).dt.normalize()
    return all(bool(((starts <= d) & (ends >= d)).any()) for d in sessions)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write to a sibling temp file, fsync, then rename over ``path``: a timeout
    (SIGTERM/SIGKILL) during the write can no longer truncate the panel."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _acquire_staging_lock(staging: Path):
    """Single writer for the staging directory (the 16:30 fresh_blind_daily top-up
    and the 06:00 supervisor both run this script)."""
    staging.mkdir(exist_ok=True)
    handle = open(staging / ".catchup.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _load_required_st_evidence(path: str | None):
    if not path:
        raise SystemExit(
            "historical ST PIT evidence is required; set --historical-st-path or "
            "QUANTAGENT_HISTORICAL_ST_PATH, or publish the ENG_ST30 register. Current snapshot broadcast is forbidden."
        )
    try:
        return load_historical_st_evidence(path)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(f"historical ST PIT evidence invalid: {exc}") from exc


def _require_st_coverage(evidence, symbols, win_start, win_end) -> None:
    """Refuse the catch-up before any vendor call when the dated ST evidence
    cannot cover every seed symbol for the whole window (the merge step would
    refuse those rows anyway, after hours of fetching)."""
    try:
        require_symbol_coverage(evidence, symbols, win_start, win_end)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(f"historical ST evidence does not cover the catch-up window: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=None, help="last date to fetch (default: today)")
    parser.add_argument(
        "--historical-st-path",
        default=default_historical_st_path(),
        help="dated PIT ST table (parquet/csv/jsonl); required",
    )
    parser.add_argument(
        "--seed-universe", choices=["cohort", "master"], default="cohort",
        help="cohort = master-listed names of the frozen FRESH cohort (default); "
             "master = every master-listed SH/SZ name (protocol change, opt-in)",
    )
    args = parser.parse_args()
    evidence = _load_required_st_evidence(args.historical_st_path)
    started = time.time()
    lock = _acquire_staging_lock(STAGING)
    if lock is None:
        print(json.dumps({"appended": 0, "note": "another catch-up holds the staging lock; skipped"}))
        return 0
    for orphan in PANEL.parent.glob(f".{PANEL.name}.tmp-*"):
        orphan.unlink()  # left by a SIGKILL during a previous atomic write; we hold the lock

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    for column in ("is_st_provenance", "st_evidence_sha256"):
        if column not in panel.columns:
            panel[column] = pd.NA

    panel_max = panel["trade_date"].max()
    win_start = panel_max + pd.Timedelta(days=1)
    win_end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()

    cst_now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    last_available = (
        cst_now.normalize()
        if cst_now.hour * 60 + cst_now.minute >= 16 * 60
        else cst_now.normalize() - pd.Timedelta(days=1)
    )
    if win_end > last_available:
        print(
            f"clamping window end {win_end.date()} -> {last_available.date()} "
            f"(close not published yet at {cst_now:%Y-%m-%d %H:%M} CST)",
            flush=True,
        )
        win_end = last_available
    if win_start > win_end:
        print(json.dumps({"appended": 0, "note": "panel already current"}))
        return 0
    if len(pd.bdate_range(win_start, win_end)) == 0:
        print(json.dumps({"appended": 0, "note": "no business day in window"}))
        return 0

    if not MASTER.exists():
        raise SystemExit(f"refusing catch-up: security master {MASTER} missing (seed source)")
    master = pd.read_parquet(MASTER)
    delistings = pd.read_parquet(DELISTINGS) if DELISTINGS.exists() else None
    seed_symbols, seed_report = seed_universe(panel, master, delistings, win_start, win_end,
                                              mode=args.seed_universe)
    # Names the dated ST evidence cannot attribute are not fetched (appending them
    # would broadcast ST state); they are named here and stay visible as gaps in
    # the FRESH integrity report -- never silently dropped.
    st_covered = set(evidence.frame["symbol"].astype(str))
    no_st = sorted(set(seed_symbols) - st_covered)
    seed_symbols = sorted(set(seed_symbols) & st_covered)
    seed_report["excluded_no_dated_st_evidence"] = no_st
    print(json.dumps({"seed": seed_report}, default=str), flush=True)
    _require_st_coverage(evidence, seed_symbols, win_start, win_end)
    try:
        sessions = window_sessions(pd.read_parquet(CALENDAR), win_start, win_end)
    except WindowCoverageError as exc:
        raise SystemExit(f"refusing catch-up: {exc}") from exc
    if not sessions:
        print(json.dumps({"appended": 0, "note": "no calendar session in window"}))
        return 0
    fetch_count = len(sessions) + FETCH_COUNT_MARGIN
    print(f"window has {len(sessions)} calendar sessions -> fetch count {fetch_count}", flush=True)

    manifest = STAGING / "window.json"
    window_tag = f"{win_start.date()}_{win_end.date()}"
    expected_manifest = {
        "window": window_tag,
        "historical_st_evidence_sha256": evidence.source_sha256,
        "fetch_count": fetch_count,
    }
    existing_manifest = {}
    if manifest.exists():
        try:
            existing_manifest = json.loads(manifest.read_text())
        except Exception:
            existing_manifest = {}
    if existing_manifest != expected_manifest:
        for path in list(STAGING.glob("chunk_*.parquet")) + list(
            STAGING.glob("done_*.json")
        ) + list(STAGING.glob("empty_*.json")):
            path.unlink()
        manifest.write_text(json.dumps(expected_manifest, sort_keys=True))

    done_symbols: set[str] = set()
    for path in sorted(STAGING.glob("done_*.json")):
        try:
            done_symbols |= set(json.loads(path.read_text()))
        except Exception:
            path.unlink()
    for path in sorted(STAGING.glob("chunk_*.parquet")):
        try:
            done_symbols |= set(
                pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str)
            )
        except Exception:
            path.unlink()

    todo = [symbol for symbol in seed_symbols if symbol not in done_symbols]
    print(
        f"window {win_start.date()}..{win_end.date()} | seed {len(seed_symbols)} "
        f"| staged {len(done_symbols)} | todo {len(todo)}",
        flush=True,
    )

    from quantagent.data.ashare.session_gaps import load_suspension_evidence

    halt_intervals, _ = load_suspension_evidence(SUSPENSION_DIRS)
    tf = rep._tf_client()
    failed: list[str] = []
    empty_unexplained: list[str] = []
    for chunk_start in range(0, len(todo), CHUNK):
        chunk_symbols = todo[chunk_start : chunk_start + CHUNK]
        rows: list[pd.DataFrame] = []
        for symbol in chunk_symbols:
            kline = rep.fetch_with_retry(tf, symbol, count=fetch_count)
            if kline is None:
                failed.append(symbol)
                continue
            in_window = (len(kline) > 0) and bool(
                ((pd.to_datetime(kline["trade_date"]) >= win_start)
                 & (pd.to_datetime(kline["trade_date"]) <= win_end)).any())
            if not in_window:
                # A SUCCESSFUL empty answer (not a failure: those are None above
                # and retried). Without a dated halt it is unexplained: recorded
                # explicitly, never silently "done", and it does not block the
                # merge -- the master seed asks again next run, so no ratchet.
                if not window_halt_covered(symbol, sessions, halt_intervals):
                    empty_unexplained.append(symbol)
                continue
            kline = kline.copy()
            kline["symbol"] = symbol
            kline["trade_date"] = pd.to_datetime(kline["trade_date"])
            kline = kline[
                (kline["trade_date"] >= win_start)
                & (kline["trade_date"] <= win_end)
            ]
            if len(kline):
                kline["volume"] = pd.to_numeric(
                    kline["volume"], errors="coerce"
                ) * 100.0
                rows.append(
                    kline[
                        [
                            "symbol",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "amount",
                        ]
                    ]
                )
        staged = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                ]
            )
        )
        chunk_failed = set(failed) & set(chunk_symbols)
        chunk_empty = sorted(set(empty_unexplained) & set(chunk_symbols))
        if chunk_empty:
            (STAGING / f"empty_{chunk_start // CHUNK:04d}_{time.time_ns()}.json").write_text(
                json.dumps(chunk_empty))
        (STAGING / f"done_{chunk_start // CHUNK:04d}_{time.time_ns()}.json").write_text(
            json.dumps(sorted(set(chunk_symbols) - chunk_failed))
        )
        staged.to_parquet(
            STAGING / f"chunk_{chunk_start // CHUNK:04d}_{time.time_ns()}.parquet",
            index=False,
        )
        print(
            f"  staged {chunk_start + len(chunk_symbols)}/{len(todo)} "
            f"(failed {len(failed)}) {time.time() - started:.0f}s",
            flush=True,
        )

    # Seed ratchet guard (Cycle 32): a symbol whose fetch failed (vendor throttling)
    # is not marked done; merging without it would drop it from the panel's last
    # date and therefore from every later seed list, permanently. Refuse the merge
    # until every seed symbol has been answered. A failed request is retried
    # (exit 75); a successful empty answer without a dated halt is recorded as
    # vendor_empty_no_evidence and does not block (Cycle 32 review).
    fetched: set[str] = set()
    for path in sorted(STAGING.glob("done_*.json")):
        fetched |= set(json.loads(path.read_text()))
    vendor_empty: set[str] = set()
    for path in sorted(STAGING.glob("empty_*.json")):
        vendor_empty |= set(json.loads(path.read_text()))
    unfetched = sorted(set(seed_symbols) - fetched)
    if unfetched:
        print(
            json.dumps(
                {
                    "appended": 0,
                    "status": "FETCH_INCOMPLETE_RETRY",
                    "n_unfetched": len(unfetched),
                    "unfetched_sample": unfetched[:20],
                    "vendor_empty_no_evidence": sorted(vendor_empty)[:50],
                    "note": "panel not written; staging kept, failed symbols are retried",
                }
            ),
            flush=True,
        )
        return EXIT_RETRY

    chunks = [pd.read_parquet(path) for path in sorted(STAGING.glob("chunk_*.parquet"))]
    new = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(new):
        new = new.drop_duplicates(["symbol", "trade_date"], keep="first")
    if not len(new):
        for path in list(STAGING.glob("chunk_*.parquet")) + list(
            STAGING.glob("done_*.json")
        ) + list(STAGING.glob("empty_*.json")):
            path.unlink()
        manifest.unlink(missing_ok=True)
        print(
            json.dumps(
                {"appended": 0, "n_failed": len(failed), "note": "no new rows fetched",
                 "vendor_empty_no_evidence": sorted(vendor_empty)}
            )
        )
        return 0

    for column in ("open", "high", "low", "close", "volume", "amount"):
        new[column] = pd.to_numeric(new[column], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    try:
        # Denominator = seed names that returned data: the 93% guard still
        # catches a latest-N hole, while dark / halted names cannot fail it.
        returned = int(new["symbol"].nunique())
        require_window_coverage(new, sessions, returned)
    except WindowCoverageError as exc:
        raise SystemExit(
            f"refusing catch-up merge (staging preserved for resume): {exc}"
        ) from exc
    # ...and against the FULL seed: a partial vendor publish (INC-P1: the day's
    # bar out for only some names) or a flood of empty answers must not merge a
    # thin session -- panel_max would advance and the hole become permanent.
    # 93% of 3,643 tolerates <=255 vendor-empty names. Retry, panel unchanged.
    try:
        require_window_coverage(new, sessions, len(seed_symbols))
    except WindowCoverageError as exc:
        print(json.dumps({"appended": 0, "status": "SEED_COVERAGE_RETRY",
                          "seed": len(seed_symbols), "returned_data": returned,
                          "vendor_empty_no_evidence": len(vendor_empty),
                          "detail": str(exc)[:400],
                          "note": "panel not written; staging kept"}), flush=True)
        return EXIT_RETRY
    try:
        new = attach_historical_st(new, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing catch-up merge because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc

    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_catchup_chunked"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    for column in panel.columns:
        if column not in new.columns:
            new[column] = np.nan
    new = new[list(panel.columns)]

    backup = PANEL.with_name(f"market_panel.pre_catchup_{win_start.date()}.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= panel_max - pd.Timedelta(days=5)].to_parquet(
            backup, index=False
        )
    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")

    window_mask = (merged["trade_date"] >= win_start) & (
        merged["trade_date"] <= win_end
    )
    chain = merged[
        (merged["trade_date"] >= panel_max - pd.Timedelta(days=5))
        & (merged["trade_date"] <= win_end)
    ][["symbol", "trade_date", "close", "volume"]].sort_values(
        ["symbol", "trade_date"]
    ).copy()
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    target = chain[chain["trade_date"] >= win_start].copy()
    try:
        target = attach_historical_st(target, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing flag rebuild because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc
    target = tradability_flags(target)
    flags = target.set_index(["symbol", "trade_date"])[
        [
            "is_limit_up",
            "is_limit_down",
            "is_suspended",
            "is_st",
            "is_st_provenance",
            "st_evidence_sha256",
        ]
    ]
    index = merged.loc[window_mask].set_index(["symbol", "trade_date"]).index
    missing_flags = ~index.isin(flags.index)
    if bool(missing_flags.any()):
        raise SystemExit(
            f"flag rebuild missing {int(missing_flags.sum())} catch-up rows; refusing panel write"
        )
    for column in (
        "is_limit_up",
        "is_limit_down",
        "is_suspended",
        "is_st",
        "is_st_provenance",
        "st_evidence_sha256",
    ):
        merged.loc[window_mask, column] = flags[column].reindex(index).to_numpy()

    merged = merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    _atomic_write_parquet(merged, PANEL)
    empty_record = PANEL.parent / VENDOR_EMPTY_NAME
    tmp = empty_record.with_name(f".{empty_record.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps({"window": window_tag, "classification": "MISSING_UNEXPLAINED",
                               "evidence": "vendor_empty_no_evidence",
                               "symbols": sorted(vendor_empty)}, indent=1))
    os.replace(tmp, empty_record)
    for path in list(STAGING.glob("chunk_*.parquet")) + list(
        STAGING.glob("done_*.json")
    ) + list(STAGING.glob("empty_*.json")):
        path.unlink()
    manifest.unlink(missing_ok=True)

    window = merged[window_mask]
    report = {
        "appended": int(len(new)),
        "n_failed": len(failed),
        "failed_sample": failed[:20],
        "new_max_date": str(merged["trade_date"].max().date()),
        "dates_added": [
            str(date.date()) for date in sorted(window["trade_date"].unique())
        ],
        "coverage_per_date": {
            str(date.date()): int(value)
            for date, value in window.groupby("trade_date")["symbol"].nunique().items()
        },
        "seed": len(seed_symbols),
        "seed_returned_data": returned,
        "min_coverage_vs_full_seed": round(
            float(window.groupby("trade_date")["symbol"].nunique().min()) / len(seed_symbols), 6)
        if len(window) and seed_symbols else None,
        "vendor_empty_no_evidence": sorted(vendor_empty),
        "historical_st_evidence_sha256": evidence.source_sha256,
        "historical_st_evidence_mode": evidence.mode,
        "runtime_s": round(time.time() - started, 1),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
