#!/usr/bin/env python3
"""Backfill FRESH cohort names that stopped printing before the panel max.

The old catch-up seed (names printed on the panel's last date) dropped ~383
cohort names for good (audit F2). This script brings each non-delisted one back
from its last bar to the panel max, through the same vendor call, ST attach,
flag derivation and atomic write as ``catchup_panel_chunked.py``.

Stages (each opt-in; the default does nothing but plan):

* plan (default, no network): which names, how many sessions, requests and an
  ETA at the measured TickFlow limit of 10 requests/minute;
* ``--fetch --allow-network``: one request per name into ``--staging``
  (resumable, per-symbol parquet, tmp + os.replace);
* verify (always, on what is staged): every session in ``(last bar, panel
  max]`` must have a bar or sit in an unbroken dated halt run from the last
  bar; only fully verified names are written;
* ``--output PATH``: a new panel version + manifest (never the live path);
* ``--apply``: the live panel, atomically, under the catch-up staging lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import catchup_panel_chunked as cu  # noqa: E402
from quantagent.data.ashare.session_gaps import load_suspension_evidence  # noqa: E402
from quantagent.data.providers.st_pit import attach_historical_st  # noqa: E402

REQUESTS_PER_MINUTE = 10
FETCH_MARGIN = 10


def plan_backfill(panel, master, delistings, calendar, evidence_frame) -> tuple[pd.DataFrame, dict]:
    """One row per trailing cohort name: last bar, sessions to fill, vendor count."""
    panel_max = pd.to_datetime(panel["trade_date"]).max()
    seed, seed_report = cu.seed_universe(panel, master, delistings, panel_max, panel_max)
    last = panel.groupby(panel["symbol"].astype(str))["trade_date"].max()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"]).dt.normalize().unique()))
    today = pd.Timestamp.now().normalize()
    st_symbols = set(evidence_frame["symbol"].astype(str))
    rows = []
    for symbol in seed:
        if symbol not in last.index or last[symbol] >= panel_max:
            continue
        lb = pd.Timestamp(last[symbol]).normalize()
        need = sessions[(sessions > lb) & (sessions <= panel_max)]
        to_today = sessions[(sessions > lb) & (sessions <= today)]
        rows.append({"symbol": symbol, "last_bar": lb, "sessions_to_fill": len(need),
                     "fetch_count": len(to_today) + FETCH_MARGIN,
                     "st_evidence": symbol in st_symbols})
    plan = pd.DataFrame(rows, columns=["symbol", "last_bar", "sessions_to_fill",
                                       "fetch_count", "st_evidence"])
    fetchable = plan[plan["st_evidence"]]
    report = {
        "panel_max": str(panel_max.date()),
        "seed": seed_report,
        "trailing_symbols": int(len(plan)),
        "blocked_no_dated_st_evidence": sorted(plan.loc[~plan["st_evidence"], "symbol"]),
        "fetchable_symbols": int(len(fetchable)),
        "session_rows_to_fill": int(fetchable["sessions_to_fill"].sum()),
        "requests": int(len(fetchable)),
        "eta_minutes_at_10_per_min": round(len(fetchable) / REQUESTS_PER_MINUTE, 1),
        "last_bar_distribution": {str(k.date()): int(v) for k, v in
                                  plan["last_bar"].value_counts().head(10).items()},
    }
    return plan, report


def _stage_path(staging: Path, symbol: str) -> Path:
    return staging / f"{symbol}.parquet"


def fetch(plan: pd.DataFrame, staging: Path, panel_max: pd.Timestamp, limit: int) -> dict:
    staging.mkdir(parents=True, exist_ok=True)
    tf = cu.rep._tf_client()
    done = failed = 0
    todo = [r for r in plan[plan["st_evidence"]].itertuples()
            if not _stage_path(staging, r.symbol).exists()]
    if limit:
        todo = todo[:limit]
    for index, row in enumerate(todo):
        kline = cu.rep.fetch_with_retry(tf, row.symbol, count=int(row.fetch_count))
        if kline is None:
            failed += 1
        else:
            k = kline.copy()
            k["symbol"] = row.symbol
            k["trade_date"] = pd.to_datetime(k["trade_date"]).dt.normalize()
            k = k[(k["trade_date"] > row.last_bar) & (k["trade_date"] <= panel_max)]
            k["volume"] = pd.to_numeric(k["volume"], errors="coerce") * 100.0  # 手 -> shares
            out = k[["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]]
            target = _stage_path(staging, row.symbol)
            tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            out.to_parquet(tmp, index=False)
            os.replace(tmp, target)
            done += 1
        if index + 1 < len(todo):
            time.sleep(60.0 / REQUESTS_PER_MINUTE)
    return {"fetched": done, "failed": failed, "attempted": len(todo)}


def verify(plan: pd.DataFrame, staging: Path, calendar, halts: pd.DataFrame,
           panel_max: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Rows of fully verified names + a per-name verdict."""
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"]).dt.normalize().unique()))
    verdicts: dict[str, str] = {}
    good: list[pd.DataFrame] = []
    for row in plan[plan["st_evidence"]].itertuples():
        path = _stage_path(staging, row.symbol)
        if not path.exists():
            verdicts[row.symbol] = "NOT_STAGED"
            continue
        rows = pd.read_parquet(path)
        have = set(pd.to_datetime(rows["trade_date"]))
        need = sessions[(sessions > row.last_bar) & (sessions <= panel_max)]
        off = have - set(need)
        if off or rows["trade_date"].duplicated().any():
            verdicts[row.symbol] = "OFF_CALENDAR_OR_DUPLICATE"
            continue
        missing = [d for d in need if d not in have]
        broken = False
        # a missing session is acceptable only inside a dated halt run that
        # starts right after a bar (no unexplained session before it)
        prev_ok = True
        for day in need:
            if day in have:
                prev_ok = True
                continue
            halted = cu.window_halt_covered(row.symbol, [day], halts)
            if not (halted and prev_ok):
                broken = True
                break
        if broken:
            verdicts[row.symbol] = f"UNEXPLAINED_MISSING_SESSIONS {len(missing)}"
            continue
        verdicts[row.symbol] = "VERIFIED"
        good.append(rows)
    frame = pd.concat(good, ignore_index=True) if good else pd.DataFrame()
    counts = pd.Series(list(verdicts.values())).str.split(" ").str[0].value_counts()
    return frame, {"verdicts": {str(k): int(v) for k, v in counts.items()},
                   "unverified_sample": {k: v for k, v in verdicts.items() if v != "VERIFIED"}
                   if len(verdicts) < 50 else
                   dict(list({k: v for k, v in verdicts.items() if v != "VERIFIED"}.items())[:50]),
                   "verified_rows": int(len(frame))}


def build_panel(panel: pd.DataFrame, new: pd.DataFrame, evidence) -> pd.DataFrame:
    """Merge verified rows with catch-up semantics (ST attach + shared flags)."""
    new = new.copy()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        new[column] = pd.to_numeric(new[column], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    new = attach_historical_st(new, evidence)
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_backfill_trailing"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    symbols = set(new["symbol"])
    prior = (panel[panel["symbol"].isin(symbols)].sort_values("trade_date")
             .groupby("symbol").tail(1)[["symbol", "trade_date", "close"]])
    chain = pd.concat([prior.assign(_new=False), new[["symbol", "trade_date", "close"]]
                       .assign(_new=True)]).sort_values(["symbol", "trade_date"])
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    new = new.merge(chain.loc[chain["_new"], ["symbol", "trade_date", "prev_close"]],
                    on=["symbol", "trade_date"], how="left")
    new = cu.tradability_flags(new).drop(columns=["prev_close"])
    for column in panel.columns:
        if column not in new.columns:
            new[column] = pd.NA
    merged = pd.concat([panel, new[list(panel.columns)]], ignore_index=True)
    if merged.duplicated(["symbol", "trade_date"]).any():
        raise SystemExit("backfill rows collide with existing panel rows; refusing")
    return merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def rebind_repo(root: Path) -> None:
    """Point every catch-up path at another checkout's runtime (read side)."""
    cu.PANEL = root / "runtime/data/v7/silver/market_panel/market_panel.parquet"
    cu.STAGING = cu.PANEL.parent / "_staging_catchup"
    cu.CALENDAR = root / "runtime/data/u0/pit/trading_calendar.parquet"
    cu.MASTER = root / "runtime/data/u0/security_master.parquet"
    cu.DELISTINGS = root / "runtime/data/u0/pit/delisting_intervals.parquet"
    cu.SUSPENSION_DIRS = [root / "runtime/data/u0/pit", root / "runtime/data/u0/pit/suspension_fresh"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=None,
                    help="read runtime/ from this checkout (default: this script's repo)")
    ap.add_argument("--extra-halt-dir", action="append", default=[],
                    help="additional U0-style halt evidence directory (read only)")
    ap.add_argument("--staging", default=None,
                    help="default: <runtime>/data/v7/silver/_staging_backfill_trailing")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N names this run")
    ap.add_argument("--output", default=None, help="write a NEW panel version here")
    ap.add_argument("--apply", action="store_true", help="replace the live panel atomically")
    ap.add_argument("--report", default=None)
    ap.add_argument("--historical-st-path", default=cu.default_historical_st_path())
    args = ap.parse_args()
    if args.repo_root:
        rebind_repo(Path(args.repo_root))
    cu.SUSPENSION_DIRS = list(cu.SUSPENSION_DIRS) + [Path(p) for p in args.extra_halt_dir]
    if args.staging is None:
        args.staging = str(cu.PANEL.parent.parent / "_staging_backfill_trailing")
    if args.fetch and not args.allow_network:
        raise SystemExit("--fetch needs --allow-network")
    if args.output and Path(args.output).resolve() == cu.PANEL.resolve():
        raise SystemExit("--output must be a new version; use --apply for the live panel")
    if args.output and Path(args.output).exists():
        raise SystemExit(f"refusing to overwrite existing {args.output}")

    evidence = cu._load_required_st_evidence(args.historical_st_path)
    panel = pd.read_parquet(cu.PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_max = panel["trade_date"].max()
    master = pd.read_parquet(cu.MASTER)
    delistings = pd.read_parquet(cu.DELISTINGS) if cu.DELISTINGS.exists() else None
    calendar = pd.read_parquet(cu.CALENDAR)
    halts, _ = load_suspension_evidence(cu.SUSPENSION_DIRS)
    plan, report = plan_backfill(panel, master, delistings, calendar, evidence.frame)
    report.update({"mode": "plan", "panel_sha256": _sha(cu.PANEL),
                   "generated_at": datetime.now(timezone.utc).isoformat()})
    staging = Path(args.staging)
    if args.fetch:
        report["mode"] = "fetch"
        report["fetch"] = fetch(plan, staging, panel_max, args.limit)
    new, verdict = verify(plan, staging, calendar, halts, panel_max)
    report["verify"] = verdict
    if (args.output or args.apply) and len(new):
        merged = build_panel(panel, new, evidence)
        report["rows_added"] = int(len(merged) - len(panel))
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            cu._atomic_write_parquet(merged, target)
            report["output"] = str(target)
            report["output_sha256"] = _sha(target)
            Path(str(target) + ".manifest.json").write_text(json.dumps(report, indent=1, default=str))
        if args.apply:
            lock = cu._acquire_staging_lock(cu.STAGING)
            if lock is None:
                raise SystemExit("catch-up holds the staging lock; not applying")
            try:
                if _sha(cu.PANEL) != report["panel_sha256"]:
                    raise SystemExit("live panel changed since planning; re-run")
                cu._atomic_write_parquet(merged, cu.PANEL)
                report["applied"] = True
            finally:
                lock.close()
    text = json.dumps(report, indent=1, default=str, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
