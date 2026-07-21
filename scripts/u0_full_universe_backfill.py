#!/usr/bin/env python3
"""H-030 Track U0: full-universe historical backfill worker (LOW PRIORITY).

Builds the next-generation point-in-time A-share data foundation covering the
boards the frozen 3,872-symbol cohort never contained (STAR, BSE) plus
post-2020 listings and historically delisted names.

TRACK-F ALWAYS WINS. Before every batch this worker:
  * takes its OWN lock (.u0_backfill.lock) — never the Track-F lock;
  * yields (sleeps) while a Track-F catch-up/runner/guard process is alive or
    the Track-F supervisor lock is held;
  * never fetches an unpublished current-day close (16:00 CST margin);
  * paces to the measured vendor limit (10 req/min);
  * checkpoints after every batch and resumes from staging;
  * writes only into runtime/data/v7/full_universe/ — the frozen ranking
    universe and the blind-paper panel are never touched.

Subcommands:
  fetch     backfill missing symbols (resumable; bounded by --max-minutes)
  assemble  merge staging -> panel + eligible universe + PIT checks + manifest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
OUT = REPO / "runtime/data/v7/full_universe"
STAGING = OUT / "_staging"
PANEL_F = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
LOCK = REPO / "runtime/paper/fresh_blind/.u0_backfill.lock"
TRACKF_LOCK = REPO / "runtime/paper/fresh_blind/.catchup_supervisor.lock"
REPORTS = REPO / "runtime/reports/h030"
BATCH = 25
REQ_INTERVAL_S = 6.2          # 10 req/min measured limit, with margin
TRACKF_PROCS = ("catchup_panel_chunked", "fresh_blind_daily", "catchup_supervisor",
                "coverage_guard", "forward_daily_inference")
IPO_INELIGIBLE_DAYS = 60      # preregistered (H-028): new listings ineligible for 60 td


def trackf_busy() -> str | None:
    """Return a reason string while Track F is active, else None."""
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        for p in TRACKF_PROCS:
            if p in ps:
                return f"track_f_process_active:{p}"
    except Exception:
        pass
    if TRACKF_LOCK.exists():
        try:   # non-blocking probe: if we can lock it, Track F is idle
            import fcntl
            with open(TRACKF_LOCK, "w") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except BlockingIOError:
                    return "track_f_supervisor_lock_held"
        except Exception:
            return None
    return None


def last_available() -> pd.Timestamp:
    cst = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    return cst.normalize() if cst.hour * 60 + cst.minute >= 16 * 60 \
        else cst.normalize() - pd.Timedelta(days=1)


def cmd_fetch(args) -> int:
    import fcntl
    import repair_fresh_window_20260704 as rep
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lf = open(LOCK, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another U0 backfill holds the lock — exiting"); return 0

    STAGING.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(MASTER)
    panel_syms = set(pd.read_parquet(PANEL_F, columns=["symbol"])["symbol"].astype(str).unique())
    done = set()
    for f in STAGING.glob("sym_*.parquet"):
        done.add(f.stem.replace("sym_", "").replace("_", "."))
    failed_log = OUT / "failed_fetch_ledger.csv"
    prior_failed = set()
    if failed_log.exists():
        prior_failed = set(pd.read_csv(failed_log)["symbol"].astype(str))

    todo = [s for s in sorted(master["symbol"].astype(str).unique())
            if s not in panel_syms and s not in done]
    print(f"master {len(master)} | already in frozen panel {len(panel_syms & set(master['symbol']))} "
          f"| staged {len(done)} | todo {len(todo)} | prior failures {len(prior_failed)}", flush=True)
    if not todo:
        print("nothing to fetch"); return 0

    end = last_available()
    tf = rep._tf_client()
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    fetched = failures = yielded = 0
    fail_rows = []
    for i, sym in enumerate(todo):
        if time.time() > deadline:
            print(f"budget reached ({args.max_minutes}min) — staging preserved", flush=True)
            break
        while (reason := trackf_busy()):
            yielded += 1
            print(f"yielding to Track F ({reason}); sleeping 120s", flush=True)
            time.sleep(120)
            if time.time() > deadline:
                break
        tick = time.time()
        try:
            k = rep.fetch_with_retry(tf, sym)
            if k is None or not len(k):
                fail_rows.append({"symbol": sym, "reason": "empty_or_unavailable",
                                  "ts": datetime.now().isoformat()})
                failures += 1
            else:
                k = k.copy()
                k["symbol"] = sym
                k["trade_date"] = pd.to_datetime(k["trade_date"])
                k = k[k["trade_date"] <= end]
                k["volume"] = pd.to_numeric(k["volume"], errors="coerce") * 100.0  # lots->shares
                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
                k[cols].to_parquet(STAGING / f"sym_{sym.replace('.', '_')}.parquet", index=False)
                fetched += 1
        except Exception as e:
            fail_rows.append({"symbol": sym, "reason": str(e)[:120],
                              "ts": datetime.now().isoformat()})
            failures += 1
        if (i + 1) % BATCH == 0:
            print(f"  {i+1}/{len(todo)} fetched={fetched} failed={failures} "
                  f"yielded={yielded} {time.time()-t0:.0f}s", flush=True)
            if fail_rows:
                pd.DataFrame(fail_rows).to_csv(
                    failed_log, mode="a", header=not failed_log.exists(), index=False)
                fail_rows = []
        elapsed = time.time() - tick
        if elapsed < REQ_INTERVAL_S:
            time.sleep(REQ_INTERVAL_S - elapsed)
    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(failed_log, mode="a",
                                       header=not failed_log.exists(), index=False)
    print(json.dumps({"fetched": fetched, "failed": failures, "yield_events": yielded,
                      "staged_total": len(list(STAGING.glob('sym_*.parquet'))),
                      "runtime_s": round(time.time() - t0, 1)}))
    return 0


def cmd_assemble(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(MASTER)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    master.to_parquet(OUT / "historical_security_master.parquet", index=False)

    files = sorted(STAGING.glob("sym_*.parquet"))
    frozen = pd.read_parquet(PANEL_F, columns=["symbol", "trade_date", "open", "high", "low",
                                               "close", "volume", "amount"])
    frozen["trade_date"] = pd.to_datetime(frozen["trade_date"])
    frozen["source_track"] = "frozen_cohort"
    parts = [frozen]
    if files:
        new = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        new["source_track"] = "u0_backfill"
        parts.append(new)
    panel = pd.concat(parts, ignore_index=True)
    panel["symbol"] = panel["symbol"].astype(str)

    checks = {}
    n0 = len(panel)
    panel = panel.drop_duplicates(["symbol", "trade_date"], keep="first")
    checks["duplicate_rows_removed"] = int(n0 - len(panel))

    lm = master.set_index("symbol")
    panel["_list"] = panel["symbol"].map(lm["listing_date"])
    panel["_delist"] = panel["symbol"].map(lm["delisting_date"])
    pre_listing = int((panel["_list"].notna() & (panel["trade_date"] < panel["_list"])).sum())
    post_delist = int((panel["_delist"].notna() & (panel["trade_date"] > panel["_delist"])).sum())
    checks["rows_before_listing_date"] = pre_listing
    checks["rows_after_delisting_date"] = post_delist
    if pre_listing:
        panel = panel[~(panel["_list"].notna() & (panel["trade_date"] < panel["_list"]))]
        checks["pre_listing_rows_dropped"] = pre_listing
    panel = panel.drop(columns=["_list", "_delist"])

    checks["negative_or_zero_close"] = int((panel["close"] <= 0).sum())
    checks["null_close"] = int(panel["close"].isna().sum())
    checks["max_date"] = str(panel["trade_date"].max().date())
    checks["min_date"] = str(panel["trade_date"].min().date())
    checks["symbols"] = int(panel["symbol"].nunique())
    checks["rows"] = int(len(panel))
    checks["unpublished_close_rows"] = int((panel["trade_date"] > last_available()).sum())

    panel.to_parquet(OUT / "full_universe_market_panel.parquet", index=False)

    # daily eligible universe: listed, price>0, and past the preregistered IPO window
    panel = panel.sort_values(["symbol", "trade_date"])
    age = panel.groupby("symbol", sort=False).cumcount()
    elig = panel[["symbol", "trade_date"]].copy()
    elig["age_td"] = age.to_numpy()
    elig["eligible"] = (age >= IPO_INELIGIBLE_DAYS).to_numpy() & (panel["close"] > 0).to_numpy()
    elig.to_parquet(OUT / "daily_full_universe_eligible.parquet", index=False)

    have = set(panel["symbol"].unique())
    missing = master[~master["symbol"].astype(str).isin(have)]
    missing[["symbol", "board", "status", "listing_date"]].assign(
        reason="not_yet_backfilled").to_csv(OUT / "missing_symbol_ledger.csv", index=False)

    by_board = (panel[["symbol"]].drop_duplicates().assign(
        board=lambda d: d["symbol"].map(lm["board"]))["board"].value_counts().to_dict())
    manifest = {
        "generated": datetime.now().isoformat(), "experiment": "H-030 Track U0",
        "purpose": "next-generation full-universe foundation; FROZEN 3,872 ranking universe untouched",
        "master_securities": int(len(master)), "panel_symbols": checks["symbols"],
        "panel_rows": checks["rows"], "date_range": [checks["min_date"], checks["max_date"]],
        "symbols_by_board": {str(k): int(v) for k, v in by_board.items()},
        "missing_symbols": int(len(missing)),
        "staged_backfill_files": len(files),
        "pit_checks": checks,
        "ipo_ineligible_days": IPO_INELIGIBLE_DAYS,
        "known_gaps": [
            "ST history (st_start/st_end) unavailable — no PIT source on disk",
            "delisting DATE not normalized per source (status carried, date often null)",
            "board-specific price-limit rules recorded in master but flags not yet recomputed",
            "corporate-action consistency not yet verified for backfilled symbols",
        ],
        "quarantine_policy": "U0 is a DATA track; no research evaluation runs on it in this ticket",
    }
    (OUT / "full_universe_manifest.json").write_text(json.dumps(manifest, indent=2))

    gates_pass = (checks["duplicate_rows_removed"] == 0 and checks["rows_after_delisting_date"] == 0
                  and checks["null_close"] == 0 and checks["negative_or_zero_close"] == 0
                  and checks["unpublished_close_rows"] == 0 and len(missing) == 0)
    verdict = "FULL_UNIVERSE_DATA_READY" if gates_pass else "FULL_UNIVERSE_DATA_NOT_READY"
    md = [f"# full_universe_quality_report — {verdict}\n\n",
          f"- securities in master: **{len(master)}** (listed {int((master['status']=='listed').sum())}, "
          f"delisted {int((master['status']=='delisted').sum())})\n",
          f"- panel symbols: **{checks['symbols']}** / rows {checks['rows']:,} "
          f"({checks['min_date']}..{checks['max_date']})\n",
          f"- still missing: **{len(missing)}** symbols\n",
          f"- by board: {by_board}\n\n## PIT checks\n\n",
          "".join(f"- {k}: {v}\n" for k, v in checks.items()),
          "\n## Known gaps\n\n", "".join(f"- {g}\n" for g in manifest["known_gaps"]),
          f"\n**Verdict: {verdict}** — the frozen 3,872-symbol ranking universe is untouched; "
          "no model training may start until every gate passes.\n"]
    (OUT / "full_universe_quality_report.md").write_text("".join(md))
    print(json.dumps({"verdict": verdict, "symbols": checks["symbols"],
                      "missing": int(len(missing)), "rows": checks["rows"],
                      "pit": checks}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--max-minutes", type=float, default=240)
    sub.add_parser("assemble")
    args = ap.parse_args()
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
