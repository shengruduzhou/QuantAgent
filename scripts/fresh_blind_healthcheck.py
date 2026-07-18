#!/usr/bin/env python3
"""Blind forward-paper standing healthcheck (H-029 activation, continuous
monitoring mandate).

Runs after the daily blind run (cron 17:30 weekdays). Checks, in order:
  1. ledger hash chain VALID;
  2. a daily_run record exists for the last expected trading day;
  3. that record has failed_job_count == 0 and all step statuses OK;
  4. panel freshness (lag <= 4 calendar days);
  5. fidelity certificate still passes=true (never relaxed silently).

Bounded auto-remediation (recorded, never silent):
  - if the ONLY failure is the data step's backlog timeout and the panel lag
    is <= 6 calendar days, run one standalone catch-up + one runner retry.
Anything else -> ALERT file (runtime/paper/fresh_blind/ALERT.txt) + nonzero
exit. A healthy check removes the ALERT file. All checks append to
healthcheck.log. NO candidate performance is read or printed here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runtime/paper/fresh_blind"
LEDGER = ROOT / "append_only_ledger.jsonl"
ALERT = ROOT / "ALERT.txt"
LOG = ROOT / "healthcheck.log"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
CERT = REPO / "runtime/reports/h028/forward_fidelity_certificate.json"

sys.path.insert(0, str(REPO / "scripts"))
from fresh_blind_status import verify_chain  # noqa: E402


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")
    print(msg, flush=True)


def latest_record() -> dict | None:
    if not LEDGER.exists():
        return None
    lines = [json.loads(x) for x in LEDGER.read_text().strip().splitlines()]
    daily = [r for r in lines if r.get("kind") == "daily_run"]
    return daily[-1] if daily else None


def main() -> int:
    os.chdir(REPO)
    problems: list[str] = []
    ok, n = verify_chain()
    if not ok:
        problems.append(f"LEDGER_CHAIN_BROKEN at record {n}")
    rec = latest_record()
    today = date.today()
    if rec is None:
        problems.append("NO_DAILY_RECORDS")
    else:
        rec_age = (pd.Timestamp(today) - pd.Timestamp(rec["run_date"])).days
        if today.weekday() < 5 and rec_age > 1:
            problems.append(f"STALE_DAILY_RECORD last={rec['run_date']} age={rec_age}d")
        if rec.get("failed_job_count", 1) != 0:
            problems.append(f"LAST_RUN_HAD_FAILURES ({rec['run_date']}: "
                            f"data={rec.get('data_status')} pred={rec.get('prediction_status')} "
                            f"orders={rec.get('order_generation_status')})")
    pmax = pd.to_datetime(pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"]).max()
    lag = (pd.Timestamp(today) - pmax).days
    if lag > 4:
        problems.append(f"PANEL_STALE max={pmax.date()} lag={lag}d")
    cert_ok = CERT.exists() and json.loads(CERT.read_text()).get("passes") is True
    if not cert_ok:
        problems.append("FIDELITY_CERT_MISSING_OR_FAILING")

    # bounded auto-remediation: data backlog only, and only once per invocation
    data_timeout = bool(rec) and str(rec.get("data_status", "")).startswith("FAILED") \
        and "TIMEOUT" in json.dumps(rec)
    if problems and data_timeout and lag <= 6 and cert_ok and ok:
        log(f"AUTO_REMEDIATION: standalone catch-up (lag {lag}d) + runner retry")
        r1 = subprocess.run([sys.executable, "scripts/update_market_panel_daily.py"],
                            capture_output=True, text=True, timeout=7200)
        r2 = subprocess.run([sys.executable, "scripts/fresh_blind_daily.py"],
                            capture_output=True, text=True, timeout=10800)
        log(f"AUTO_REMEDIATION result: update rc={r1.returncode} runner rc={r2.returncode}")
        if r2.returncode == 0:
            problems = [p for p in problems if not p.startswith(("LAST_RUN_HAD_FAILURES", "PANEL_STALE"))]

    if problems:
        ALERT.write_text(f"{datetime.now().isoformat()}\n" + "\n".join(problems) + "\n")
        log("UNHEALTHY: " + "; ".join(problems))
        return 1
    if ALERT.exists():
        ALERT.unlink()
    log(f"HEALTHY (chain {n} records, panel max {pmax.date()}, cert passes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
