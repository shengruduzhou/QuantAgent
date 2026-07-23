#!/usr/bin/env python3
"""H-032A Track U0: representative STAR/BSE provider probe (diagnostic).

Before any complete board backfill, this probes a small, representative set of
STAR and BSE securities to determine WHY those boards are uncovered — symbol
format, entitlement, endpoint support, pagination, date parsing, old/new BSE
code mapping, or actual absence — rather than silently excluding a board.

It is bounded (a fixed symbol list), rate-limited to the measured vendor limit,
yields to any live Track-F job, and NEVER fetches an unpublished current-day bar.
It writes only a diagnostic report; it does not stage or assemble any panel.

Usage: AI_quant_venv/bin/python3 scripts/u0_star_bse_probe.py --allow-network
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
OUT = REPO / "runtime/data/u0/star_bse_probe_report.json"
REQ_INTERVAL_S = 6.2
TRACKF_PROCS = ("catchup_panel_chunked", "fresh_blind_daily", "catchup_supervisor",
                "coverage_guard", "forward_daily_inference")
TRACKF_LOCK = REPO / "runtime/paper/fresh_blind/.catchup_supervisor.lock"

# Known-real BSE symbols for a code-mapping cross-check (public listings).
KNOWN_REAL_BSE_8X = ["830799.BJ", "831195.BJ", "833171.BJ", "835174.BJ",
                     "836221.BJ", "838924.BJ", "871981.BJ", "873169.BJ"]
KNOWN_REAL_BSE_920 = ["920002.BJ", "920008.BJ", "920019.BJ"]


def trackf_busy() -> str | None:
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        for p in TRACKF_PROCS:
            if p in ps:
                return f"track_f_process_active:{p}"
    except Exception:
        pass
    if TRACKF_LOCK.exists():
        try:
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


def representative_symbols() -> dict:
    m = pd.read_parquet(MASTER)
    m["listing_date"] = pd.to_datetime(m["listing_date"], errors="coerce")
    out = {}
    for board in ("STAR", "BSE"):
        b = m[m.board == board].dropna(subset=["listing_date"]).copy()
        b["yr"] = b.listing_date.dt.year
        reps = b.sort_values("listing_date").groupby("yr").first().reset_index()
        out[board] = [{"symbol": r.symbol, "listing_date": str(r.listing_date.date()),
                       "status": r.status, "code": str(r.code)} for r in reps.itertuples()]
    return out


def probe_one(tf, symbol: str, end: pd.Timestamp, attempts: int = 3) -> dict:
    """One wide-window request; the vendor caps at ~100 bars so this confirms
    existence without a full backfill. Returns a classified result."""
    start = pd.Timestamp("2019-01-01")
    last_exc = None
    for i in range(attempts):
        try:
            k = tf.klines.get(symbol, period="1d",
                              start_time=int(start.timestamp() * 1000),
                              end_time=int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1,
                              adjust="none", as_dataframe=True)
            if k is None or not len(k):
                return {"symbol": symbol, "status": "EMPTY_PROVIDER_RESPONSE", "rows": 0}
            df = pd.DataFrame(k)
            tcol = "trade_date" if "trade_date" in df.columns else (
                "timestamp" if "timestamp" in df.columns else df.columns[0])
            td = pd.to_datetime(df[tcol], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai") \
                if df[tcol].dtype.kind in "iu" else pd.to_datetime(df[tcol])
            return {"symbol": symbol, "status": "COVERED", "rows": int(len(df)),
                    "first": str(td.min().date()), "last": str(td.max().date())}
        except Exception as e:  # noqa: BLE001
            last_exc = f"{type(e).__name__}:{str(e)[:120]}"
            if i < attempts - 1:
                time.sleep((2, 5)[min(i, 1)])
    return {"symbol": symbol, "status": "RETRYABLE_FAILURE", "rows": 0, "error": last_exc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true",
                    help="explicit confirmation required before any vendor call")
    ap.add_argument("--max-symbols", type=int, default=40)
    args = ap.parse_args()
    if not args.allow_network:
        print("refusing to probe: --allow-network not confirmed"); return 2

    import repair_fresh_window_20260704 as rep
    tf = rep._tf_client()
    end = last_available()
    reps = representative_symbols()

    plan = []
    for s in reps["STAR"]:
        plan.append(("STAR", "master_rep", s["symbol"], s["listing_date"], s["status"]))
    for s in reps["BSE"]:
        plan.append(("BSE", "master_rep_920x", s["symbol"], s["listing_date"], s["status"]))
    for s in KNOWN_REAL_BSE_8X:
        plan.append(("BSE", "known_real_8xxxxx", s, None, "cross_check"))
    for s in KNOWN_REAL_BSE_920:
        plan.append(("BSE", "known_real_920x", s, None, "cross_check"))
    plan = plan[: args.max_symbols]

    # connectivity control: a known-good main-board symbol must succeed
    control = probe_one(tf, "600000.SH", end)
    print(f"connectivity control 600000.SH -> {control['status']} ({control.get('rows')} rows)", flush=True)

    results = []
    for i, (board, cohort, symbol, listing, status) in enumerate(plan):
        while (reason := trackf_busy()):
            print(f"yield to Track F ({reason}); sleep 120s", flush=True)
            time.sleep(120)
        r = probe_one(tf, symbol, end)
        r.update({"board": board, "cohort": cohort, "listing_date": listing, "master_status": status})
        results.append(r)
        print(f"  [{i+1}/{len(plan)}] {board:4} {cohort:18} {symbol:12} -> {r['status']} "
              f"({r.get('rows')} rows{'' if r['status']!='COVERED' else ' '+r.get('first','')+'..'+r.get('last','')})",
              flush=True)
        time.sleep(REQ_INTERVAL_S)

    def rate(board, cohort=None):
        sub = [r for r in results if r["board"] == board and (cohort is None or r["cohort"] == cohort)]
        cov = sum(1 for r in sub if r["status"] == "COVERED")
        return {"probed": len(sub), "covered": cov,
                "empty": sum(1 for r in sub if r["status"] == "EMPTY_PROVIDER_RESPONSE"),
                "retryable": sum(1 for r in sub if r["status"] == "RETRYABLE_FAILURE")}

    # ---- diagnosis -----------------------------------------------------------
    star = rate("STAR")
    bse_master = rate("BSE", "master_rep_920x")
    bse_real8 = rate("BSE", "known_real_8xxxxx")
    bse_real920 = rate("BSE", "known_real_920x")

    star_diag = ("FETCHABLE_NOT_PROBED" if star["covered"] > 0
                 else "UNSUPPORTED_OR_ABSENT")
    # BSE code-mapping diagnosis
    if bse_real8["covered"] > 0 and bse_master["covered"] == 0:
        bse_diag = "MASTER_CODE_MAP_WRONG:vendor_supports_BSE_8xxxxx_but_master_has_only_920x_placeholders"
    elif bse_real8["covered"] == 0 and bse_real920["covered"] == 0 and bse_master["covered"] == 0:
        bse_diag = "BSE_UNSUPPORTED_ENTITLEMENT_OR_ENDPOINT:no_.BJ_symbol_returned_data"
    elif bse_master["covered"] > 0:
        bse_diag = "BSE_920x_FETCHABLE_NOT_PROBED"
    else:
        bse_diag = "INCONCLUSIVE"

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "H-032A STAR/BSE representative probe",
        "connectivity_control": control,
        "vendor_page_cap_note": "vendor caps ~100 bars/response; a wide window confirms existence, not full history",
        "rates": {"STAR_master": star, "BSE_master_920x": bse_master,
                  "BSE_known_real_8xxxxx": bse_real8, "BSE_known_real_920x": bse_real920},
        "diagnosis": {"STAR": star_diag, "BSE": bse_diag},
        "master_bse_universe_note": ("master carries 327 BSE codes ALL in the 920xxx range and ZERO 8xxxxx "
                                     "codes; the real BSE universe is mostly 8xxxxx — the master's BSE "
                                     "identity is structurally incomplete regardless of vendor coverage"),
        "results": results,
        "blinding": "no candidate performance included",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"STAR": star_diag, "BSE": bse_diag, "rates": report["rates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
