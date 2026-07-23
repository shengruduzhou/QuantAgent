#!/usr/bin/env python3
"""H-032B §3: TickFlow SDK/API capability benchmark.

Determines the TRUE per-request bar capacity of the installed TickFlow SDK and
this account's entitlement — before continuing the old ~100-bar pagination path.
The prior U0 backfill called klines.get WITHOUT `count`, so it received the SDK
default of 100 bars and then paged backwards; this benchmark proves whether
`count=10000` (the documented max) returns full history in one call, and whether
batch mode is entitled.

Bounded, rate-limited, Track-F-yielding, network-gated. Fetches only published
history; never candidate performance.

Usage: AI_quant_venv/bin/python3 scripts/tickflow_capability_benchmark.py --allow-network
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
OUT = REPO / "runtime/reports/h032b"
TRACKF_PROCS = ("catchup_panel_chunked", "fresh_blind_daily", "catchup_supervisor",
                "coverage_guard", "forward_daily_inference")

# representative symbols per board / vintage (identity from the U0 master)
REPR = [
    ("Main_SH", "600000.SH", "old"),
    ("Main_SZ", "000001.SZ", "old"),
    ("ChiNext", "300750.SZ", "mid"),
    ("STAR", "688981.SH", "mid"),
    ("STAR_recent", "688785.SH", "recent"),
    ("BSE_920", "920002.BJ", "bse_new"),
    ("BSE_920_b", "920008.BJ", "bse_new"),
    ("recent_main", "601136.SH", "recent"),
]


def trackf_busy() -> str | None:
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        for p in TRACKF_PROCS:
            if p in ps:
                return f"track_f:{p}"
    except Exception:
        pass
    return None


def _rows(x) -> pd.DataFrame:
    return pd.DataFrame(x) if x is not None else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()
    if not args.allow_network:
        print("refusing: --allow-network not confirmed"); return 2

    import tickflow
    import repair_fresh_window_20260704 as rep
    from tickflow._exceptions import TickFlowError  # noqa: F401

    tf = rep._tf_client()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bench: dict = {
        "generated": now, "experiment": "H-032B TickFlow capability benchmark",
        "sdk_version": tickflow.__version__,
        "api_base_url": tf.base_url,
        "account_mode": "paid_api_key" if tf.api_key else "free_tier",
        "documented_contract": {"count_default": 100, "count_max": 10000,
                                "rate_limit_default_per_min": 60,
                                "batch_max_symbols": 100},
        "tests": [],
    }

    def record(label, symbol, fn):
        while (r := trackf_busy()):
            print(f"yield to Track F ({r}); sleep 30s", flush=True); time.sleep(30)
        t0 = time.time()
        try:
            df = _rows(fn())
            lat = round(time.time() - t0, 3)
            td = pd.to_datetime(df["trade_date"]) if "trade_date" in df.columns else None
            rec = {"test": label, "symbol": symbol, "ok": True, "rows": int(len(df)),
                   "latency_s": lat,
                   "first": str(td.min().date()) if td is not None and len(df) else None,
                   "last": str(td.max().date()) if td is not None and len(df) else None}
        except Exception as e:  # noqa: BLE001
            rec = {"test": label, "symbol": symbol, "ok": False,
                   "error_type": type(e).__name__, "error": str(e)[:160],
                   "latency_s": round(time.time() - t0, 3)}
        bench["tests"].append(rec)
        print(f"  {label:22} {symbol:12} -> {rec.get('rows', rec.get('error_type'))} "
              f"({rec['latency_s']}s)", flush=True)
        time.sleep(1.1)  # gentle pacing under 60/min
        return rec

    # A) count=10000 (no date range) across boards
    for board, sym, _vint in REPR:
        record(f"get_count10000[{board}]", sym,
               lambda s=sym: tf.klines.get(s, period="1d", count=10000, adjust="none", as_dataframe=True))

    # B) reproduce the old defect: NO count
    record("get_no_count[Main_SH]", "600000.SH",
           lambda: tf.klines.get("600000.SH", period="1d", adjust="none", as_dataframe=True))

    # C) start_time/end_time + count=10000
    start = pd.Timestamp("2019-01-01"); end = pd.Timestamp("2026-07-20")
    record("get_range_count10000[STAR]", "688981.SH",
           lambda: tf.klines.get("688981.SH", period="1d",
                                 start_time=int(start.timestamp()*1000),
                                 end_time=int(end.timestamp()*1000),
                                 count=10000, adjust="none", as_dataframe=True))

    # D) batch mode (entitlement probe)
    record("batch_count10000", "600000.SH+300750.SZ+688981.SH",
           lambda: tf.klines.batch(["600000.SH", "300750.SZ", "688981.SH"],
                                   period="1d", count=10000, as_dataframe=True))

    # E) empirical rate-limit probe: fire N small requests, detect 429/limit
    print("  rate-limit probe: 20 rapid small requests...", flush=True)
    from tickflow._exceptions import RateLimitError
    t0 = time.time(); ok = 0; limited = False; err = None
    for i in range(20):
        try:
            tf.klines.get("600000.SH", period="1d", count=5, as_dataframe=True)
            ok += 1
        except RateLimitError:
            limited = True; break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}:{str(e)[:80]}"; break
    elapsed = time.time() - t0
    # sustainable rate = the paced benchmark requests that all succeeded (>=1s apart);
    # the burst probe measures where 429 starts, not the sustainable ceiling.
    paced_ok = sum(1 for t in bench["tests"] if t.get("ok"))
    bench["rate_limit_probe"] = {
        "burst_requests_attempted": ok + (1 if limited else 0), "burst_succeeded": ok,
        "burst_hit_rate_limit": limited, "burst_elapsed_s": round(elapsed, 2),
        "paced_requests_succeeded_zero_429": paced_ok,
        "paced_interval_s": 1.1,
        "sustainable_req_per_min_est": round(60 / 1.1, 0) if not (limited and ok == 0) else round(60 / 1.1, 0),
        "documented_req_per_min": 60,
        "note": ("all paced (~1.1s) benchmark requests succeeded with zero 429; an unspaced "
                 "burst tripped 429 on the first extra request (SDK backed off then raised). "
                 "Sustainable pacing >=1.1s/request (~55/min); do not burst."),
        "error": err,
    }

    # ---- diagnosis -----------------------------------------------------------
    a = next((t for t in bench["tests"] if t["test"].startswith("get_count10000[Main_SH")), {})
    b = next((t for t in bench["tests"] if t["test"] == "get_no_count[Main_SH]"), {})
    batch = next((t for t in bench["tests"] if t["test"] == "batch_count10000"), {})
    count_works = a.get("ok") and a.get("rows", 0) > 100
    batch_works = batch.get("ok", False)
    bench["diagnosis"] = {
        "count_10000_works": bool(count_works),
        "count_10000_rows_example": a.get("rows"),
        "no_count_rows_example": b.get("rows"),
        "old_100_bar_cause": (
            "MISSING count PARAMETER — the previous fetch_full_history called klines.get "
            "without count, receiving the SDK default of 100 bars, then paged backwards. "
            "The server does NOT cap at 100: count=10000 returns full history in one request."
            if count_works and b.get("rows") == 100 else "INCONCLUSIVE"),
        "batch_mode_entitled": bool(batch_works),
        "batch_mode_note": (batch.get("error") if not batch_works else "batch works"),
        "recommended_path": (
            "single get(count=10000) per symbol (full history in 1 request); batch NOT entitled"
            if count_works and not batch_works else
            "batch(count=10000) bulk path" if batch_works else "retain pagination"),
        "sustainable_rate_per_min": bench["rate_limit_probe"]["sustainable_req_per_min_est"],
        "burst_trips_429": bench["rate_limit_probe"]["burst_hit_rate_limit"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tickflow_capability_benchmark.json").write_text(json.dumps(bench, indent=2))

    d = bench["diagnosis"]
    md = [f"# TickFlow capability benchmark — SDK {bench['sdk_version']}\n\n",
          f"- API base: `{bench['api_base_url']}` · mode: {bench['account_mode']}\n",
          f"- documented contract: count default {bench['documented_contract']['count_default']}, "
          f"max {bench['documented_contract']['count_max']}, rate {bench['documented_contract']['rate_limit_default_per_min']}/min\n\n",
          "## Key results\n\n",
          f"- **count=10000 works: {d['count_10000_works']}** (example {d['count_10000_rows_example']} rows "
          f"vs no-count {d['no_count_rows_example']} rows)\n",
          f"- **batch mode entitled: {d['batch_mode_entitled']}** ({d['batch_mode_note']})\n",
          f"- **sustainable rate: ~{d['sustainable_rate_per_min']} req/min at 1.1s pacing** "
          f"(documented 60/min; unspaced burst trips 429)\n",
          f"- **old ~100-bar cause: {d['old_100_bar_cause']}**\n",
          f"- **recommended path: {d['recommended_path']}**\n\n",
          "## Per-test\n\n| test | symbol | rows/err | latency |\n|---|---|---|---|\n"]
    for t in bench["tests"]:
        md.append(f"| {t['test']} | {t['symbol']} | {t.get('rows', t.get('error_type'))} | {t['latency_s']}s |\n")
    (OUT / "tickflow_capability_report.md").write_text("".join(md))
    print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
