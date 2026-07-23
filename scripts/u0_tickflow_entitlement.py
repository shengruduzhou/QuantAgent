#!/usr/bin/env python3
"""H-032C §6: TickFlow entitlement re-test (does NOT change the primary bar source).

Re-tests, with minimal calls, the three entitlements that shape U0 strategy:
  * ex_factors (adjustment / corporate-action identity);
  * batch K-line;
  * account rate limit (via a single count=10000 get + the server's own message).

TickFlow remains the primary BAR provider regardless of the outcome. Yields to
any live Track-F job (absolute priority over the shared 10/min TickFlow budget),
so this never steals Track-F's rate budget. Network-gated; records exact status.

Usage: AI_quant_venv/bin/python3 scripts/u0_tickflow_entitlement.py --allow-network
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "runtime/reports/h032c"
TRACKF_PROCS = ("catchup_panel_chunked", "fresh_blind_daily", "catchup_supervisor",
                "coverage_guard", "forward_daily_inference")


def trackf_busy() -> str | None:
    import re
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if "grep" in line or "ps -eo" in line or " eval " in line:
                continue
            for p in TRACKF_PROCS:
                if re.search(rf"{p}\.(py|sh)\b", line):
                    return p
    except Exception:
        pass
    return None


def _probe(tf, label, fn):
    t0 = time.time()
    try:
        r = fn()
        n = 0 if r is None else len(r)
        return {"test": label, "ok": True, "rows": int(n), "latency_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"test": label, "ok": False, "error_type": type(e).__name__,
                "error": str(e)[:160], "latency_s": round(time.time() - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--max-wait-min", type=float, default=45)
    args = ap.parse_args()
    if not args.allow_network:
        print("refusing: --allow-network not confirmed"); return 2

    deadline = time.time() + args.max_wait_min * 60
    while (p := trackf_busy()):
        if time.time() > deadline:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "tickflow_entitlement_audit.json").write_text(json.dumps({
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "DEFERRED", "reason": f"Track-F ({p}) held TickFlow past max-wait; "
                          "re-run when idle", "primary_bar_provider": "TickFlow (unchanged)"}, indent=2))
            print("Track-F still busy past max-wait; wrote DEFERRED audit"); return 0
        print(f"yield to Track-F ({p}); sleep 60s", flush=True); time.sleep(60)

    import tickflow
    import repair_fresh_window_20260704 as rep
    tf = rep._tf_client()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results = []
    results.append(_probe(tf, "count10000_get", lambda: tf.klines.get(
        "600000.SH", period="1d", count=10000, adjust="none", as_dataframe=True)))
    time.sleep(6.5)
    results.append(_probe(tf, "batch_klines", lambda: tf.klines.batch(
        ["600000.SH", "300750.SZ"], period="1d", count=10000, as_dataframe=True)))
    time.sleep(6.5)
    results.append(_probe(tf, "ex_factors", lambda: tf.klines.ex_factors(
        ["600000.SH"], as_dataframe=True)))

    def status_of(label):
        r = next((x for x in results if x["test"] == label), {})
        if r.get("ok"):
            return "ENTITLED"
        return f"NOT_ENTITLED ({r.get('error_type')}: {r.get('error')})" \
            if r.get("error_type") == "PermissionError" else f"ERROR ({r.get('error_type')})"

    audit = {
        "generated": now, "experiment": "H-032C TickFlow entitlement re-test",
        "sdk_version": tickflow.__version__, "api_base_url": tf.base_url,
        "primary_bar_provider": "TickFlow (unchanged, primary bar source)",
        "count_10000_get": status_of("count10000_get"),
        "batch_klines": status_of("batch_klines"),
        "ex_factors": status_of("ex_factors"),
        "measured_rate_limit_per_min": 10,
        "rate_limit_evidence": "server RateLimitError message '请求频率超限 (10/min)' (H-032B); SDK retries 429",
        "corporate_action_classification": (
            "ALTERNATIVE_SOURCE_REQUIRED" if "NOT_ENTITLED" in status_of("ex_factors")
            else "TICKFLOW_AVAILABLE"),
        "raw_results": results,
        "note": "adjustment identity is never fabricated; if ex_factors stays unentitled, "
                "corporate actions require an alternative authoritative source.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tickflow_entitlement_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({k: audit[k] for k in ("count_10000_get", "batch_klines", "ex_factors",
                                            "measured_rate_limit_per_min",
                                            "corporate_action_classification")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
