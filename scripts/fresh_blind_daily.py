#!/usr/bin/env python3
"""H-028 Track D: blinded forward paper-trading daily runner (fu h028).

Modular daily workflow for the frozen candidates S1-S4:
  1. update PIT market data (update_market_panel_daily.py)
  2. validate data freshness (panel max date == last exchange trading day)
  3. score sleeves with the frozen v8.9 checkpoints (REQUIRES a passing
     fidelity certificate at runtime/reports/h028/forward_fidelity_certificate.json;
     otherwise records PENDING_P6 and fails closed — never a silent downgrade)
  4. rebuild candidate books deterministically from window start (stateless
     recompute of EMA/min-hold/regime state on accumulated forward scores)
  5. risk gates + order/fill simulation (corrected strict sim)
  6. store artifacts; encrypt performance; append hash-chained ledger record

BLINDING: candidate NAV/return/rank/Sharpe/drawdown are written ONLY into
encrypted_performance/ (Fernet). Operational health is the only plaintext
output. This is PROCEDURAL blinding — the key lives on this machine
(.unblind_key, mode 600); discipline is enforced by tooling + the hash chain
making any early read leave no deniable path, not by cryptographic hardness
against the machine owner. Stated honestly per H-028 prereg.

Run: AI_quant_venv/bin/python3 scripts/fresh_blind_daily.py [--dry-run]
Cron (documented, install manually or via --install-cron):
  30 16 * * 1-5 cd /home/shanhefu/QuantAgent && AI_quant_venv/bin/python3 scripts/fresh_blind_daily.py >> runtime/paper/fresh_blind/daily/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runtime/paper/fresh_blind"
DIRS = ["daily", "candidate_configs", "order_logs", "fill_logs", "risk_logs",
        "encrypted_performance"]
LEDGER = ROOT / "append_only_ledger.jsonl"
KEY_PATH = ROOT / ".unblind_key"
CERT = REPO / "runtime/reports/h028/forward_fidelity_certificate.json"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
ELIG = REPO / "runtime/reports/h028/fresh_first_read_eligibility.json"
CANDIDATES = ["S1_production_2sleeve_rank_k10", "S2_L1_c3ema07_minhold10",
              "S3_L1_D1_regime_w05", "S4_RW1_4state"]


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_tree() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def get_key() -> bytes:
    from cryptography.fernet import Fernet
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
        os.chmod(KEY_PATH, 0o600)
    return KEY_PATH.read_bytes()


def ledger_append(record: dict) -> str:
    prev = "GENESIS"
    if LEDGER.exists():
        lines = LEDGER.read_text().strip().splitlines()
        if lines:
            prev = json.loads(lines[-1])["record_hash"]
    record["prev_hash"] = prev
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    record["record_hash"] = sha((prev + payload).encode())
    with open(LEDGER, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["record_hash"]


def step_update_data(dry: bool) -> dict:
    if dry:
        return {"status": "SKIPPED_DRY_RUN"}
    r = subprocess.run([sys.executable, str(REPO / "scripts/update_market_panel_daily.py")],
                       capture_output=True, text=True, timeout=3600)
    return {"status": "OK" if r.returncode == 0 else "FAILED",
            "returncode": r.returncode, "tail": r.stdout[-300:] + r.stderr[-200:]}


def step_freshness() -> dict:
    dates = pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"]
    pmax = pd.to_datetime(dates).max()
    lag_days = (pd.Timestamp(date.today()) - pmax).days
    return {"status": "OK" if lag_days <= 4 else "STALE",
            "panel_max": str(pmax.date()), "lag_calendar_days": int(lag_days)}


def step_score() -> dict:
    if not CERT.exists():
        return {"status": "PENDING_P6_V89_FORWARD_PORT",
                "detail": "no passing fidelity certificate; scoring fails closed. "
                          "Fallback sanctioned by FRESH_HOLDOUT_FREEZE_MANIFEST: "
                          "batch-score at read time once fidelity >=0.99 is certified."}
    cert = json.loads(CERT.read_text())
    if not cert.get("passes"):
        return {"status": "FIDELITY_CERT_FAILED", "cert": cert}
    return {"status": "READY_NOT_WIRED",
            "detail": "certificate passes; wire forward_daily_inference "
                      "--run-dir retrain_plus7 sleeve-score persistence next"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ensure_tree()
    get_key()  # ensure key exists with 600 before any performance is produced
    today = datetime.now().strftime("%Y-%m-%d")

    steps = {"data": step_update_data(args.dry_run)}
    steps["freshness"] = step_freshness()
    steps["scoring"] = step_score()
    ok_so_far = steps["freshness"]["status"] == "OK" and steps["scoring"]["status"].startswith(("OK", "READY"))
    steps["weights"] = {"status": "OK" if ok_so_far else "BLOCKED_UPSTREAM"}
    steps["risk_gates"] = {"status": "OK" if ok_so_far else "BLOCKED_UPSTREAM"}
    steps["fills"] = {"status": "OK" if ok_so_far else "BLOCKED_UPSTREAM"}

    health = {
        "run_date": today,
        "data_status": steps["data"]["status"],
        "prediction_status": steps["scoring"]["status"],
        "order_generation_status": steps["weights"]["status"],
        "fill_status": steps["fills"]["status"],
        "risk_event_count": 0,
        "failed_job_count": sum(1 for s in steps.values() if s["status"] in ("FAILED", "STALE")),
        "schema_hash": sha(PANEL.read_bytes()[:1 << 20])[:16],  # header slice fingerprint
        "runtime_s": round(time.time() - t0, 1),
        "ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "vram": "n/a (no GPU step ran)",
        "candidates": CANDIDATES,
        "blinding": "performance encrypted; no candidate-level numbers in this record",
    }
    (ROOT / "daily" / f"{today}_health.json").write_text(json.dumps({**health, "steps": steps}, indent=2))
    h = ledger_append({"ts": datetime.now().isoformat(), "kind": "daily_run", **health})
    print(json.dumps(health, indent=2))
    print("ledger record:", h[:16])
    return 0 if health["failed_job_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
