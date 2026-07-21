#!/usr/bin/env python3
"""H-030 Track F1: authoritative shadow-day registry + seven-day gate.

Derives valid shadow days from the append-only ledger (NOT from filenames):
per trading date the latest record is authoritative, earlier ones superseded
(the 2026-07-21 pre-remediation record stays in history but never counts as a
separate day). Applies the preregistered operational gate, writes the registry,
and — only once seven valid days exist — emits the shadow-test certificate.

Idempotent and side-effect-free apart from its own artifacts, so it is safe to
run after every healthcheck. Never reads, decrypts or reports candidate
performance: it checks that encrypted files EXIST, nothing about their content.

Usage: AI_quant_venv/bin/python3 scripts/shadow_day_registry.py [--quiet]
Exit: 0 = certificate written (>=7 valid days) or accumulating; 2 = gate error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runtime/paper/fresh_blind"
LEDGER = ROOT / "append_only_ledger.jsonl"
DAILY = ROOT / "daily"
CERT = REPO / "runtime/reports/h028/forward_fidelity_certificate.json"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
SHADOW_START = pd.Timestamp("2026-07-15")
REQUIRED_DAYS = 7
# data statuses acceptable under the T-1 policy: a bounded top-up that ran out
# of budget is not a failure (staging persists; supervisor completes it).
OK_DATA = {"OK", "PARTIAL_STAGED"}
CANDIDATES = ("S1", "S2", "S3")
# dates whose ingested bars were proven corrupt and dropped (INC-P1)
KNOWN_CORRUPT_INGEST = {"2026-07-21_pre_remediation"}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_chain() -> tuple[bool, int]:
    prev, n = "GENESIS", 0
    if not LEDGER.exists():
        return True, 0
    for line in LEDGER.read_text().strip().splitlines():
        rec = json.loads(line)
        claimed = rec.pop("record_hash")
        if rec.get("prev_hash") != prev:
            return False, n
        payload = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        if hashlib.sha256((prev + payload).encode()).hexdigest() != claimed:
            return False, n
        prev, n = claimed, n + 1
    return True, n


def panel_coverage() -> dict:
    p = pd.read_parquet(PANEL, columns=["trade_date"])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    c = p.groupby("trade_date").size()
    ref = int(c.tail(20).median())
    return {str(d.date()): {"rows": int(v), "ratio": round(v / ref, 3)} for d, v in c.tail(20).items()}


def unblind_accesses() -> list:
    """Any ledger record that is not a routine daily run = access to audit."""
    out = []
    for line in (LEDGER.read_text().strip().splitlines() if LEDGER.exists() else []):
        r = json.loads(line)
        if r.get("kind") not in ("daily_run",):
            out.append({"kind": r.get("kind"), "ts": r.get("ts")})
    return out


def build() -> dict:
    chain_ok, n_records = verify_chain()
    cert_ok, cert_hash = False, None
    if CERT.exists():
        cert_hash = sha_file(CERT)
        cert_ok = bool(json.loads(CERT.read_text()).get("passes"))
    cov = panel_coverage()

    by_date: dict[str, list] = {}
    for idx, line in enumerate(LEDGER.read_text().strip().splitlines()):
        r = json.loads(line)
        if r.get("kind") != "daily_run":
            continue
        d = r.get("run_date")
        if not d or pd.Timestamp(d) < SHADOW_START:
            continue
        r["_idx"], r["_id"] = idx, r.get("record_hash", "")[:16]
        by_date.setdefault(d, []).append(r)

    rows = []
    for d in sorted(by_date):
        recs = sorted(by_date[d], key=lambda r: r.get("ts", ""))
        auth, sup = recs[-1], recs[:-1]
        hp = DAILY / f"{d}_health.json"
        health = json.loads(hp.read_text()) if hp.exists() else {}
        steps = health.get("steps", {})
        wh = (steps.get("weights", {}) or {}).get("books", {}) or {}
        orders = [(ROOT / "order_logs" / f"{d}_{c}_weights.json").exists() for c in CANDIDATES]
        fills = [(ROOT / "fill_logs" / f"{d}_{c}_fills.json").exists() for c in CANDIDATES]
        enc = [(ROOT / "encrypted_performance" / f"{d}_{c}.bin").exists() for c in CANDIDATES]
        pmax = (steps.get("freshness", {}) or {}).get("panel_max")
        lag = (steps.get("freshness", {}) or {}).get("lag_calendar_days")
        pcov = cov.get(str(pmax), {}).get("ratio")

        reasons = []
        if not hp.exists():
            reasons.append("no_health_record")
        if auth.get("data_status") not in OK_DATA:
            reasons.append(f"data_status={auth.get('data_status')}")
        if auth.get("failed_job_count", 1) != 0:
            reasons.append(f"failed_job_count={auth.get('failed_job_count')}")
        if auth.get("prediction_status") != "OK":
            reasons.append(f"prediction_status={auth.get('prediction_status')}")
        if auth.get("order_generation_status") != "OK":
            reasons.append(f"order_status={auth.get('order_generation_status')}")
        if auth.get("fill_status") != "OK":
            reasons.append(f"fill_status={auth.get('fill_status')}")
        if lag is not None and lag > 4:
            reasons.append(f"stale_panel_lag={lag}d")
        if pcov is not None and pcov < 0.93:
            reasons.append(f"partial_cross_section_ratio={pcov}")
        if not cert_ok:
            reasons.append("fidelity_certificate_not_passing")
        if not auth.get("schema_hash"):
            reasons.append("no_schema_hash")
        if len(wh) < 3:
            reasons.append("missing_weights_hashes")
        if not all(orders):
            reasons.append("missing_order_files")
        if not all(fills):
            reasons.append("missing_fill_files")
        if not all(enc):
            reasons.append("missing_encrypted_files")
        if not chain_ok:
            reasons.append("ledger_chain_invalid")

        rows.append({
            "trade_date": d,
            "ledger_record_ids": "|".join(r["_id"] for r in recs),
            "authoritative_record_id": auth["_id"],
            "superseded_record_ids": "|".join(r["_id"] for r in sup),
            "panel_max_date": pmax, "data_status": auth.get("data_status"),
            "prediction_status": auth.get("prediction_status"),
            "order_generation_status": auth.get("order_generation_status"),
            "fill_status": auth.get("fill_status"),
            "failed_job_count": auth.get("failed_job_count"),
            "schema_hash": auth.get("schema_hash"),
            "certificate_hash": (cert_hash or "")[:16],
            "S1_weights_hash": (wh.get("S1") or {}).get("weights_hash", ""),
            "S2_weights_hash": (wh.get("S2") or {}).get("weights_hash", ""),
            "S3_weights_hash": (wh.get("S3") or {}).get("weights_hash", ""),
            "order_files_present": sum(orders), "fill_files_present": sum(fills),
            "encrypted_files_present": sum(enc),
            "ledger_chain_valid": chain_ok,
            "valid_shadow_day": not reasons,
            "invalid_reason": ";".join(reasons),
        })

    valid = [r["trade_date"] for r in rows if r["valid_shadow_day"]]
    reg = {
        "generated": datetime.now().isoformat(), "experiment": "H-030 Track F1",
        "shadow_start": str(SHADOW_START.date()), "required_days": REQUIRED_DAYS,
        "ledger_records_total": n_records, "ledger_chain_valid": chain_ok,
        "fidelity_certificate_passes": cert_ok, "certificate_sha256": cert_hash,
        "valid_shadow_days": len(valid), "valid_dates": valid,
        "unblind_or_nonroutine_accesses": unblind_accesses(),
        "days": rows,
        "note": ("2026-07-21 carries three records: the 19:30 cron run (data step failed), "
                 "the 20:18 auto-repair handoff built on the INC-P1 corrupted panel, and the "
                 "23:56 post-remediation run. Only the last is authoritative; the earlier two "
                 "remain in append-only history and are NOT counted as separate shadow days."),
        "blinding": "existence-only checks; no candidate performance read or reported",
    }
    return reg


def write_certificate(reg: dict) -> None:
    valid = reg["valid_dates"][:REQUIRED_DAYS]
    days = {r["trade_date"]: r for r in reg["days"]}
    decision = "FROZEN_BLIND_PAPER_ACTIVE"
    if not reg["ledger_chain_valid"]:
        decision = "FROZEN_BLIND_PAPER_NOT_READY_LEDGER"
    elif not reg["fidelity_certificate_passes"]:
        decision = "FROZEN_BLIND_PAPER_NOT_READY_DATA"
    cert = {
        "generated": datetime.now().isoformat(), "experiment": "H-030 Track F1",
        "dates": valid,
        "authoritative_record_ids": [days[d]["authoritative_record_id"] for d in valid],
        "schema_hashes": sorted({days[d]["schema_hash"] for d in valid}),
        "fidelity_certificate_sha256": reg["certificate_sha256"],
        "file_completeness": {d: {"orders": days[d]["order_files_present"],
                                  "fills": days[d]["fill_files_present"],
                                  "encrypted": days[d]["encrypted_files_present"]} for d in valid},
        "ledger_chain_valid": reg["ledger_chain_valid"],
        "ledger_records_total": reg["ledger_records_total"],
        "unblind_access_audit": reg["unblind_or_nonroutine_accesses"] or "none",
        "known_limitations": [
            "procedural blinding: key is local; discipline enforced by tooling + hash chain",
            "S4 decisions not produced by the daily runner (see s4_readiness_certificate.json)",
            "T-1 cadence: decisions use the prior published close (vendor 10 req/min limit)",
            "fixed 3,872-symbol cohort excludes STAR/BSE and post-2020 listings (Track U0)",
        ],
        "decision": decision,
        "blinding": "no candidate performance included",
    }
    (ROOT / "shadow_test_certificate.json").write_text(json.dumps(cert, indent=2))
    md = [f"# shadow_test_report — {decision}\n",
          f"\nSeven valid operational days: {', '.join(valid)}\n",
          f"\nLedger chain VALID across {reg['ledger_records_total']} records; ",
          f"fidelity certificate {reg['certificate_sha256'][:16]} passes.\n",
          "\n## Per-day evidence\n\n| date | authoritative | data | orders/fills/enc | failed |\n|---|---|---|---|---|\n"]
    for d in valid:
        r = days[d]
        md.append(f"| {d} | {r['authoritative_record_id']} | {r['data_status']} | "
                  f"{r['order_files_present']}/{r['fill_files_present']}/{r['encrypted_files_present']} | "
                  f"{r['failed_job_count']} |\n")
    md.append("\n## Limitations\n\n" + "\n".join(f"- {x}" for x in cert["known_limitations"]) + "\n")
    md.append("\nNo candidate performance is exposed in this report.\n")
    (ROOT / "shadow_test_report.md").write_text("".join(md))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    reg = build()
    (ROOT / "shadow_day_registry.json").write_text(json.dumps(reg, indent=2))
    cols = list(reg["days"][0].keys()) if reg["days"] else []
    with open(ROOT / "shadow_day_registry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg["days"])
    if reg["valid_shadow_days"] >= REQUIRED_DAYS:
        write_certificate(reg)
        status = "SHADOW_TEST_COMPLETE"
    else:
        status = "SHADOW_TEST_ACCUMULATING"
    if not args.quiet:
        print(json.dumps({"status": status, "valid_shadow_days": reg["valid_shadow_days"],
                          "valid_dates": reg["valid_dates"],
                          "chain_valid": reg["ledger_chain_valid"]}, indent=2))
        for r in reg["days"]:
            if not r["valid_shadow_day"]:
                print(f"  excluded {r['trade_date']}: {r['invalid_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
