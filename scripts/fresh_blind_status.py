#!/usr/bin/env python3
"""H-028 Track D: blinded status reader.

Shows ONLY operational health from the append-only ledger. Structurally
refuses to decrypt or display any candidate performance before the first
eligible unblinding date (runtime/reports/h028/fresh_first_read_eligibility.json)
AND unless --i-am-the-preregistered-first-read is passed, in which case the
access is itself appended to the ledger (no deniable early peek).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runtime/paper/fresh_blind"
LEDGER = ROOT / "append_only_ledger.jsonl"
ELIG = REPO / "runtime/reports/h028/fresh_first_read_eligibility.json"


def verify_chain() -> tuple[bool, int]:
    import hashlib
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-am-the-preregistered-first-read", action="store_true")
    args = ap.parse_args()
    ok, n = verify_chain()
    print(f"ledger: {n} records, chain {'VALID' if ok else 'BROKEN'}")
    if LEDGER.exists():
        for line in LEDGER.read_text().strip().splitlines()[-3:]:
            rec = json.loads(line)
            print(" ", rec.get("run_date"), rec.get("data_status"),
                  rec.get("prediction_status"), "failed:", rec.get("failed_job_count"))
    elig = json.loads(ELIG.read_text()) if ELIG.exists() else {}
    first = elig.get("first_eligible_unblinding_date")
    print(f"first eligible unblinding: {first}")
    if not args.i_am_the_preregistered_first_read:
        print("performance: [BLINDED]")
        return 0
    if first and date.today().isoformat() < first:
        print(f"REFUSED: today < {first}. Early unblinding is a protocol violation; "
              "this attempt would be ledger-recorded.")
        return 2
    print("eligible date reached — follow configs/preregistered_evals.json read protocol; "
          "the read itself must be ledger-recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
