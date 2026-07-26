#!/usr/bin/env python3
"""Track U0: PIT field source audit + strict point-in-time readiness certificate.

Classifies each mandatory point-in-time execution field as AVAILABLE (with the
row counts and provenance of the interval table that backs it) or
BLOCKED_BY_DATA, and issues the certificate that gates model training. The
classification is derived by :mod:`quantagent.data.ashare.readiness` from the
interval manifests that ``u0_pit_intervals.py`` actually produced — it is never
inferred from current metadata and an unavailable status is never defaulted to
false.

Outputs:
  runtime/data/u0/u0_strict_pit_certificate.json
  runtime/data/u0/pit_field_availability.json
  runtime/data/u0/pit_source_audit.json

Usage: AI_quant_venv/bin/python3 scripts/u0_pit_readiness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.readiness import READY, Evidence, build_certificates  # noqa: E402

OUT = REPO / "runtime/data/u0"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    certificates = build_certificates(REPO)
    certificate = certificates["pit"]
    evidence = Evidence.load(REPO)
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "u0_strict_pit_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False, default=str))
    (OUT / "pit_field_availability.json").write_text(json.dumps({
        "generated": certificate["generated"],
        "pit_field_availability": certificate["pit_field_availability"],
        "blocked_fields": certificate["blocked_pit_fields"],
        "securities": (evidence.master_manifest or {}).get("securities"),
        "by_board": (evidence.master_manifest or {}).get("by_board"),
        "by_status": (evidence.master_manifest or {}).get("by_status"),
        "honesty_note": ("Absent sources are BLOCKED_BY_DATA, never fabricated and never "
                         "defaulted to false. Historical intervals are never inferred from "
                         "current metadata."),
    }, indent=2, ensure_ascii=False, default=str))
    (OUT / "pit_source_audit.json").write_text(json.dumps({
        "generated": certificate["generated"],
        "pit_interval_manifests": evidence.pit_manifests,
        "source_of_truth": "runtime/data/u0/pit/*_manifest.json produced by u0_pit_intervals.py",
    }, indent=2, ensure_ascii=False, default=str))

    print(json.dumps({"decision": certificate["decision"],
                      "training_permitted": certificate["training_permitted"],
                      "blocked_pit_fields": certificate["blocked_pit_fields"]},
                     indent=2, ensure_ascii=False))
    return 0 if certificate["decision"] == READY else 3


if __name__ == "__main__":
    raise SystemExit(main())
