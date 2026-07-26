#!/usr/bin/env python3
"""Track U0: bar-data readiness certificate (separate from the strict PIT decision).

Judges only whether clean, correctly-attributed OHLCV exists for the full
universe — identity, provider capability, coverage and structural quality —
independent of the PIT-metadata question. The gate values come from
:mod:`quantagent.data.ashare.readiness`, which reads produced artifacts; this
script no longer asserts unit or adjustment facts as constants.

Decisions: U0_BAR_READY / U0_BAR_NOT_READY_COVERAGE / U0_BAR_NOT_READY_IDENTITY /
U0_BAR_NOT_READY_PROVIDER / U0_BAR_NOT_READY_QUALITY.

Output: runtime/data/u0/u0_bar_readiness_certificate.json

Usage: AI_quant_venv/bin/python3 scripts/u0_bar_readiness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.readiness import BAR_READY, build_certificates  # noqa: E402

OUT = REPO / "runtime/data/u0"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    certificate = build_certificates(REPO)["bar"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u0_bar_readiness_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({"decision": certificate["decision"],
                      "gate_pass": certificate["gate_pass"],
                      "coverage_by_board": certificate["coverage"].get("by_board"),
                      "quality_failures": certificate["quality"].get("failures")},
                     indent=2, ensure_ascii=False))
    return 0 if certificate["decision"] == BAR_READY else 3


if __name__ == "__main__":
    raise SystemExit(main())
