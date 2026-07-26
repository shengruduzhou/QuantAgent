#!/usr/bin/env python3
"""Track U0: full-universe readiness audit (governed command audit-u0-full-universe).

Emits the composite readiness certificate from real artifacts. Previously this
script computed several of its own gates as literal ``True`` constants next to a
comment claiming the fact they were meant to prove ("adjustment_method_explicit
= True  # adjust='none'"), which is exactly how a panel that silently mixed
forward-adjusted and unadjusted prices passed its own audit. The gate logic now
lives in :mod:`quantagent.data.ashare.readiness` and reads only produced
evidence: the panel manifest, the validation report, the provider capability
matrix, the coverage matrix and the PIT interval manifests.

Prerequisites (each is a governed command of its own):
  u0_security_master.py -> u0_acquire_bars.py -> u0_assemble_panel.py
  u0_pit_intervals.py   -> u0_validate.py     -> ashare_capability_probe.py

Output: runtime/data/u0/full_universe_readiness_certificate.json
        runtime/data/u0/full_universe_readiness_report.md

Usage: AI_quant_venv/bin/python3 scripts/u0_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.readiness import build_certificates, render_report  # noqa: E402

U0 = REPO / "runtime/data/u0"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    certificates = build_certificates(REPO)
    overall = certificates["overall"]
    U0.mkdir(parents=True, exist_ok=True)
    (U0 / "full_universe_readiness_certificate.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False, default=str))
    (U0 / "full_universe_readiness_report.md").write_text(render_report(overall))
    print(json.dumps({"data_readiness_state": overall["data_readiness_state"],
                      "training_permitted": overall["training_permitted"],
                      "gate_pass": overall["gate_pass"],
                      "missing_evidence": overall["missing_evidence"]},
                     indent=2, ensure_ascii=False))
    return 0 if overall["training_permitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
