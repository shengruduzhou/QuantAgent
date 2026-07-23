#!/usr/bin/env python3
"""H-031 Track U0: full-universe readiness audit (audit-u0-full-universe).

Runs the mandatory H-031 §9 gates over the assembled full-universe artifacts and
returns exactly one §10 data-readiness state. Training stays blocked until the
state is FULL_UNIVERSE_DATA_READY. A missing mandatory source produces
BLOCKED_BY_DATA / a NOT_READY state — never a silent default-false pass.

State precedence (most upstream blocker wins):
  INTEGRATION -> PROVIDER -> COVERAGE -> PIT -> READY

Reads only existence/quality/gate fields; never candidate performance.

Output: runtime/data/u0/full_universe_readiness_certificate.json
        runtime/data/u0/full_universe_readiness_report.md

Usage: AI_quant_venv/bin/python3 scripts/u0_audit.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
FULL = REPO / "runtime/data/v7/full_universe"
COVERAGE = U0 / "provider_coverage_matrix.parquet"
COVERAGE_SUMMARY = U0 / "provider_coverage_summary.json"
MASTER = U0 / "historical_security_master.parquet"
PIT_AVAIL = U0 / "pit_field_availability.json"
PANEL = FULL / "full_universe_market_panel.parquet"
MANIFEST = FULL / "full_universe_manifest.json"

REQUIRED_BOARDS = ("SH_Main", "SZ_Main", "ChiNext", "STAR", "BSE")


def _load_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def audit() -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    integration_missing = [str(p.relative_to(REPO)) for p in
                           (COVERAGE, COVERAGE_SUMMARY, MASTER, PIT_AVAIL) if not p.exists()]

    gates: dict[str, dict] = {}
    state = None

    # ---- INTEGRATION gate: prerequisite artifacts must exist -----------------
    gates["integration"] = {
        "required_artifacts_present": not integration_missing,
        "missing": integration_missing,
    }
    if integration_missing:
        state = "FULL_UNIVERSE_DATA_NOT_READY_INTEGRATION"

    cov = pd.read_parquet(COVERAGE) if COVERAGE.exists() else pd.DataFrame()
    cov_summary = _load_json(COVERAGE_SUMMARY) or {}
    pit = _load_json(PIT_AVAIL) or {}

    # ---- PROVIDER gate: is a bar provider CAPABLE of serving each board? -----
    # A board is a PROVIDER failure only if its symbols were PROBED and the
    # provider returned nothing (systematic EMPTY) — i.e. entitlement/capability
    # is the blocker. A board that is merely NOT_PROBED (backfill has not reached
    # it) is a COVERAGE/backlog problem and is deferred to the coverage gate.
    covered = cov[cov["selected_bar_provider"] == "tickflow"] if len(cov) else cov
    covered_by_board = covered["board"].value_counts().to_dict() if len(covered) else {}
    all_board_totals = cov["board"].value_counts().to_dict() if len(cov) else {}
    empty_responses = int((cov["tickflow_status"] == "EMPTY").sum()) if len(cov) else 0
    provider_unable_boards = []
    for b in REQUIRED_BOARDS:
        if len(cov) == 0 or all_board_totals.get(b, 0) == 0:
            continue
        board_rows = cov[cov["board"] == b]
        probed = board_rows[board_rows["tickflow_status"].isin(["COVERED_FROZEN_COHORT",
                                                                "COVERED_BACKFILL", "EMPTY"])]
        n_covered = covered_by_board.get(b, 0)
        # probed at least once, yet zero covered -> provider genuinely cannot serve it
        if len(probed) > 0 and n_covered == 0:
            provider_unable_boards.append(b)
    gates["provider"] = {
        "securities": int(len(cov)),
        "covered_by_any_bar_provider": int(len(covered)),
        "provider_unable_boards": provider_unable_boards,
        "tickflow_empty_responses": empty_responses,
        "fallback_providers_exercised": False,
        "note": ("EMPTY vendor responses are recorded as EMPTY, not as 'no history'; NOT_PROBED boards "
                 "(STAR/BSE) are a backfill-backlog COVERAGE issue, not a provider-capability failure. "
                 "A fallback provider sweep (akshare/qlib) has not been run for uncovered names."),
    }
    provider_ok = len(cov) > 0 and len(provider_unable_boards) == 0

    # ---- COVERAGE gate: every board & status represented, low missing --------
    boards_present = [b for b in REQUIRED_BOARDS if covered_by_board.get(b, 0) > 0]
    boards_absent = [b for b in REQUIRED_BOARDS if covered_by_board.get(b, 0) == 0]
    blocked_by_data = int(cov["blocked_reason"].astype(str).str.startswith("BLOCKED_BY_DATA").sum()) if len(cov) else 0
    partial = int(cov["blocked_reason"].astype(str).str.startswith("PARTIAL_COVERAGE").sum()) if len(cov) else 0
    delisted_covered = int(covered[covered["current_status"] == "delisted"].shape[0]) if len(covered) else 0
    manifest = _load_json(MANIFEST) or {}
    null_close = int(manifest.get("pit_checks", {}).get("null_close", -1))
    gates["coverage"] = {
        "boards_present": boards_present, "boards_absent": boards_absent,
        "covered_by_board": covered_by_board,
        "blocked_by_data": blocked_by_data, "partial_coverage": partial,
        "delisted_names_covered": delisted_covered,
        "main_board_reported": covered_by_board.get("SH_Main", 0) + covered_by_board.get("SZ_Main", 0),
        "chinext_reported": covered_by_board.get("ChiNext", 0),
        "star_reported": covered_by_board.get("STAR", 0),
        "bse_reported": covered_by_board.get("BSE", 0),
        "panel_null_close": null_close,
    }
    coverage_ok = provider_ok and not boards_absent and blocked_by_data == 0 and partial == 0

    # ---- PIT gate: mandatory execution fields present ------------------------
    avail = pit.get("pit_field_availability", {})
    blocked_pit = {k: v for k, v in avail.items() if str(v).startswith("BLOCKED_BY_DATA")}
    gates["pit"] = {
        "field_availability": avail,
        "blocked_fields": sorted(blocked_pit),
        "st_history": "PRESENT" if "st_intervals" not in blocked_pit else "BLOCKED_BY_DATA",
        "suspension_history": "PRESENT" if "suspension_intervals" not in blocked_pit else "BLOCKED_BY_DATA",
        "delisting_status": "PRESENT" if "delisting_date" not in blocked_pit else "BLOCKED_BY_DATA",
        "board_price_limits": "PARTIAL(current-snapshot)",
        "ipo_special_limit": "PRESENT",
        "corporate_actions": "PRESENT" if "corporate_action_identity" not in blocked_pit else "BLOCKED_BY_DATA",
    }
    pit_ok = len(blocked_pit) == 0

    # ---- decide (precedence) -------------------------------------------------
    if state is None:
        if not provider_ok:
            state = "FULL_UNIVERSE_DATA_NOT_READY_PROVIDER"
        elif not coverage_ok:
            state = "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE"
        elif not pit_ok:
            state = "FULL_UNIVERSE_DATA_NOT_READY_PIT"
        else:
            state = "FULL_UNIVERSE_DATA_READY"

    cert = {
        "generated": now, "experiment": "H-031 Track U0 audit",
        "data_readiness_state": state,
        "training_permitted": state == "FULL_UNIVERSE_DATA_READY",
        "gates": gates,
        "gate_pass": {"integration": not integration_missing, "provider": provider_ok,
                      "coverage": coverage_ok, "pit": pit_ok},
        "coverage_summary": {k: cov_summary.get(k) for k in
                             ("master_securities", "covered_bar_history", "blocked_by_data",
                              "tickflow_empty", "by_board_covered") if k in cov_summary},
        "state_precedence": "INTEGRATION > PROVIDER > COVERAGE > PIT > READY",
        "blinding": "no candidate performance included",
    }
    U0.mkdir(parents=True, exist_ok=True)
    (U0 / "full_universe_readiness_certificate.json").write_text(json.dumps(cert, indent=2))

    md = [f"# full_universe_readiness_report — {state}\n\n",
          f"**Training permitted: {cert['training_permitted']}**  (state precedence: {cert['state_precedence']})\n\n",
          "## Gate results\n\n| gate | pass |\n|---|---|\n"]
    for g, ok in cert["gate_pass"].items():
        md.append(f"| {g} | {'PASS' if ok else 'FAIL'} |\n")
    md.append("\n## Coverage by board (covered bar history)\n\n")
    md.append(f"- Main Board: **{gates['coverage']['main_board_reported']}**\n")
    md.append(f"- ChiNext: **{gates['coverage']['chinext_reported']}**\n")
    md.append(f"- STAR: **{gates['coverage']['star_reported']}**\n")
    md.append(f"- BSE: **{gates['coverage']['bse_reported']}**\n")
    md.append(f"- boards absent from covered set: **{gates['coverage']['boards_absent']}**\n")
    md.append(f"- BLOCKED_BY_DATA securities: **{gates['coverage']['blocked_by_data']}**\n")
    md.append("\n## PIT execution fields\n\n")
    for k in ("st_history", "suspension_history", "delisting_status", "board_price_limits",
              "ipo_special_limit", "corporate_actions"):
        md.append(f"- {k}: {gates['pit'][k]}\n")
    md.append(f"\n**Decision: {state}** — no model training may begin unless the state is "
              "FULL_UNIVERSE_DATA_READY. Missing sources are BLOCKED_BY_DATA, not default-false.\n")
    (U0 / "full_universe_readiness_report.md").write_text("".join(md))
    return cert


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    c = audit()
    print(json.dumps({"data_readiness_state": c["data_readiness_state"],
                      "training_permitted": c["training_permitted"],
                      "gate_pass": c["gate_pass"]}, indent=2))
    return 0 if c["training_permitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
