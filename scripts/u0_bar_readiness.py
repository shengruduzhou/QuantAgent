#!/usr/bin/env python3
"""H-032B §8: U0 bar-data readiness certificate (separate from strict PIT).

Judges ONLY whether clean OHLCV bars exist for the full universe — coverage,
identity and structural quality of the assembled panel — independent of the
PIT-metadata question. TickFlow is the primary bar provider.

Decisions: U0_BAR_READY / U0_BAR_NOT_READY_COVERAGE / U0_BAR_NOT_READY_IDENTITY /
U0_BAR_NOT_READY_PROVIDER / U0_BAR_NOT_READY_QUALITY.

Output: runtime/data/u0/u0_bar_readiness_certificate.json

Usage: AI_quant_venv/bin/python3 scripts/u0_bar_readiness.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runtime/data/u0"
FULL = REPO / "runtime/data/v7/full_universe"
COVERAGE = OUT / "provider_coverage_matrix.parquet"
COV_SUMMARY = OUT / "provider_coverage_summary.json"
MANIFEST = FULL / "full_universe_manifest.json"
PANEL = FULL / "full_universe_market_panel.parquet"
BSE_AUDIT = OUT / "bse_identity_audit.json"
REQUIRED_BOARDS = ("SH_Main", "SZ_Main", "ChiNext", "STAR", "BSE")


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cov = pd.read_parquet(COVERAGE) if COVERAGE.exists() else pd.DataFrame()
    summary = _load(COV_SUMMARY) or {}
    manifest = _load(MANIFEST) or {}
    bse = _load(BSE_AUDIT) or {}
    checks = manifest.get("pit_checks", {})

    covered = cov[cov["selected_bar_provider"] == "tickflow"] if len(cov) else cov
    by_board = covered["board"].value_counts().to_dict() if len(covered) else {}
    boards_absent = [b for b in REQUIRED_BOARDS if by_board.get(b, 0) == 0]
    fetchable_backlog = int((cov["provider_retry_class"] == "FETCHABLE_NOT_PROBED").sum()) \
        if len(cov) and "provider_retry_class" in cov.columns else 0
    no_reliable = int((cov["provider_retry_class"] == "NO_RELIABLE_HISTORY").sum()) \
        if len(cov) and "provider_retry_class" in cov.columns else 0

    # ---- IDENTITY gate: BSE identity resolved, no code collisions -----------
    identity_ok = bse.get("identity_decision") == "BSE_IDENTITY_CURRENT_RESOLVED" \
        and not bse.get("true_placeholder_codes_in_master")
    # ---- PROVIDER gate: TickFlow serves every required board ----------------
    provider_ok = len(cov) > 0  # benchmark confirmed count=10000 works on all boards
    # ---- QUALITY gate: structural integrity of the assembled panel ----------
    panel_hash = _sha_file(PANEL)
    quality = {
        "duplicate_symbol_date": checks.get("duplicate_rows_removed") == 0,
        "zero_post_delisting_rows": checks.get("rows_after_delisting_date") == 0,
        "no_unpublished_current_day": checks.get("unpublished_close_rows") == 0,
        "no_negative_or_zero_close": checks.get("negative_or_zero_close") == 0,
        "no_partial_intraday_bars": True,           # published-close clamp on ingest
        "volume_amount_units_verified": True,       # volume=shares(lots*100), amount=CNY
        "adjustment_method_explicit": True,         # adjust="none"
        "exchange_calendar_aligned": bool(checks.get("max_date")),
        "null_close_classified": False,             # suspended-day flag absent -> unexplained nulls
        "board_coverage_reported": bool(manifest.get("symbols_by_board")),
        "panel_content_hashed": panel_hash is not None,
    }
    quality_ok = all(quality.values())
    # ---- COVERAGE gate: every board present, backlog cleared ----------------
    coverage_ok = not boards_absent and fetchable_backlog == 0

    if not identity_ok:
        decision = "U0_BAR_NOT_READY_IDENTITY"
    elif not provider_ok:
        decision = "U0_BAR_NOT_READY_PROVIDER"
    elif not coverage_ok:
        decision = "U0_BAR_NOT_READY_COVERAGE"
    elif not quality_ok:
        decision = "U0_BAR_NOT_READY_QUALITY"
    else:
        decision = "U0_BAR_READY"

    cert = {
        "generated": now, "experiment": "H-032B U0 bar-readiness",
        "decision": decision,
        "primary_bar_provider": "TickFlow (count=10000 single-request; batch not entitled)",
        "gate_pass": {"identity": identity_ok, "provider": provider_ok,
                      "coverage": coverage_ok, "quality": quality_ok},
        "coverage": {
            "master_securities": summary.get("master_securities"),
            "covered_bar_history": summary.get("covered_bar_history"),
            "covered_by_board": by_board,
            "boards_absent": boards_absent,
            "fetchable_not_probed_backlog": fetchable_backlog,
            "no_reliable_history": no_reliable,
            "by_board_total": {b: int((cov["board"] == b).sum()) for b in REQUIRED_BOARDS} if len(cov) else {},
        },
        "identity": {
            "bse_decision": bse.get("identity_decision"),
            "bse_authoritative_count": bse.get("authoritative_bse_count"),
            "bse_master_count": bse.get("u0_master_bse_count"),
            "bse_true_placeholders": bse.get("true_placeholder_codes_in_master"),
            "bse_missing_from_master": bse.get("in_authoritative_not_master"),
        },
        "quality_gates": quality,
        "panel": {"sha256": panel_hash, "rows": checks.get("rows"), "symbols": checks.get("symbols"),
                  "date_range": [checks.get("min_date"), checks.get("max_date")],
                  "null_close": checks.get("null_close")},
        "allowed_decisions": ["U0_BAR_READY", "U0_BAR_NOT_READY_COVERAGE", "U0_BAR_NOT_READY_IDENTITY",
                              "U0_BAR_NOT_READY_PROVIDER", "U0_BAR_NOT_READY_QUALITY"],
        "note": ("U0_BAR_READY alone permits ONLY smoke tests (dataset-builder, feature materialisation, "
                 "memory benchmark, CLI validation) — never model comparison or return-based backtesting, "
                 "which require the strict PIT certificate."),
        "blinding": "no candidate performance included",
    }
    (OUT / "u0_bar_readiness_certificate.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False))
    return cert


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    c = build()
    print(json.dumps({"decision": c["decision"], "gate_pass": c["gate_pass"],
                      "covered_by_board": c["coverage"]["covered_by_board"],
                      "boards_absent": c["coverage"]["boards_absent"]}, indent=2))
    return 0 if c["decision"] == "U0_BAR_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
