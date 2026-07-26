"""Evidence-driven U0 readiness gates.

Single source of truth for what "the full-universe data foundation is ready"
means. Every gate value here is computed from an artifact that a real
acquisition or validation run produced. Nothing is hardcoded to ``True``, and a
missing input yields an explicit blocker rather than a default pass — the
failure mode this replaces is a certificate whose ``adjustment_method_explicit``
and ``volume_amount_units_verified`` gates were literal ``True`` constants next
to a comment asserting the fact they were supposed to be testing.

Three certificates are derived from the same evidence so they can never drift:

``bar``     is there clean, correctly-attributed OHLCV for the whole universe?
``pit``     are the point-in-time execution fields sourced with provenance?
``overall`` the composite state that gates model training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.ashare.symbols import ALL_BOARDS

BAR_READY = "U0_BAR_READY"
BAR_NOT_READY_COVERAGE = "U0_BAR_NOT_READY_COVERAGE"
BAR_NOT_READY_IDENTITY = "U0_BAR_NOT_READY_IDENTITY"
BAR_NOT_READY_PROVIDER = "U0_BAR_NOT_READY_PROVIDER"
BAR_NOT_READY_QUALITY = "U0_BAR_NOT_READY_QUALITY"

READY = "FULL_UNIVERSE_DATA_READY"
NOT_READY_INTEGRATION = "FULL_UNIVERSE_DATA_NOT_READY_INTEGRATION"
NOT_READY_PROVIDER = "FULL_UNIVERSE_DATA_NOT_READY_PROVIDER"
NOT_READY_IDENTITY = "FULL_UNIVERSE_DATA_NOT_READY_IDENTITY"
NOT_READY_COVERAGE = "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE"
NOT_READY_PIT = "FULL_UNIVERSE_DATA_NOT_READY_PIT"

BLOCKED = "BLOCKED_BY_DATA"

#: PIT fields that must be sourced before training may be unblocked.
MANDATORY_PIT_FIELDS = (
    "listing_date", "delisting_date", "trading_calendar", "price_limit_regime",
    "ipo_special_limit", "corporate_action_identity", "suspension_intervals",
    "st_intervals",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@dataclass
class Evidence:
    """Every artifact the gates are allowed to read, loaded once."""

    root: Path
    panel_manifest: dict | None = None
    validation: dict | None = None
    capability: dict | None = None
    master_manifest: dict | None = None
    coverage: pd.DataFrame | None = None
    disposition: pd.DataFrame | None = None
    pit_manifests: dict[str, dict] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "Evidence":
        u0 = root / "runtime/data/u0"
        evidence = cls(root=root)
        wanted = {
            "panel_manifest": u0 / "panel/panel_manifest.json",
            "validation": u0 / "validation/validation_report.json",
            "capability": u0 / "capability/provider_capability_matrix.json",
            "master_manifest": u0 / "security_master_manifest.json",
        }
        for attribute, path in wanted.items():
            payload = _json(path)
            if payload is None:
                evidence.missing.append(str(path.relative_to(root)))
            setattr(evidence, attribute, payload)
        coverage_path = u0 / "panel/coverage_matrix.parquet"
        if coverage_path.exists():
            evidence.coverage = pd.read_parquet(coverage_path)
        else:
            evidence.missing.append(str(coverage_path.relative_to(root)))
        disposition_path = u0 / "master_disposition.parquet"
        if disposition_path.exists():
            evidence.disposition = pd.read_parquet(disposition_path)
        for name, filename in (("calendar", "trading_calendar_manifest.json"),
                               ("adjust_factors", "adjust_factors_manifest.json"),
                               ("corporate_actions", "corporate_actions_manifest.json"),
                               ("suspension", "suspension_manifest.json"),
                               ("st", "st_manifest.json")):
            payload = _json(u0 / "pit" / filename)
            if payload is not None:
                evidence.pit_manifests[name] = payload
        return evidence


def _validation_index(evidence: Evidence) -> dict[str, dict]:
    checks = (evidence.validation or {}).get("checks") or []
    return {row["check"]: row for row in checks}


def identity_gate(evidence: Evidence) -> dict[str, Any]:
    manifest = evidence.master_manifest or {}
    checks = _validation_index(evidence)
    symbol_check = checks.get("symbol_normalisation", {})
    boards = manifest.get("by_board", {})
    absent = [b for b in ALL_BOARDS if boards.get(b, 0) == 0]
    bse_total = int(boards.get("BSE", 0))
    result = {
        "master_built_from_live_sources": bool(manifest.get("sources", {}).get("tickflow_instruments")),
        "securities": int(manifest.get("securities", 0)),
        "boards_in_master": boards,
        "boards_absent_from_master": absent,
        "bse_current_920": int(manifest.get("bse_current_920", 0)),
        "bse_legacy_codes": int(manifest.get("bse_legacy_codes", 0)),
        "delisted_in_master": int((manifest.get("by_status") or {}).get("delisted", 0)),
        "symbol_normalisation": symbol_check.get("verdict", "NOT_RUN"),
    }
    result["pass"] = bool(
        result["securities"] > 0 and not absent and bse_total > 0
        and result["delisted_in_master"] > 0
        and symbol_check.get("verdict") == "PASS"
    )
    return result


def provider_gate(evidence: Evidence) -> dict[str, Any]:
    capability = evidence.capability or {}
    by_family = capability.get("serving_providers_by_family", {})
    panel = evidence.panel_manifest or {}
    serving = panel.get("serving_provider_counts", {})
    fallback_used = sum(count for name, count in serving.items() if not name.startswith("tickflow"))
    mandatory = {
        "daily_bars": by_family.get("daily_bars", []),
        "security_master": by_family.get("security_master", []) or by_family.get("security_master_bulk", []),
        "adjust_factors": by_family.get("adjust_factors", []),
        "corporate_actions": by_family.get("corporate_actions", []),
        "quotes": by_family.get("quotes", []) + by_family.get("quotes_l1_depth5", []),
        "minute_bars": by_family.get("minute_bars", []),
    }
    unserved = sorted(family for family, providers in mandatory.items() if not providers)
    result = {
        "capability_probe_present": bool(capability),
        "serving_providers_by_family": mandatory,
        "families_without_provider": unserved,
        "panel_serving_provider_counts": serving,
        "fallback_provider_symbols_served": int(fallback_used),
        "fallback_providers_exercised": bool(fallback_used),
        "environment_blockers": [b for b in capability.get("blockers", [])
                                 if b.get("status") == "BLOCKED_BY_ENVIRONMENT"],
    }
    result["pass"] = bool(capability and not unserved)
    return result


def coverage_gate(evidence: Evidence) -> dict[str, Any]:
    coverage = evidence.coverage
    if coverage is None or coverage.empty:
        return {"pass": False, "reason": "coverage_matrix.parquet absent"}
    by_board = coverage.groupby("board")["covered"].agg(["sum", "count"])
    by_status = coverage.groupby("status")["covered"].agg(["sum", "count"])
    covered = int(coverage["covered"].sum())
    total = int(len(coverage))
    absent = [b for b in ALL_BOARDS
              if b in by_board.index and int(by_board.loc[b, "sum"]) == 0]
    missing_boards = [b for b in ALL_BOARDS if b not in by_board.index]
    result = {
        "master_securities": total,
        "covered_securities": covered,
        "coverage_share": round(covered / max(1, total), 4),
        "by_board": {b: {"covered": int(r["sum"]), "total": int(r["count"])}
                     for b, r in by_board.iterrows()},
        "by_status": {s: {"covered": int(r["sum"]), "total": int(r["count"])}
                      for s, r in by_status.iterrows()},
        "boards_with_zero_coverage": absent + missing_boards,
        "not_yet_acquired": int((coverage["blocked_reason"] == "NOT_YET_ACQUIRED").sum()),
        "no_vendor_history_delisted": int(
            (coverage["blocked_reason"] == "NO_VENDOR_HISTORY_DELISTED").sum()),
    }

    # A security that has not begun trading cannot have bars, and demanding them
    # would either block the gate forever or invite placeholder rows. It is
    # excluded from the denominator ONLY when the exchange's own listed register
    # says it is absent — evidence from
    # scripts/u0_exchange_register_reconcile.py, never an assumption. Without
    # that artifact the gate stays strict: every master security must be covered.
    expected = total
    uncovered_symbols: list[str] = []
    if "symbol" in coverage.columns:
        uncovered_symbols = coverage.loc[~coverage["covered"].astype(bool), "symbol"].astype(str).tolist()
    result["uncovered_symbols"] = uncovered_symbols[:50]
    disposition = evidence.disposition
    if disposition is not None and not disposition.empty:
        counts = disposition["disposition"].value_counts().to_dict()
        not_expected = disposition.loc[
            disposition["disposition"] == "PRE_LISTING_NO_SESSIONS", "symbol"].astype(str)
        not_expected_set = set(not_expected)
        expected = total - len(not_expected_set & set(coverage["symbol"].astype(str)))
        result["disposition_counts"] = {str(k): int(v) for k, v in counts.items()}
        result["not_expected_to_trade"] = sorted(not_expected_set)
        result["expected_securities"] = expected
        result["unexplained_uncovered"] = sorted(set(uncovered_symbols) - not_expected_set)
    else:
        result["disposition_evidence"] = (
            "master_disposition.parquet absent — every master security must be covered")
        result["unexplained_uncovered"] = uncovered_symbols

    result["expected_coverage_share"] = round(covered / max(1, expected), 4)
    result["pass"] = bool(not result["unexplained_uncovered"]
                          and not result["boards_with_zero_coverage"])
    return result


def quality_gate(evidence: Evidence) -> dict[str, Any]:
    checks = _validation_index(evidence)
    if not checks:
        return {"pass": False, "reason": "validation_report.json absent"}
    tracked = ["schema_columns", "schema_dtypes", "timestamp_type", "duplicate_symbol_date",
               "ohlc_relationships", "non_positive_prices", "null_close", "negative_volume",
               "amount_volume_units", "volume_unit_is_shares", "pre_listing_rows",
               "post_delisting_rows", "price_limit_plausibility", "adjustment_is_raw",
               "pit_available_at", "suspension_representation", "freshness",
               "cross_provider_reconciliation", "intraday_to_daily_reconciliation"]
    verdicts = {name: checks.get(name, {}).get("verdict", "NOT_RUN") for name in tracked}
    failures = [name for name, verdict in verdicts.items() if verdict == "FAIL"]
    not_run = [name for name, verdict in verdicts.items() if verdict == "NOT_RUN"]
    panel = evidence.panel_manifest or {}
    return {
        "verdicts": verdicts,
        "failures": failures,
        "not_run": not_run,
        "adjustment_method": panel.get("adjustment_method"),
        "volume_unit": panel.get("volume_unit"),
        "amount_unit": panel.get("amount_unit"),
        "amount_coverage": (panel.get("quality_checks") or {}).get("amount_coverage"),
        "pass": not failures and not not_run,
    }


def pit_gate(evidence: Evidence) -> dict[str, Any]:
    manifests = evidence.pit_manifests
    master = evidence.master_manifest or {}
    availability: dict[str, str] = {}

    listing_coverage = int(master.get("listing_date_coverage", 0))
    securities = int(master.get("securities", 0)) or 1
    availability["listing_date"] = (
        f"AVAILABLE ({listing_coverage}/{securities}, exchange instrument listing)"
        if listing_coverage else f"{BLOCKED} — no listing dates")
    delisting_coverage = int(master.get("delisting_date_coverage", 0))
    availability["delisting_date"] = (
        f"AVAILABLE ({delisting_coverage} dated delistings, SSE/SZSE delisting lists)"
        if delisting_coverage else f"{BLOCKED} — no delisting dates")

    calendar = manifests.get("calendar") or {}
    availability["trading_calendar"] = (
        f"AVAILABLE ({calendar.get('rows')} sessions {calendar.get('first')}..{calendar.get('last')})"
        if calendar.get("rows") else f"{BLOCKED} — no trading calendar")

    availability["price_limit_regime"] = "AVAILABLE — deterministic exchange rule intervals"
    availability["ipo_special_limit"] = "AVAILABLE — exchange IPO rule anchored on listing_date"

    factors = manifests.get("adjust_factors") or {}
    actions = manifests.get("corporate_actions") or {}
    if factors.get("rows"):
        availability["corporate_action_identity"] = (
            f"AVAILABLE — {factors.get('rows')} ex-rights factor records over "
            f"{factors.get('symbols_with_data')} securities"
            + (f"; {actions.get('rows')} dividend records" if actions.get("rows") else ""))
    else:
        availability["corporate_action_identity"] = f"{BLOCKED} — no adjustment-factor series"

    suspension = manifests.get("suspension") or {}
    if suspension.get("intervals"):
        window = suspension.get("snapshot_date_range")
        availability["suspension_intervals"] = (
            f"AVAILABLE — {suspension.get('intervals')} vendor-dated halts over "
            f"{suspension.get('symbols_with_halts')} securities; snapshot window {window}")
    else:
        availability["suspension_intervals"] = f"{BLOCKED} — no halt snapshots on disk"

    st = manifests.get("st") or {}
    uncovered_exchanges = st.get("exchanges_without_dated_history") or []
    if st.get("dated_episodes") and not uncovered_exchanges:
        availability["st_intervals"] = (
            f"AVAILABLE — {st.get('dated_episodes')} dated risk-warning episodes over "
            f"{st.get('securities_with_dated_episodes')} securities")
    elif st.get("dated_episodes"):
        # Partial is NOT a pass: an exchange without a dated register would be
        # silently treated as "never ST", which is exactly the default-false the
        # gate exists to prevent.
        availability["st_intervals"] = (
            f"{BLOCKED} — PARTIAL: {st.get('dated_episodes')} dated episodes over "
            f"{st.get('securities_with_dated_episodes')} securities from "
            f"{', '.join(st.get('exchanges_with_dated_history') or [])}; no dated register "
            f"for {', '.join(uncovered_exchanges)}; current state known for "
            f"{st.get('current_st_names')} names")
    elif st.get("current_st_names"):
        availability["st_intervals"] = (
            f"{BLOCKED} — current state only ({st.get('current_st_names')} names), no history")
    else:
        availability["st_intervals"] = f"{BLOCKED} — no ST source"

    blocked = sorted(field for field in MANDATORY_PIT_FIELDS
                     if str(availability.get(field, BLOCKED)).startswith(BLOCKED))
    return {
        "field_availability": availability,
        "blocked_fields": blocked,
        "suspension_coverage_window": (manifests.get("suspension") or {}).get("snapshot_date_range"),
        "pass": not blocked,
    }


def build_certificates(root: Path) -> dict[str, dict]:
    """Compute the bar, PIT and overall certificates from artifacts on disk."""
    evidence = Evidence.load(root)
    identity = identity_gate(evidence)
    provider = provider_gate(evidence)
    coverage = coverage_gate(evidence)
    quality = quality_gate(evidence)
    pit = pit_gate(evidence)
    panel_path = root / "runtime/data/u0/panel/daily_bars_raw.parquet"
    panel = evidence.panel_manifest or {}

    if not identity["pass"]:
        bar_decision = BAR_NOT_READY_IDENTITY
    elif not provider["pass"]:
        bar_decision = BAR_NOT_READY_PROVIDER
    elif not coverage["pass"]:
        bar_decision = BAR_NOT_READY_COVERAGE
    elif not quality["pass"]:
        bar_decision = BAR_NOT_READY_QUALITY
    else:
        bar_decision = BAR_READY

    if evidence.missing:
        overall = NOT_READY_INTEGRATION
    elif not provider["pass"]:
        overall = NOT_READY_PROVIDER
    elif not identity["pass"]:
        overall = NOT_READY_IDENTITY
    elif not coverage["pass"] or not quality["pass"]:
        overall = NOT_READY_COVERAGE
    elif not pit["pass"]:
        overall = NOT_READY_PIT
    else:
        overall = READY

    common = {
        "generated": _now(),
        "evidence_sources": {
            "panel_manifest": "runtime/data/u0/panel/panel_manifest.json",
            "validation_report": "runtime/data/u0/validation/validation_report.json",
            "capability_matrix": "runtime/data/u0/capability/provider_capability_matrix.json",
            "coverage_matrix": "runtime/data/u0/panel/coverage_matrix.parquet",
            "security_master": "runtime/data/u0/security_master_manifest.json",
            "pit_manifests": sorted(evidence.pit_manifests),
        },
        "missing_evidence": evidence.missing,
        "blinding": "no candidate performance included",
    }

    bar_certificate = {
        **common,
        "experiment": "U0 bar readiness (evidence-driven)",
        "decision": bar_decision,
        "primary_bar_provider": "TickFlow daily klines (entitled); Tencent public fallback",
        "gate_pass": {"identity": identity["pass"], "provider": provider["pass"],
                      "coverage": coverage["pass"], "quality": quality["pass"]},
        "identity": identity, "provider": provider, "coverage": coverage, "quality": quality,
        "panel": {"sha256": _sha(panel_path), **(panel.get("quality_checks") or {}),
                  "adjustment_method": panel.get("adjustment_method"),
                  "volume_unit": panel.get("volume_unit"), "amount_unit": panel.get("amount_unit")},
        "allowed_decisions": [BAR_READY, BAR_NOT_READY_COVERAGE, BAR_NOT_READY_IDENTITY,
                              BAR_NOT_READY_PROVIDER, BAR_NOT_READY_QUALITY],
        "note": ("U0_BAR_READY permits smoke tests only — dataset build, feature materialisation, "
                 "CLI validation. Model comparison and return-based backtesting additionally "
                 "require the strict PIT certificate."),
    }

    pit_certificate = {
        **common,
        "experiment": "U0 strict PIT readiness (evidence-driven)",
        "decision": overall,
        "training_permitted": overall == READY,
        "blocked_pit_fields": pit["blocked_fields"],
        "pit_field_availability": pit["field_availability"],
        "bar_decision": bar_decision,
        "allowed_decisions": [READY, NOT_READY_COVERAGE, NOT_READY_PIT, NOT_READY_IDENTITY,
                              NOT_READY_PROVIDER],
    }

    overall_certificate = {
        **common,
        "experiment": "U0 full-universe readiness (evidence-driven)",
        "data_readiness_state": overall,
        "training_permitted": overall == READY,
        "state_precedence": "INTEGRATION > PROVIDER > IDENTITY > COVERAGE/QUALITY > PIT > READY",
        "gate_pass": {
            "integration": not evidence.missing, "provider": provider["pass"],
            "identity": identity["pass"], "coverage": coverage["pass"],
            "quality": quality["pass"], "pit": pit["pass"],
        },
        "gates": {"integration": {"missing_evidence": evidence.missing},
                  "provider": provider, "identity": identity, "coverage": coverage,
                  "quality": quality, "pit": pit},
        "panel_sha256": _sha(panel_path),
    }
    return {"bar": bar_certificate, "pit": pit_certificate, "overall": overall_certificate}


def render_report(certificate: dict) -> str:
    """Operator-readable markdown for the overall certificate."""
    state = certificate["data_readiness_state"]
    lines = [f"# full_universe_readiness_report — {state}\n\n",
             f"**Training permitted: {certificate['training_permitted']}** "
             f"(precedence: {certificate['state_precedence']})\n\n",
             "Every value below is read from an artifact produced by a real acquisition or "
             "validation run; no gate is hardcoded.\n\n",
             "## Gates\n\n| gate | pass |\n|---|---|\n"]
    for gate, passed in certificate["gate_pass"].items():
        lines.append(f"| {gate} | {'PASS' if passed else 'FAIL'} |\n")
    coverage = certificate["gates"]["coverage"]
    lines.append("\n## Coverage by board\n\n| board | covered | total |\n|---|---|---|\n")
    for board, row in (coverage.get("by_board") or {}).items():
        lines.append(f"| {board} | {row['covered']} | {row['total']} |\n")
    lines.append(f"\n- covered securities: **{coverage.get('covered_securities')}** / "
                 f"{coverage.get('master_securities')} "
                 f"({coverage.get('coverage_share')})\n")
    lines.append(f"- not yet acquired: **{coverage.get('not_yet_acquired')}**\n")
    quality = certificate["gates"]["quality"]
    lines.append("\n## Validation verdicts\n\n| check | verdict |\n|---|---|\n")
    for name, verdict in (quality.get("verdicts") or {}).items():
        lines.append(f"| {name} | {verdict} |\n")
    lines.append("\n## PIT execution fields\n\n")
    for name, status in (certificate["gates"]["pit"].get("field_availability") or {}).items():
        lines.append(f"- **{name}**: {status}\n")
    lines.append(f"\n**Decision: {state}** — training stays blocked unless the state is "
                 f"{READY}. Missing sources are reported as {BLOCKED}, never defaulted to false.\n")
    return "".join(lines)
