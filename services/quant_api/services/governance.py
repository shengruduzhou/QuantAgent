from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from services.quant_api.config import ApiSettings

# Performance tokens that must NEVER appear in a governance payload. The upstream
# manifests are performance-free by construction; this is a defensive backstop so
# a future manifest change cannot silently leak candidate-level numbers to the UI.
# Word boundaries avoid false positives ("nav" inside "unavailable").
_BANNED = ("nav", "sharpe", "cagr", "drawdown", "return_pct", "pnl", "calmar", "sortino")
_BANNED_RE = re.compile(r"\b(" + "|".join(_BANNED) + r")\b")


class PerformanceLeakError(RuntimeError):
    """Raised if a governance payload would expose candidate performance."""


class GovernanceService:
    """Read-only surface over frozen operational manifests (H-031).

    Exposes ONLY existence-level and gate-level fields — shadow valid-day count,
    Track-F health, fidelity-certificate hash, S4 readiness, U0 coverage/PIT
    gates, blocked boards, and lineage. It never reads, decrypts or reports
    candidate performance, and asserts that invariant before returning.
    """

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.runtime = settings.runtime_root

    # -- helpers --------------------------------------------------------------
    def _read_json(self, rel: str) -> dict | None:
        path = (self.runtime / rel)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _hash(self, rel: str) -> str | None:
        path = self.runtime / rel
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()[:16]

    def _repo_json(self, rel: str) -> dict | None:
        """branch_lineage etc. live under runtime/reports; read relative to runtime."""
        return self._read_json(rel)

    # -- sections -------------------------------------------------------------
    def _shadow(self) -> dict[str, Any]:
        reg = self._read_json("paper/fresh_blind/shadow_day_registry.json")
        acc = self._read_json("paper/fresh_blind/shadow_accumulating_status.json")
        cert = self._read_json("paper/fresh_blind/shadow_test_certificate.json")
        if reg is None:
            return {"status": "unavailable",
                    "reason": "shadow_day_registry.json not found; run validate-shadow-days"}
        decision = "FROZEN_BLIND_PAPER_ACTIVE" if cert else "SHADOW_TEST_ACCUMULATING"
        excluded = [{"date": d["trade_date"], "reason": d["invalid_reason"]}
                    for d in reg.get("days", []) if not d.get("valid_shadow_day")]
        return {
            "status": "ready",
            "decision": decision,
            "validDays": reg.get("valid_shadow_days"),
            "requiredDays": reg.get("required_days"),
            "validDates": reg.get("valid_dates", []),
            "excludedDates": excluded,
            "nextExpectedValidDate": (acc or {}).get("next_expected_valid_date"),
            "ledgerChainValid": reg.get("ledger_chain_valid"),
            "ledgerRecordsTotal": reg.get("ledger_records_total"),
            "fidelityCertificatePasses": reg.get("fidelity_certificate_passes"),
            "fidelityCertificateHash": (reg.get("certificate_sha256") or "")[:16] or None,
            "unblindOrNonRoutineAccesses": len(reg.get("unblind_or_nonroutine_accesses", []) or []),
            "certificateWritten": cert is not None,
        }

    def _s4(self) -> dict[str, Any]:
        cert = self._read_json("reports/h030/s4_readiness_certificate.json")
        rever = self._read_json("reports/h031/s4_reverification.json")
        if cert is None:
            return {"status": "unavailable",
                    "reason": "s4_readiness_certificate.json not found; run certify-s4-batch-replay"}
        return {
            "status": "ready",
            "decision": cert.get("decision"),
            "exactReproduction": cert.get("exact_reproduction_vs_frozen_trace"),
            "deterministic": cert.get("deterministic_double_run"),
            "archivedInputsComplete": cert.get("archived_inputs_complete"),
            "refitCutoffsReplayed": cert.get("refit_cutoffs_replayed"),
            "semanticsChanged": cert.get("semantics_changed"),
            "freshAccess": cert.get("fresh_access"),
            "reverified": rever is not None,
            "codeOrTraceHashChanged": (rever or {}).get("code_or_trace_hash_changed_since_h030"),
        }

    def _u0(self) -> dict[str, Any]:
        """Composite U0 readiness, read from the evidence-driven certificate.

        Every field here traces to an artifact a real acquisition or validation
        run produced; a missing artifact surfaces as ``missingEvidence`` rather
        than being smoothed over with a default.
        """
        cert = self._read_json("data/u0/full_universe_readiness_certificate.json")
        if cert is None:
            return {"status": "unavailable",
                    "reason": "full_universe_readiness_certificate.json not found; "
                              "run audit-u0-full-universe"}
        gates = cert.get("gates", {})
        coverage = gates.get("coverage", {})
        identity = gates.get("identity", {})
        provider = gates.get("provider", {})
        quality = gates.get("quality", {})
        pit_gate = gates.get("pit", {})
        panel = self._read_json("data/u0/panel/panel_manifest.json") or {}
        return {
            "status": "ready",
            "dataReadinessState": cert.get("data_readiness_state"),
            "trainingPermitted": cert.get("training_permitted"),
            "gatePass": cert.get("gate_pass", {}),
            "missingEvidence": cert.get("missing_evidence", []),
            "evidenceSources": cert.get("evidence_sources", {}),
            "coverageByBoard": coverage.get("by_board", {}),
            "coverageByStatus": coverage.get("by_status", {}),
            "boardsAbsent": coverage.get("boards_with_zero_coverage", []),
            "coveredSecurities": coverage.get("covered_securities"),
            "masterSecurities": coverage.get("master_securities"),
            "coverageShare": coverage.get("coverage_share"),
            "notYetAcquired": coverage.get("not_yet_acquired"),
            "identity": {
                "securities": identity.get("securities"),
                "bseCurrent920": identity.get("bse_current_920"),
                "bseLegacyCodes": identity.get("bse_legacy_codes"),
                "delistedInMaster": identity.get("delisted_in_master"),
                "symbolNormalisation": identity.get("symbol_normalisation"),
            },
            "provider": {
                "servingProvidersByFamily": provider.get("serving_providers_by_family", {}),
                "familiesWithoutProvider": provider.get("families_without_provider", []),
                "fallbackProvidersExercised": provider.get("fallback_providers_exercised"),
                "fallbackSymbolsServed": provider.get("fallback_provider_symbols_served"),
                "environmentBlockers": provider.get("environment_blockers", []),
            },
            "quality": {
                "verdicts": quality.get("verdicts", {}),
                "failures": quality.get("failures", []),
                "notRun": quality.get("not_run", []),
                "adjustmentMethod": quality.get("adjustment_method"),
                "volumeUnit": quality.get("volume_unit"),
                "amountUnit": quality.get("amount_unit"),
                "amountCoverage": quality.get("amount_coverage"),
            },
            "pitFieldAvailability": pit_gate.get("field_availability", {}),
            "blockedPitFields": pit_gate.get("blocked_fields", []),
            "suspensionCoverageWindow": pit_gate.get("suspension_coverage_window"),
            "panel": {
                "sha256": cert.get("panel_sha256"),
                "rows": (panel.get("quality_checks") or {}).get("rows"),
                "symbols": (panel.get("quality_checks") or {}).get("symbols"),
                "dateRange": [(panel.get("quality_checks") or {}).get("min_date"),
                              (panel.get("quality_checks") or {}).get("max_date")],
                "sessionGapsSuspended": (panel.get("quality_checks") or {}).get("session_gaps_suspended"),
                "sessionGapsUnexplained": (panel.get("quality_checks") or {}).get("session_gaps_missing_unexplained"),
                "sessionGapsProviderTruncated": (panel.get("quality_checks") or {})
                .get("session_gaps_provider_history_truncated"),
                "ohlcViolationsQuarantined": (panel.get("quality_checks") or {})
                .get("ohlc_relationship_violations"),
                "servingProviderCounts": panel.get("serving_provider_counts", {}),
            },
        }

    def _ashare_foundation(self) -> dict[str, Any]:
        """Provider capability / entitlement matrix and acquisition provenance."""
        capability = self._read_json("data/u0/capability/provider_capability_matrix.json")
        intraday = self._read_json("data/u0/intraday/intraday_manifest.json")
        forensics = self._read_json("data/u0/validation/adjustment_forensics.json")
        validation = self._read_json("data/u0/validation/validation_report.json")
        master = self._read_json("data/u0/security_master_manifest.json")
        if capability is None and validation is None:
            return {"status": "unavailable",
                    "reason": "no capability matrix or validation report; run "
                              "probe-ashare-capabilities and validate-u0-data"}
        out: dict[str, Any] = {"status": "ready"}
        if capability:
            out["capability"] = {
                "probes": capability.get("probes"),
                "supportedProbes": capability.get("supported_probes"),
                "providersWithAnySupport": capability.get("providers_with_any_support", []),
                "servingProvidersByFamily": capability.get("serving_providers_by_family", {}),
                "familiesWithoutAnyProvider": capability.get("families_without_any_provider", []),
                "blockers": capability.get("blockers", []),
                "environment": capability.get("environment", {}),
            }
        if master:
            out["securityMaster"] = {
                "securities": master.get("securities"),
                "byBoard": master.get("by_board", {}),
                "byStatus": master.get("by_status", {}),
                "byOrigin": master.get("by_origin", {}),
                "currentStNames": master.get("current_st_names"),
                "delistingDateCoverage": master.get("delisting_date_coverage"),
                "precedence": master.get("precedence"),
            }
        if intraday:
            out["intraday"] = {
                "frequencyMinutes": intraday.get("frequency_minutes"),
                "symbolsWithBars": intraday.get("symbols_with_bars"),
                "rows": intraday.get("rows"),
                "symbolSessions": intraday.get("symbol_sessions"),
                "byBoard": intraday.get("by_board", {}),
                "servingProviders": intraday.get("serving_providers", {}),
                "depthLimitation": intraday.get("depth_limitation"),
            }
        if forensics:
            out["adjustmentForensics"] = {
                "method": forensics.get("method"),
                "results": forensics.get("results", []),
            }
        if validation:
            out["validation"] = {
                "panelRows": validation.get("panel_rows"),
                "panelSymbols": validation.get("panel_symbols"),
                "dateRange": validation.get("date_range", []),
                "verdicts": validation.get("verdicts", {}),
            }
        return out

    def _u0_h032b(self) -> dict[str, Any]:
        """H-032B: bar readiness vs strict PIT readiness reported SEPARATELY."""
        bar = self._read_json("data/u0/u0_bar_readiness_certificate.json")
        pit = self._read_json("data/u0/u0_strict_pit_certificate.json")
        bench = self._read_json("reports/h032b/tickflow_capability_benchmark.json")
        bse = self._read_json("data/u0/bse_identity_audit.json")
        src = self._read_json("data/u0/pit_source_audit.json")
        out: dict[str, Any] = {"status": "ready" if (bar or pit) else "unavailable"}
        if bar:
            out["barReadiness"] = {
                "decision": bar.get("decision"),
                "gatePass": bar.get("gate_pass", {}),
                "coveredByBoard": bar.get("coverage", {}).get("covered_by_board", {}),
                "boardsAbsent": bar.get("coverage", {}).get("boards_absent", []),
                "fetchableBacklog": bar.get("coverage", {}).get("fetchable_not_probed_backlog"),
                "panelSha256": bar.get("panel", {}).get("sha256"),
            }
        if pit:
            out["strictPitReadiness"] = {
                "decision": pit.get("decision"),
                "trainingPermitted": pit.get("training_permitted"),
                "blockedPitFields": pit.get("blocked_pit_fields", []),
            }
        if src:
            out["pitSourceAudit"] = {f: v.get("tickflow") for f, v in (src.get("fields") or {}).items()}
        if bench:
            d = bench.get("diagnosis", {})
            out["tickflowBenchmark"] = {
                "sdkVersion": bench.get("sdk_version"),
                "count10000Works": d.get("count_10000_works"),
                "batchEntitled": d.get("batch_mode_entitled"),
                "measuredRatePerMin": (bench.get("rate_limit_probe", {})
                                       .get("measured_hard_limit_per_min")),
                "recommendedPath": d.get("recommended_path"),
                "old100BarCause": d.get("old_100_bar_cause"),
            }
        if bse:
            out["bseIdentity"] = {
                "decision": bse.get("identity_decision"),
                "authoritativeCount": bse.get("authoritative_bse_count"),
                "masterCount": bse.get("u0_master_bse_count"),
                "truePlaceholders": bse.get("true_placeholder_codes_in_master"),
                "missingFromMaster": bse.get("in_authoritative_not_master"),
            }
        # H-032C: PIT-metadata sourcing closures, entitlement re-test, reconciliation
        meta = self._read_json("data/u0/pit/pit_metadata_manifest.json")
        if meta:
            out["pitMetadataSourcing"] = {
                "closedFields": meta.get("closed_fields", []),
                "blockedFields": meta.get("blocked_fields", []),
                "delistingDatesSourced": meta.get("delisting_dates_sourced"),
            }
        ent = self._read_json("reports/h032c/tickflow_entitlement_audit.json")
        if ent:
            out["tickflowEntitlement"] = {k: ent.get(k) for k in (
                "count_10000_get", "batch_klines", "ex_factors",
                "measured_rate_limit_per_min", "corporate_action_classification")
                if k in ent} or {"status": ent.get("status")}
        recon = self._read_json("data/u0/universe_reconciliation.json")
        if recon:
            out["reconciliation"] = {
                "supplementalAdditions": recon.get("supplemental_additions"),
                "supplementalSymbols": recon.get("supplemental_additions_symbols", []),
                "dualIdentityCollisions": recon.get("dual_identity_guard", {}).get("dual_identity_collisions"),
                "starCovered": recon.get("star_covered"),
                "starTotal": recon.get("star_total"),
            }
        return out

    def _lineage(self) -> dict[str, Any]:
        lin = self._read_json("reports/h031/branch_lineage.json")
        if lin is None:
            return {"status": "unavailable", "reason": "branch_lineage.json not found"}
        return {
            "status": "ready",
            "headCommit": lin.get("head_commit"),
            "originMainCommit": lin.get("origin_main_commit"),
            "headEqualsOriginMain": lin.get("head_equals_origin_main"),
            "h030RemotelyRecoverable": lin.get("h030_remotely_recoverable"),
            "overlappingFiles": lin.get("overlapping_files", []),
            "expectedConflictAreas": lin.get("expected_conflict_areas", []),
            "integrationBranch": lin.get("integration_branch"),
        }

    def _governed_commands(self) -> list[dict[str, Any]]:
        from services.quant_api.services.jobs import COMMANDS
        ids = ("validate-shadow-days", "certify-s4-batch-replay", "build-u0-security-master",
               "report-u0-provider-coverage", "assemble-u0-full-universe",
               "audit-u0-full-universe", "backfill-u0-market-panel", "probe-u0-star-bse",
               "benchmark-tickflow-capability", "audit-bse-identity",
               "audit-u0-pit-readiness", "report-u0-bar-readiness",
               "source-u0-pit-metadata", "audit-tickflow-entitlement",
               "report-u0-reconciliation",
               # A-share data foundation pipeline, in execution order
               "probe-ashare-capabilities", "build-u0-live-security-master",
               "acquire-u0-daily-bars", "build-u0-pit-intervals",
               "acquire-u0-intraday-bars", "assemble-u0-raw-panel",
               "validate-u0-data", "audit-u0-adjustment-forensics")
        out = []
        for cid in ids:
            spec = COMMANDS.get(cid)
            if not spec:
                continue
            out.append({
                "commandId": cid, "type": spec["type"],
                "requiresNetwork": bool(spec.get("control")),
                "parameters": sorted(spec.get("allowed", set())),
            })
        return out

    def status(self) -> dict[str, Any]:
        payload = {
            "shadow": self._shadow(),
            "s4": self._s4(),
            "u0": self._u0(),
            "u0BarPit": self._u0_h032b(),
            "ashareFoundation": self._ashare_foundation(),
            "lineage": self._lineage(),
            "governedCommands": self._governed_commands(),
            "blinding": "existence- and gate-level fields only; no candidate performance",
        }
        self._assert_no_performance(payload)
        return payload

    @staticmethod
    def _assert_no_performance(payload: dict) -> None:
        match = _BANNED_RE.search(json.dumps(payload).lower())
        if match:
            raise PerformanceLeakError(f"governance payload leaked performance field: {match.group(1)}")
