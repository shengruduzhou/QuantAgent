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
        cert = self._read_json("data/u0/full_universe_readiness_certificate.json")
        cov = self._read_json("data/u0/provider_coverage_summary.json")
        pit = self._read_json("data/u0/pit_field_availability.json")
        manifest = self._read_json("data/v7/full_universe/full_universe_manifest.json")
        if cert is None:
            return {"status": "unavailable",
                    "reason": "full_universe_readiness_certificate.json not found; run audit-u0-full-universe"}
        gates = cert.get("gates", {})
        coverage_gate = gates.get("coverage", {})
        backfill = None
        if manifest:
            backfill = {
                "masterSecurities": manifest.get("master_securities"),
                "panelSymbols": manifest.get("panel_symbols"),
                "missingSymbols": manifest.get("missing_symbols"),
                "stagedBackfillFiles": manifest.get("staged_backfill_files"),
            }
        return {
            "status": "ready",
            "dataReadinessState": cert.get("data_readiness_state"),
            "trainingPermitted": cert.get("training_permitted"),
            "gatePass": cert.get("gate_pass", {}),
            "coverageByBoard": coverage_gate.get("covered_by_board", {}),
            "boardsAbsent": coverage_gate.get("boards_absent", []),
            "blockedByData": coverage_gate.get("blocked_by_data"),
            "pitGate": {k: gates.get("pit", {}).get(k) for k in (
                "st_history", "suspension_history", "delisting_status",
                "board_price_limits", "ipo_special_limit", "corporate_actions")},
            "pitFieldAvailability": (pit or {}).get("pit_field_availability", {}),
            "coveredBarHistory": (cov or {}).get("covered_bar_history"),
            "backfill": backfill,
        }

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
               "audit-u0-full-universe", "backfill-u0-market-panel")
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
