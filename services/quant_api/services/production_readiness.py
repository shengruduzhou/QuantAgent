"""Read-only production-readiness truth for the operator Governance surface.

The service never connects to a broker, probes QMT, instantiates a fresh
KillSwitch, or infers certification from code existence. It reads only fixed
repository/runtime evidence and verifies the existing model-trust certificate.
Missing or malformed evidence remains non-green.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from quantagent.execution.economic_model_gate import evaluate_economic_model_gate
from quantagent.execution.live_model_trust import (
    REQUIRED_METRIC_SEMANTICS,
    evaluate_live_model_trust,
)
from quantagent.execution.live_session import (
    ORDER_RISK_AUTHORIZATION_SEMANTICS,
    TARGET_RISK_AUTHORIZATION_SEMANTICS,
)
from quantagent.safety.operating_mode import describe_policy


@dataclass(frozen=True)
class ReadinessDimension:
    key: str
    label: str
    state: str
    severity: str
    reasons: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class ProductionReadinessService:
    """Build eight independent operator states from machine evidence only."""

    BROKER_QUERY_CERT = Path("certificates/broker_query_readiness.json")
    RECONCILIATION_CERT = Path("certificates/reconciliation.json")
    HOST_CERT = Path("certificates/qmt_host_certification.json")
    KILL_SWITCH_STATE = Path("live/kill_switch_state.json")
    KILL_SWITCH_SCHEMA_VERSION = 1

    def __init__(
        self,
        project_root: str | Path,
        runtime_root: str | Path,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def status(self) -> dict[str, Any]:
        cards = [
            self._model_trust(),
            self._broker_query_readiness(),
            self._target_risk(),
            self._order_risk(),
            self._kill_switch(),
            self._reconciliation(),
            self._product_arming(),
            self._host_certification(),
        ]
        return {
            "schemaVersion": 1,
            "generatedAt": self._now().isoformat(),
            # Deliberately absent as a decision. A green broker-query certificate
            # must never hide a blocked model, reconciliation or product policy.
            "aggregateTradingReady": None,
            "aggregateStateSemantics": "intentionally_not_computed_show_all_dimensions",
            "cards": [card.to_dict() for card in cards],
        }

    def _model_trust(self) -> ReadinessDimension:
        manifest = self.project_root / "configs/live_model_trust.json"
        report = evaluate_live_model_trust(manifest)
        economic_ok, economic_reasons = evaluate_economic_model_gate(report)
        return ReadinessDimension(
            key="modelTrust",
            label="Model Trust",
            state="PASS" if economic_ok else "BLOCKED",
            severity="ok" if economic_ok else "blocked",
            reasons=economic_reasons,
            evidence={
                "modelId": report.model_id,
                "certificateStatus": report.status,
                "trustClass": report.trust_class,
                "evidenceVerificationOk": report.ok,
                "economicLiveEligible": economic_ok,
                "requiredMetricSemantics": REQUIRED_METRIC_SEMANTICS,
                "observedMetricSemantics": report.evidence.get("strict_backtest_metric_semantics"),
                "manifest": self._display_path(manifest),
            },
        )

    def _broker_query_readiness(self) -> ReadinessDimension:
        path, cert, error = self._read_runtime_json(self.BROKER_QUERY_CERT)
        if cert is None:
            return self._missing_or_invalid(
                key="brokerQuery",
                label="Broker Query Readiness",
                path=path,
                error=error,
                missing_state="NOT_CERTIFIED",
            )
        temporal = self._certificate_time_state(cert)
        if temporal is not None:
            return self._temporal_card(
                key="brokerQuery",
                label="Broker Query Readiness",
                path=path,
                cert=cert,
                temporal=temporal,
            )
        required_true = ("query_only_ready", "preflight_ok", "health_ok")
        failures = [name for name in required_true if cert.get(name) is not True]
        return ReadinessDimension(
            key="brokerQuery",
            label="Broker Query Readiness",
            state="READY" if not failures else "BLOCKED",
            severity="ok" if not failures else "blocked",
            reasons=tuple(f"certificate_{name}_not_true" for name in failures),
            evidence={
                "certificate": self._display_path(path),
                "asOf": cert.get("as_of"),
                "validUntil": cert.get("valid_until"),
                "queryOnly": cert.get("query_only_ready"),
            },
        )

    @staticmethod
    def _target_risk() -> ReadinessDimension:
        return ReadinessDimension(
            key="targetRisk",
            label="Target Risk",
            state="WIRED",
            severity="info",
            reasons=("wiring_presence_is_not_a_runtime_pass",),
            evidence={"semantics": TARGET_RISK_AUTHORIZATION_SEMANTICS},
        )

    @staticmethod
    def _order_risk() -> ReadinessDimension:
        return ReadinessDimension(
            key="orderRisk",
            label="Order Risk",
            state="WIRED",
            severity="info",
            reasons=("wiring_presence_is_not_a_runtime_pass",),
            evidence={"semantics": ORDER_RISK_AUTHORIZATION_SEMANTICS},
        )

    def _kill_switch(self) -> ReadinessDimension:
        path, payload, error = self._read_runtime_json(self.KILL_SWITCH_STATE)
        if payload is None:
            return self._missing_or_invalid(
                key="killSwitch",
                label="KillSwitch",
                path=path,
                error=error,
                missing_state="NOT_CONFIGURED",
            )

        # Read the canonical durable KillSwitch schema. The persisted state does
        # not contain a derived ``triggered`` field; reconstruct it from the two
        # authoritative causes rather than requiring an invented duplicate value.
        version = payload.get("version")
        manual = payload.get("manual_triggered")
        raw_reasons = payload.get("reasons")
        if (
            version != self.KILL_SWITCH_SCHEMA_VERSION
            or not isinstance(manual, bool)
            or not isinstance(raw_reasons, list)
        ):
            return ReadinessDimension(
                key="killSwitch",
                label="KillSwitch",
                state="INVALID",
                severity="blocked",
                reasons=("kill_switch_state_schema_invalid",),
                evidence={"stateFile": self._display_path(path)},
            )
        reasons = tuple(str(item).strip() for item in raw_reasons if str(item).strip())
        triggered = bool(manual or reasons)
        return ReadinessDimension(
            key="killSwitch",
            label="KillSwitch",
            state="KILLED" if triggered else "CLEAR",
            severity="blocked" if triggered else "ok",
            reasons=reasons if triggered else (),
            evidence={
                "stateFile": self._display_path(path),
                "stateVersion": version,
                "manualTriggered": manual,
            },
        )

    def _reconciliation(self) -> ReadinessDimension:
        path, cert, error = self._read_runtime_json(self.RECONCILIATION_CERT)
        if cert is None:
            return self._missing_or_invalid(
                key="reconciliation",
                label="Reconciliation",
                path=path,
                error=error,
                missing_state="NOT_CERTIFIED",
            )
        temporal = self._certificate_time_state(cert)
        if temporal is not None:
            return self._temporal_card(
                key="reconciliation",
                label="Reconciliation",
                path=path,
                cert=cert,
                temporal=temporal,
            )
        fields = (
            "unexplained_order_differences",
            "unexplained_trade_differences",
            "unexplained_position_differences",
            "unexplained_cash_differences",
        )
        invalid = [name for name in fields if not self._nonnegative_int(cert.get(name))]
        if invalid:
            return ReadinessDimension(
                key="reconciliation",
                label="Reconciliation",
                state="INVALID",
                severity="blocked",
                reasons=tuple(f"invalid_{name}" for name in invalid),
                evidence={"certificate": self._display_path(path)},
            )
        mismatches = {name: int(cert[name]) for name in fields}
        clean = all(value == 0 for value in mismatches.values()) and cert.get("complete") is True
        reasons: list[str] = []
        if cert.get("complete") is not True:
            reasons.append("reconciliation_not_complete")
        reasons.extend(f"{name}:{value}" for name, value in mismatches.items() if value)
        return ReadinessDimension(
            key="reconciliation",
            label="Reconciliation",
            state="CLEAN" if clean else "BLOCKED",
            severity="ok" if clean else "blocked",
            reasons=tuple(reasons),
            evidence={
                "certificate": self._display_path(path),
                "asOf": cert.get("as_of"),
                "validUntil": cert.get("valid_until"),
                "unexplainedDifferences": mismatches,
            },
        )

    @staticmethod
    def _product_arming() -> ReadinessDimension:
        policy = describe_policy()
        armed = policy.get("liveTradingAvailable") is True
        return ReadinessDimension(
            key="productArming",
            label="Product Arming",
            state="ARMED" if armed else "NOT_ARMED",
            severity="ok" if armed else "blocked",
            reasons=() if armed else ("product_policy_live_disabled",),
            evidence={
                "liveTradingAvailable": bool(policy.get("liveTradingAvailable")),
                "writeApis": policy.get("writeApis"),
                "brokerOrExchangeIntegration": policy.get("brokerOrExchangeIntegration"),
            },
        )

    def _host_certification(self) -> ReadinessDimension:
        path, cert, error = self._read_runtime_json(self.HOST_CERT)
        extra = {"portableContractIsNotHostCertification": True}
        if cert is None:
            return self._missing_or_invalid(
                key="hostCertification",
                label="Host / Platform Certification",
                path=path,
                error=error,
                missing_state="NOT_CERTIFIED",
                extra=extra,
            )
        temporal = self._certificate_time_state(cert)
        if temporal is not None:
            return self._temporal_card(
                key="hostCertification",
                label="Host / Platform Certification",
                path=path,
                cert=cert,
                temporal=temporal,
                extra=extra,
            )
        platform = str(cert.get("platform") or "").strip().lower()
        broker = str(cert.get("broker") or "").strip().lower()
        controlled = cert.get("controlled_host") is True
        certified = cert.get("certified") is True
        valid_platform = platform.startswith("windows")
        valid_broker = broker in {"qmt", "miniqmt", "xtquant"}
        ok = bool(certified and controlled and valid_platform and valid_broker)
        reasons: list[str] = []
        if not certified:
            reasons.append("host_not_certified")
        if not controlled:
            reasons.append("controlled_host_not_proven")
        if not valid_platform:
            reasons.append("windows_host_not_proven")
        if not valid_broker:
            reasons.append("miniqmt_qmt_stack_not_proven")
        return ReadinessDimension(
            key="hostCertification",
            label="Host / Platform Certification",
            state="CERTIFIED" if ok else "BLOCKED",
            severity="ok" if ok else "blocked",
            reasons=tuple(reasons),
            evidence={
                "certificate": self._display_path(path),
                "platform": cert.get("platform"),
                "broker": cert.get("broker"),
                "validUntil": cert.get("valid_until"),
                **extra,
            },
        )

    def _read_runtime_json(
        self,
        relative_path: Path,
    ) -> tuple[Path, dict[str, Any] | None, str | None]:
        """Read a fixed runtime artifact without allowing symlink/path substitution."""
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return self.runtime_root, None, "invalid_relative_path"
        candidate = self.runtime_root / relative_path

        # Reject a symlink at any existing path component. Otherwise a fixed
        # certificate name can silently be redirected after review.
        current = self.runtime_root
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                return candidate, None, "symlink_not_allowed"

        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.runtime_root)
        except (OSError, ValueError):
            return candidate, None, "path_outside_runtime"
        if not candidate.exists() or not candidate.is_file():
            return candidate, None, "not_found"
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - corruption is operator evidence
            return candidate, None, type(exc).__name__
        if not isinstance(payload, dict):
            return candidate, None, "not_object"
        return candidate, payload, None

    def _certificate_time_state(self, cert: dict[str, Any]) -> tuple[str, str] | None:
        raw = cert.get("valid_until")
        if not raw:
            return ("INVALID", "certificate_valid_until_missing")
        valid_until = self._parse_timestamp(raw)
        if valid_until is None:
            return ("INVALID", "certificate_valid_until_invalid")
        if valid_until <= self._now():
            return ("EXPIRED", "certificate_expired")
        return None

    def _temporal_card(
        self,
        *,
        key: str,
        label: str,
        path: Path,
        cert: dict[str, Any],
        temporal: tuple[str, str],
        extra: dict[str, Any] | None = None,
    ) -> ReadinessDimension:
        state, reason = temporal
        evidence = {
            "certificate": self._display_path(path),
            "asOf": cert.get("as_of"),
            "validUntil": cert.get("valid_until"),
        }
        if extra:
            evidence.update(extra)
        return ReadinessDimension(
            key=key,
            label=label,
            state=state,
            severity="blocked",
            reasons=(reason,),
            evidence=evidence,
        )

    def _missing_or_invalid(
        self,
        *,
        key: str,
        label: str,
        path: Path,
        error: str | None,
        missing_state: str,
        extra: dict[str, Any] | None = None,
    ) -> ReadinessDimension:
        missing = error == "not_found"
        evidence: dict[str, Any] = {"path": self._display_path(path)}
        if extra:
            evidence.update(extra)
        return ReadinessDimension(
            key=key,
            label=label,
            state=missing_state if missing else "INVALID",
            severity="unknown" if missing else "blocked",
            reasons=(f"runtime_evidence_{'missing' if missing else 'invalid'}:{error or 'unknown'}",),
            evidence=evidence,
        )

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        try:
            text = str(value).strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except Exception:
            return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _now(self) -> datetime:
        now = self._now_fn()
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        for root, prefix in ((self.runtime_root, "runtime"), (self.project_root, "repo")):
            try:
                return f"{prefix}/{resolved.relative_to(root)}"
            except ValueError:
                continue
        return path.name

    @staticmethod
    def _nonnegative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = ["ProductionReadinessService", "ReadinessDimension"]
