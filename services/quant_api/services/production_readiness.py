"""Read-only production-readiness truth for the operator Governance surface.

This service deliberately does not connect to a broker, instantiate a fresh
KillSwitch, or infer runtime certification from code existence.  It only reads
machine evidence from fixed repository/runtime locations and verifies the
existing model-trust certificate.  Missing evidence remains non-green.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

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
            "aggregateTradingReady": None,
            "aggregateStateSemantics": "intentionally_not_computed_show_all_dimensions",
            "cards": [card.to_dict() for card in cards],
        }

    def _model_trust(self) -> ReadinessDimension:
        manifest = self.project_root / "configs/live_model_trust.json"
        report = evaluate_live_model_trust(manifest)
        evidence = {
            "modelId": report.model_id,
            "certificateStatus": report.status,
            "trustClass": report.trust_class,
            "requiredMetricSemantics": REQUIRED_METRIC_SEMANTICS,
            "observedMetricSemantics": report.evidence.get("strict_backtest_metric_semantics"),
            "manifest": self._display_path(manifest),
        }
        return ReadinessDimension(
            key="modelTrust",
            label="Model Trust",
            state="PASS" if report.ok else "BLOCKED",
            severity="ok" if report.ok else "blocked",
            reasons=tuple(report.reasons),
            evidence=evidence,
        )

    def _broker_query_readiness(self) -> ReadinessDimension:
        path = self.runtime_root / self.BROKER_QUERY_CERT
        cert, error = self._read_json(path)
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
        ready = not failures
        return ReadinessDimension(
            key="brokerQuery",
            label="Broker Query Readiness",
            state="READY" if ready else "BLOCKED",
            severity="ok" if ready else "blocked",
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
        path = self.runtime_root / self.KILL_SWITCH_STATE
        payload, error = self._read_json(path)
        if payload is None:
            return self._missing_or_invalid(
                key="killSwitch",
                label="KillSwitch",
                path=path,
                error=error,
                missing_state="NOT_CONFIGURED",
            )
        triggered = payload.get("triggered")
        reasons = payload.get("reasons")
        if not isinstance(triggered, bool) or not isinstance(reasons, list):
            return ReadinessDimension(
                key="killSwitch",
                label="KillSwitch",
                state="INVALID",
                severity="blocked",
                reasons=("kill_switch_state_schema_invalid",),
                evidence={"stateFile": self._display_path(path)},
            )
        clean_reasons = tuple(str(item) for item in reasons if str(item).strip())
        return ReadinessDimension(
            key="killSwitch",
            label="KillSwitch",
            state="KILLED" if triggered else "CLEAR",
            severity="blocked" if triggered else "ok",
            reasons=clean_reasons if triggered else (),
            evidence={
                "stateFile": self._display_path(path),
                "updatedAt": payload.get("updated_at"),
            },
        )

    def _reconciliation(self) -> ReadinessDimension:
        path = self.runtime_root / self.RECONCILIATION_CERT
        cert, error = self._read_json(path)
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
        invalid_fields = [name for name in fields if not self._nonnegative_int(cert.get(name))]
        if invalid_fields:
            return ReadinessDimension(
                key="reconciliation",
                label="Reconciliation",
                state="INVALID",
                severity="blocked",
                reasons=tuple(f"invalid_{name}" for name in invalid_fields),
                evidence={"certificate": self._display_path(path)},
            )
        mismatches = {name: int(cert[name]) for name in fields}
        clean = all(value == 0 for value in mismatches.values()) and cert.get("complete") is True
        reasons: list[str] = []
        if cert.get("complete") is not True:
            reasons.append("reconciliation_not_complete")
        reasons.extend(f"{name}:{value}" for name, value in mismatches.items() if value != 0)
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
        path = self.runtime_root / self.HOST_CERT
        cert, error = self._read_json(path)
        if cert is None:
            return self._missing_or_invalid(
                key="hostCertification",
                label="Host / Platform Certification",
                path=path,
                error=error,
                missing_state="NOT_CERTIFIED",
                extra={"portableContractIsNotHostCertification": True},
            )
        temporal = self._certificate_time_state(cert)
        if temporal is not None:
            return self._temporal_card(
                key="hostCertification",
                label="Host / Platform Certification",
                path=path,
                cert=cert,
                temporal=temporal,
                extra={"portableContractIsNotHostCertification": True},
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
                "portableContractIsNotHostCertification": True,
            },
        )

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
            reasons=(
                f"runtime_evidence_{'missing' if missing else 'invalid'}:{error or 'unknown'}",
            ),
            evidence=evidence,
        )

    @staticmethod
    def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
        if not path.exists() or not path.is_file():
            return None, "not_found"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — corruption is operator evidence
            return None, f"{type(exc).__name__}"
        if not isinstance(payload, dict):
            return None, "not_object"
        return payload, None

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
        resolved = path.resolve()
        for root, prefix in ((self.project_root, "repo"), (self.runtime_root, "runtime")):
            try:
                return f"{prefix}/{resolved.relative_to(root)}"
            except ValueError:
                continue
        return resolved.name

    @staticmethod
    def _nonnegative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = ["ProductionReadinessService", "ReadinessDimension"]
