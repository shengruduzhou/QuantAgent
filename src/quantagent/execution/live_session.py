"""Canonical research-to-broker readiness gate.

A connected broker adapter is not permission to trade.  ``LiveTradingSession``
AND-composes the independent evidence domains that must agree before an economic
live route could ever be armed: model trust, persistent operational risk state,
broker preflight/health, and the global product policy.

The current product policy remains LIVE_DISABLED, so ``economic_submit_allowed``
is intentionally false on today's mainline even if a controlled QMT host is
query-ready.  This class exists to create one explicit arming boundary instead
of letting future callers compose those checks ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.execution.broker_base import OrderIntent
from quantagent.execution.live_model_trust import LiveModelTrustReport, evaluate_live_model_trust
from quantagent.risk.risk_gate import RiskGate, RiskGateResult
from quantagent.safety.operating_mode import describe_policy


@dataclass(frozen=True)
class LiveSessionReadiness:
    query_only_ready: bool
    economic_submit_allowed: bool
    model_trust_ok: bool
    kill_switch_clear: bool
    broker_preflight_ok: bool
    broker_health_ok: bool
    product_policy_armed: bool
    reasons: tuple[str, ...]
    model_report: dict[str, Any]
    broker_preflight: dict[str, Any]
    broker_health: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveOrderAuthorization:
    allowed: bool
    reasons: tuple[str, ...]
    risk_result: RiskGateResult
    readiness: LiveSessionReadiness


class LiveTradingSession:
    """Single fail-closed arming boundary for controlled broker integration."""

    def __init__(
        self,
        gateway: object,
        *,
        risk_gate: RiskGate,
        model_trust_manifest: str | Path,
    ) -> None:
        self.gateway = gateway
        self.risk_gate = risk_gate
        self.model_trust_manifest = Path(model_trust_manifest)

    def readiness(self) -> LiveSessionReadiness:
        reasons: list[str] = []
        model: LiveModelTrustReport = evaluate_live_model_trust(self.model_trust_manifest)
        if not model.ok:
            reasons.extend(f"model:{reason}" for reason in model.reasons)

        kill_clear = not self.risk_gate.kill_switch.triggered
        if not kill_clear:
            reasons.extend(f"kill_switch:{reason}" for reason in self.risk_gate.kill_switch.reasons)

        preflight: dict[str, Any]
        health: dict[str, Any]
        try:
            raw = self.gateway.preflight()  # type: ignore[attr-defined]
            preflight = dict(raw) if isinstance(raw, dict) else {"ok": False, "raw": raw}
        except Exception as exc:
            preflight = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
        preflight_ok = preflight.get("ok") is True
        if not preflight_ok:
            reasons.append("broker_preflight_not_ok")

        try:
            raw_health = self.gateway.health()  # type: ignore[attr-defined]
            health = dict(raw_health) if isinstance(raw_health, dict) else {"ok": False, "raw": raw_health}
        except Exception as exc:
            health = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
        health_ok = health.get("ok") is True
        if not health_ok:
            reasons.append("broker_health_not_ok")

        policy = describe_policy()
        policy_armed = policy.get("liveTradingAvailable") is True
        if not policy_armed:
            reasons.append("product_policy_live_disabled")

        # Query-only certification deliberately does NOT require model alpha to
        # be promoted or product live policy to be armed.  It proves only that
        # the controlled broker/account state can be read and reconciled while
        # operational risk is not already killed.
        query_only_ready = bool(kill_clear and preflight_ok and health_ok)
        economic_allowed = bool(
            query_only_ready
            and model.ok
            and policy_armed
        )
        return LiveSessionReadiness(
            query_only_ready=query_only_ready,
            economic_submit_allowed=economic_allowed,
            model_trust_ok=model.ok,
            kill_switch_clear=kill_clear,
            broker_preflight_ok=preflight_ok,
            broker_health_ok=health_ok,
            product_policy_armed=policy_armed,
            reasons=tuple(dict.fromkeys(reasons)),
            model_report=model.to_dict(),
            broker_preflight=preflight,
            broker_health=health,
        )

    def authorize_order(
        self,
        intent: OrderIntent,
        *,
        market_state: pd.DataFrame | None = None,
        cash_available: float = float("inf"),
    ) -> LiveOrderAuthorization:
        readiness = self.readiness()
        risk = self.risk_gate.check_order_intents(
            [intent],
            market_state=market_state,
            cash_available=cash_available,
        )
        reasons = list(readiness.reasons)
        if not risk.passed:
            reasons.extend(risk.violations)
            reasons.extend(f"order:{key}:{value}" for key, value in risk.rejected_symbols.items())
        allowed = bool(readiness.economic_submit_allowed and risk.passed)
        if not allowed and not reasons:
            reasons.append("live_session_not_authorized")
        return LiveOrderAuthorization(
            allowed=allowed,
            reasons=tuple(dict.fromkeys(reasons)),
            risk_result=risk,
            readiness=readiness,
        )


__all__ = [
    "LiveOrderAuthorization",
    "LiveSessionReadiness",
    "LiveTradingSession",
]
