"""Canonical research-to-broker readiness and risk-authorisation boundary.

A connected broker adapter is not permission to trade. ``LiveTradingSession``
AND-composes independent evidence domains and, critically, requires portfolio-
level target authorisation before order-level authorisation. The class still
contains no broker-submit call. Current product policy remains LIVE_DISABLED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import secrets
from typing import Any

import pandas as pd

from quantagent.execution.broker_base import OrderIntent
from quantagent.execution.live_model_trust import LiveModelTrustReport, evaluate_live_model_trust
from quantagent.risk.portfolio_risk import PortfolioRiskSnapshot
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
class LiveTargetAuthorization:
    allowed: bool
    reasons: tuple[str, ...]
    risk_result: RiskGateResult
    readiness: LiveSessionReadiness
    target_fingerprint: str | None
    session_token: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class LiveOrderAuthorization:
    allowed: bool
    reasons: tuple[str, ...]
    risk_result: RiskGateResult
    readiness: LiveSessionReadiness
    target_authorization_present: bool


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
        # Capability-style nonce: target authorisations are valid only inside the
        # exact session instance that issued them. This prevents accidental or
        # cross-session construction from bypassing the target-level gate.
        self._target_authorization_token = secrets.token_hex(32)

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
        # be promoted or product live policy to be armed. It proves only that
        # the controlled broker/account state can be read and reconciled while
        # operational risk is not already killed.
        query_only_ready = bool(kill_clear and preflight_ok and health_ok)
        economic_allowed = bool(query_only_ready and model.ok and policy_armed)
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

    def authorize_targets(
        self,
        target_weights: pd.Series,
        *,
        current_weights: pd.Series | None = None,
        market_state: pd.DataFrame | None = None,
        sector: pd.Series | None = None,
        data_quality_score: float = 1.0,
        model_drift_score: float = 0.0,
        conformal_width: pd.Series | None = None,
        risk_snapshot: PortfolioRiskSnapshot | None = None,
    ) -> LiveTargetAuthorization:
        """Production-profile portfolio gate that must precede order creation."""
        readiness = self.readiness()
        risk = self.risk_gate.check_target_weights(
            target_weights,
            current_weights=current_weights,
            market_state=market_state,
            sector=sector,
            data_quality_score=data_quality_score,
            model_drift_score=model_drift_score,
            conformal_width=conformal_width,
            risk_snapshot=risk_snapshot,
            production_mode=True,
        )
        reasons = list(readiness.reasons)
        if not risk.passed:
            reasons.extend(f"target:{reason}" for reason in risk.violations)
            reasons.extend(f"target:unknown:{reason}" for reason in risk.unknowns)
            reasons.extend(f"target:{key}:{value}" for key, value in risk.rejected_symbols.items())
        allowed = bool(readiness.economic_submit_allowed and risk.passed)
        if not allowed and not reasons:
            reasons.append("target_authorization_not_allowed")
        return LiveTargetAuthorization(
            allowed=allowed,
            reasons=tuple(dict.fromkeys(reasons)),
            risk_result=risk,
            readiness=readiness,
            target_fingerprint=None if risk_snapshot is None else risk_snapshot.target_fingerprint,
            session_token=self._target_authorization_token,
        )

    def authorize_order(
        self,
        intent: OrderIntent,
        *,
        target_authorization: LiveTargetAuthorization | None = None,
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

        target_ok = False
        if target_authorization is None:
            reasons.append("target_risk_authorization_missing")
        elif not secrets.compare_digest(
            target_authorization.session_token,
            self._target_authorization_token,
        ):
            reasons.append("target_risk_session_mismatch")
        elif not target_authorization.allowed:
            reasons.append("target_risk_authorization_blocked")
        else:
            checked = target_authorization.risk_result.checked_weights
            if checked is None:
                reasons.append("target_risk_checked_weights_missing")
            elif intent.symbol not in checked.index:
                reasons.append("target_risk_symbol_missing")
            elif abs(float(checked.loc[intent.symbol]) - float(intent.target_weight)) > 1e-10:
                reasons.append("target_risk_target_mismatch")
            else:
                target_ok = True

        if not risk.passed:
            reasons.extend(risk.violations)
            reasons.extend(f"order:{key}:{value}" for key, value in risk.rejected_symbols.items())
        allowed = bool(readiness.economic_submit_allowed and target_ok and risk.passed)
        if not allowed and not reasons:
            reasons.append("live_session_not_authorized")
        return LiveOrderAuthorization(
            allowed=allowed,
            reasons=tuple(dict.fromkeys(reasons)),
            risk_result=risk,
            readiness=readiness,
            target_authorization_present=target_authorization is not None,
        )


__all__ = [
    "LiveOrderAuthorization",
    "LiveSessionReadiness",
    "LiveTargetAuthorization",
    "LiveTradingSession",
]
