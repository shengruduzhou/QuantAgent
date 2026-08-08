from __future__ import annotations

from pathlib import Path

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.execution.live_session import LiveTradingSession
from quantagent.risk.kill_switch import KillSwitch
from quantagent.risk.risk_gate import RiskGate


class FakeHealthyGateway:
    def preflight(self):
        return {"ok": True, "connected": True}

    def health(self):
        return {"ok": True, "connected": True}


class FakeBrokenGateway:
    def preflight(self):
        return {"ok": False, "errors": ["state_sync_failed"]}

    def health(self):
        return {"ok": False}


def _manifest() -> Path:
    return Path("configs/live_model_trust.json")


def _intent(quantity: int = 100) -> OrderIntent:
    return OrderIntent(
        intent_id="intent-1",
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=quantity,
        target_weight=0.01,
        reference_price=10.0,
        signal_id="signal-1",
        model_version="blocked-current-model",
        feature_version="f1",
        strategy_version="s1",
        risk_check_result="not_checked",
        timestamp="2026-08-08T10:00:00+08:00",
    )


def test_healthy_broker_can_be_query_only_ready_without_being_live_armed() -> None:
    session = LiveTradingSession(
        FakeHealthyGateway(),
        risk_gate=RiskGate(kill_switch=KillSwitch()),
        model_trust_manifest=_manifest(),
    )
    report = session.readiness()
    assert report.query_only_ready is True
    assert report.economic_submit_allowed is False
    assert report.model_trust_ok is False
    assert report.product_policy_armed is False
    assert "product_policy_live_disabled" in report.reasons


def test_kill_switch_blocks_even_query_only_certification(tmp_path) -> None:
    switch = KillSwitch(state_path=tmp_path / "kill.json")
    switch.trigger("severe_reconciliation_mismatch")
    session = LiveTradingSession(
        FakeHealthyGateway(),
        risk_gate=RiskGate(kill_switch=switch),
        model_trust_manifest=_manifest(),
    )
    report = session.readiness()
    assert report.query_only_ready is False
    assert any(reason.startswith("kill_switch:") for reason in report.reasons)


def test_broker_preflight_and_health_are_both_required() -> None:
    session = LiveTradingSession(
        FakeBrokenGateway(),
        risk_gate=RiskGate(kill_switch=KillSwitch()),
        model_trust_manifest=_manifest(),
    )
    report = session.readiness()
    assert report.query_only_ready is False
    assert "broker_preflight_not_ok" in report.reasons
    assert "broker_health_not_ok" in report.reasons


def test_order_authorization_cannot_bypass_model_policy_or_risk() -> None:
    session = LiveTradingSession(
        FakeHealthyGateway(),
        risk_gate=RiskGate(kill_switch=KillSwitch()),
        model_trust_manifest=_manifest(),
    )
    authorization = session.authorize_order(_intent(quantity=123))
    assert authorization.allowed is False
    assert authorization.risk_result.passed is False
    assert any("min_lot_size" in reason for reason in authorization.reasons)
