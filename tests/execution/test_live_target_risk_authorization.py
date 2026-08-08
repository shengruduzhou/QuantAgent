from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.execution.live_session import LiveTradingSession
from quantagent.risk.portfolio_risk import build_portfolio_risk_snapshot
from quantagent.risk.risk_gate import RiskGate
from quantagent.risk.risk_limits import V6RiskLimits


class RecordingGateway:
    def __init__(self) -> None:
        self.submit_calls = 0

    def preflight(self):
        return {"ok": True, "connected": True}

    def health(self):
        return {"ok": True, "connected": True}

    def submit(self, *_args, **_kwargs):
        self.submit_calls += 1
        raise AssertionError("LiveTradingSession must not submit")


def _session(gateway: RecordingGateway, limits: V6RiskLimits | None = None) -> LiveTradingSession:
    return LiveTradingSession(
        gateway,
        risk_gate=RiskGate(limits=limits),
        model_trust_manifest=Path("configs/live_model_trust.json"),
    )


def _intent(target_weight: float = 0.40) -> OrderIntent:
    return OrderIntent(
        intent_id="intent-target-1",
        symbol="A",
        side=OrderSide.BUY,
        quantity=100,
        target_weight=target_weight,
        reference_price=10.0,
        signal_id="signal-target-1",
        model_version="blocked-current-model",
        feature_version="f1",
        strategy_version="s1",
        risk_check_result="not_checked",
        timestamp="2026-08-08T10:00:00+08:00",
    )


def _market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "is_suspended": [False, False],
            "is_st": [False, False],
            "is_limit_up": [False, False],
            "is_limit_down": [False, False],
        }
    )


def test_order_authorization_requires_prior_target_level_gate() -> None:
    gateway = RecordingGateway()
    authorization = _session(gateway).authorize_order(_intent())
    assert authorization.allowed is False
    assert authorization.target_authorization_present is False
    assert "target_risk_authorization_missing" in authorization.reasons
    assert gateway.submit_calls == 0


def test_rejected_target_produces_no_submit_and_cannot_authorize_order() -> None:
    gateway = RecordingGateway()
    limits = V6RiskLimits(
        max_name_weight=1.0,
        max_sector_weight=1.0,
        max_turnover=2.0,
        max_leverage=1.0,
        beta_exposure_limit=1.2,
    )
    session = _session(gateway, limits)
    weights = pd.Series({"A": 0.60, "B": 0.60})
    snapshot = build_portfolio_risk_snapshot(
        weights,
        beta=pd.Series({"A": 1.0, "B": 1.0}),
        sector=pd.Series({"A": "bank", "B": "tech"}),
        beta_pit_safe=True,
        sector_pit_safe=True,
        beta_freshness_days=0.0,
        sector_freshness_days=0.0,
        as_of="2026-08-08",
    )
    target_auth = session.authorize_targets(
        weights,
        risk_snapshot=snapshot,
        conformal_width=pd.Series({"A": 0.01, "B": 0.01}),
        market_state=_market(),
    )
    assert target_auth.risk_result.passed is False
    assert "max_leverage" in target_auth.risk_result.violations
    order_auth = session.authorize_order(_intent(target_weight=0.60), target_authorization=target_auth)
    assert order_auth.allowed is False
    assert "target_risk_authorization_blocked" in order_auth.reasons
    assert gateway.submit_calls == 0


def test_target_snapshot_must_match_the_authorized_weight_vector() -> None:
    gateway = RecordingGateway()
    limits = V6RiskLimits(
        max_name_weight=1.0,
        max_sector_weight=1.0,
        max_turnover=2.0,
        max_leverage=1.0,
        beta_exposure_limit=1.2,
    )
    session = _session(gateway, limits)
    original = pd.Series({"A": 0.40, "B": 0.40})
    changed = pd.Series({"A": 0.30, "B": 0.50})
    snapshot = build_portfolio_risk_snapshot(
        original,
        beta=pd.Series({"A": 1.0, "B": 1.0}),
        sector=pd.Series({"A": "bank", "B": "tech"}),
        beta_pit_safe=True,
        sector_pit_safe=True,
        beta_freshness_days=0.0,
        sector_freshness_days=0.0,
    )
    target_auth = session.authorize_targets(
        changed,
        risk_snapshot=snapshot,
        conformal_width=pd.Series({"A": 0.01, "B": 0.01}),
        market_state=_market(),
    )
    assert "portfolio_risk_snapshot_target_mismatch" in target_auth.risk_result.violations
    assert gateway.submit_calls == 0
