from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.execution.live_session import LiveTradingSession
from quantagent.risk.portfolio_risk import build_portfolio_risk_snapshot
from quantagent.risk.risk_gate import RiskGate
from quantagent.risk.risk_limits import V6RiskLimits


class Gateway:
    def preflight(self):
        return {"ok": True}

    def health(self):
        return {"ok": True}


def _session() -> LiveTradingSession:
    limits = V6RiskLimits(
        max_name_weight=1.0,
        max_sector_weight=1.0,
        max_turnover=2.0,
        max_leverage=1.0,
    )
    return LiveTradingSession(
        Gateway(),
        risk_gate=RiskGate(limits=limits),
        model_trust_manifest=Path("configs/live_model_trust.json"),
    )


def test_target_authorization_cannot_cross_live_session_boundary() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    snapshot = build_portfolio_risk_snapshot(
        weights,
        beta=pd.Series({"A": 1.0, "B": 1.0}),
        sector=pd.Series({"A": "bank", "B": "tech"}),
        beta_pit_safe=True,
        sector_pit_safe=True,
        beta_freshness_days=0.0,
        sector_freshness_days=0.0,
    )
    market = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "is_suspended": [False, False],
            "is_st": [False, False],
            "is_limit_up": [False, False],
            "is_limit_down": [False, False],
        }
    )
    source = _session()
    other = _session()
    target_authorization = source.authorize_targets(
        weights,
        risk_snapshot=snapshot,
        conformal_width=pd.Series({"A": 0.01, "B": 0.01}),
        market_state=market,
    )
    intent = OrderIntent(
        intent_id="intent-session-boundary",
        symbol="A",
        side=OrderSide.BUY,
        quantity=100,
        target_weight=0.4,
        reference_price=10.0,
        signal_id="signal-session-boundary",
        model_version="blocked-current-model",
        feature_version="f1",
        strategy_version="s1",
        risk_check_result="not_checked",
        timestamp="2026-08-08T10:00:00+08:00",
    )
    result = other.authorize_order(intent, target_authorization=target_authorization)
    assert result.allowed is False
    assert "target_risk_session_mismatch" in result.reasons
