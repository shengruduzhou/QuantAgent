"""`/api/paper/account` states which account it is and what the risk engine sees.

The route used to expose cash, positions and NAV only: no account identity and
no risk state, so a latched kill switch, the drawdown from peak or a missing
industry map were invisible to the operator. Unmeasured values are null with a
reason, never 0 — a 0 would read as "no drawdown" or "no turnover".
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.broker import MarketSnapshot
from quantagent.paper.risk import SCOPE_PORTFOLIO
from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings
from services.quant_api.services.paper_orders import PaperOrderService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYM = "600000.SH"
DAY = "2026-08-04"


def market(symbol: str, trade_date: str) -> MarketSnapshot | None:
    if symbol != SYM:
        return None
    return MarketSnapshot(symbol=symbol, trade_date=trade_date, last_price=10.0,
                          previous_close=10.0, session_volume=1e8, board="SH_Main")


def buy(svc: PaperOrderService, key: str = "k1") -> dict:
    svc.submit({
        "idempotencyKey": key, "runId": "acct", "symbol": SYM, "side": "BUY",
        "quantity": 1000, "limitPrice": 10.0, "tradeDate": DAY, "signalId": key,
    })
    return svc.drain()[-1]


@pytest.fixture
def client(tmp_path):
    settings = ApiSettings(
        project_root=PROJECT_ROOT,
        runtime_root=tmp_path / "runtime",
        cache_root=tmp_path / "runtime" / "cache",
        jobs_root=tmp_path / "runtime" / "jobs",
    ).ensure()
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.services.paper_orders.close()


def test_a_fresh_account_reports_identity_and_risk_state_through_the_route(client):
    data = client.get("/api/paper/account").json()["data"]

    identity = data["accountIdentity"]
    assert identity["portfolioId"] == "paper_api"
    assert identity["accountInstanceId"] is None
    assert identity["identitySha256"] is None
    assert "no_identity_record" in identity["reasons"]["accountInstanceId"]

    risk = data["riskState"]
    assert risk["riskEngineAttached"] is True
    assert risk["limits"]["max_drawdown"] == 0.20
    assert risk["killSwitch"]["active"] is False
    assert risk["killSwitch"]["scope"] is None
    assert risk["reasons"]["killSwitch"] == "no_active_kill_switch"
    assert risk["peakEquity"] == 1_000_000.0          # inception capital
    assert risk["drawdownFromPeak"] == 0.0             # measured: all cash
    assert risk["sessionTurnover"] is None
    assert "no_session_observed" in risk["reasons"]["sessionTurnover"]
    assert risk["unpriceableSymbols"] == []
    # No sector map in this deployment: the limit is enforced by refusal.
    assert risk["industryLimitEnforced"] is True
    assert risk["industryLimit"]["mode"] == "refuse_unmeasured"
    assert "industry_unmeasured" in risk["reasons"]["industryLimit"]


def test_risk_state_after_trading_and_a_latched_switch(tmp_path):
    svc = PaperOrderService(tmp_path / "p", market_source=market,
                            industry_map={SYM: "bank"})
    try:
        assert buy(svc)["venueStatus"] == "FILLED"
        svc.broker.risk_engine.kill_switch.trigger(SCOPE_PORTFOLIO, "drawdown breach")
        risk = svc.account()["riskState"]
    finally:
        svc.close()

    assert risk["killSwitch"]["active"] is True
    assert risk["killSwitch"]["scope"] == "PORTFOLIO"
    assert risk["killSwitch"]["reduceOnly"] is True
    assert risk["killSwitch"]["reason"] == "drawdown breach"
    assert risk["killSwitch"]["triggeredAt"]
    assert risk["sessionTurnover"]["session"] == DAY
    assert risk["sessionTurnover"]["notional"] == pytest.approx(10_000.0)
    assert risk["industryLimit"]["mode"] == "measured_with_map"
    assert risk["lastPortfolioCheck"]["verdict"] in {"APPROVED", "REJECTED"}
    assert risk["drawdownFromPeak"] is not None


def test_an_unmarked_holding_after_restart_is_null_with_a_reason_not_zero(tmp_path):
    root = tmp_path / "p"
    svc = PaperOrderService(root, market_source=market, industry_map={SYM: "bank"})
    buy(svc)
    svc.close()

    restarted = PaperOrderService(root, market_source=market, industry_map={SYM: "bank"})
    try:
        risk = restarted.account()["riskState"]
    finally:
        restarted.close()

    assert risk["unpriceableSymbols"] == [SYM]
    assert risk["nav"] is None
    assert risk["drawdownFromPeak"] is None
    assert risk["reasons"]["drawdownFromPeak"] == "nav_unavailable"
    assert risk["sessionTurnover"] is None


def test_an_identity_record_next_to_the_ledger_is_reported(tmp_path):
    root = tmp_path / "p"
    svc = PaperOrderService(root, market_source=market)
    try:
        identity = ensure_paper_account_identity(
            canonical_ledger_path=svc.ledger_path, portfolio_id="paper_api",
            initial_cash=1_000_000.0,
        )
        reported = svc.account()["accountIdentity"]
    finally:
        svc.close()

    assert reported["accountInstanceId"] == identity.account_instance_id
    assert reported["identitySha256"] == identity.payload_sha256
    assert reported["reasons"] == {}
