"""Portfolio limits and durable risk state on the `/api/paper/orders` venue.

`PaperOrderService` is built as `services/quant_api/services/container.py`
builds it, except that a deterministic market source and an industry map are
injected (production wires neither by default).

Before these fixes:
* `RiskEngine.check_portfolio` (drawdown 20%, daily loss 20,000, gross 1.0) had
  zero production call sites: a BUY filled with the book 24.41% under water;
* the kill switch, the drawdown reference and "daily" turnover lived in process
  memory: a restart cleared a latched switch and forgot today's turnover, while
  a long-lived process refused day 2 on day 1's turnover;
* `max_orders_per_symbol_per_day` was read only inside `reconcile()`, which no
  production path calls.
"""

from __future__ import annotations

import pytest

from quantagent.paper.broker import MarketSnapshot
from quantagent.paper.risk import SCOPE_GLOBAL, RiskLimits, RiskRejection
from services.quant_api.services.paper_orders import PaperOrderService

D1, D2, D3, D4 = "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06"
SYM = "600000.SH"
INDUSTRY_MAP = {SYM: "bank"}


class Market:
    """(symbol, date) -> snapshot."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], MarketSnapshot] = {}

    def set(self, symbol, date, last, prev, *, volume=1e8):
        self.rows[(symbol, date)] = MarketSnapshot(
            symbol=symbol, trade_date=date, last_price=last, previous_close=prev,
            session_volume=volume, board="SH_Main",
        )

    def __call__(self, symbol, date):
        return self.rows.get((symbol, date))


def service(root, market, limits=None) -> PaperOrderService:
    return PaperOrderService(root, market_source=market, initial_cash=1_000_000.0,
                             risk_limits=limits, industry_map=INDUSTRY_MAP)


class Orders:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, svc, *, side="BUY", qty=1000, px=10.0, date=D1):
        self.n += 1
        key = f"k{self.n}"
        svc.submit({
            "idempotencyKey": key, "runId": "risk", "symbol": SYM, "side": side,
            "quantity": qty, "limitPrice": px, "tradeDate": date, "signalId": key,
        })
        return svc.drain()[-1]


@pytest.fixture
def order():
    return Orders()


def test_a_buy_after_the_drawdown_and_daily_loss_limits_are_breached_is_refused(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    m.set(SYM, D4, 7.29, 8.1)  # three limit-down days later
    # Concentration relaxed so the single 90% name is allowed; drawdown and
    # daily loss stay at their defaults (20% / 20,000).
    limits = RiskLimits(max_participation=1.0, max_single_name_weight=1.0,
                        max_industry_weight=1.0, max_order_notional=1e9)
    svc = service(tmp_path / "p", m, limits)
    try:
        first = order(svc, qty=90_000, date=D1)
        after = order(svc, qty=1_000, px=7.29, date=D4)
        equity = svc.broker.portfolio.equity({SYM: 7.29})
        switches = svc.broker.kill_switch.active()
    finally:
        svc.close()

    assert first["venueStatus"] == "FILLED"
    assert 1 - equity / 1_000_000.0 > limits.max_drawdown
    assert after["venueStatus"] == "REJECTED"
    assert "kill switch" in after["reason"]
    assert [s["scope"] for s in switches] == ["PORTFOLIO"]


def test_a_risk_engine_kill_switch_survives_a_restart(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    root = tmp_path / "p"
    svc = service(root, m)
    svc.broker.risk_engine.kill_switch.trigger(SCOPE_GLOBAL, "operator halt")
    blocked = order(svc, qty=100)
    svc.close()

    restarted = service(root, m)
    try:
        after = order(restarted, qty=100)
    finally:
        restarted.close()

    assert blocked["venueStatus"] == "REJECTED"
    assert after["venueStatus"] == "REJECTED", "a restart silently cleared the kill switch"
    assert "operator halt" in after["reason"]


def test_a_broker_level_kill_switch_survives_a_restart(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    root = tmp_path / "p"
    svc = service(root, m)
    svc.broker.trigger_kill_switch("operator halt")
    svc.close()

    restarted = service(root, m)
    try:
        after = order(restarted, qty=100)
    finally:
        restarted.close()

    assert after["venueStatus"] == "REJECTED"


def test_clearing_requires_a_human_and_the_clear_is_durable_too(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    root = tmp_path / "p"
    svc = service(root, m)
    svc.broker.trigger_kill_switch("operator halt")
    with pytest.raises(RiskRejection, match="human confirmation"):
        svc.broker.clear_kill_switch(SCOPE_GLOBAL)
    assert svc.broker.clear_kill_switch(SCOPE_GLOBAL, human_confirmation=True)
    svc.close()

    restarted = service(root, m)
    try:
        after = order(restarted, qty=100)
    finally:
        restarted.close()

    assert after["venueStatus"] == "FILLED"


def test_daily_turnover_is_keyed_by_exchange_session_not_process_life(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    m.set(SYM, D2, 10.0, 10.0)
    limits = RiskLimits(max_participation=1.0, max_single_name_weight=1.0,
                        max_industry_weight=1.0, max_daily_turnover=0.5)
    svc = service(tmp_path / "p", m, limits)
    try:
        day1 = [order(svc, qty=15_000, date=D1) for _ in range(3)]  # 450k of a 1M book
        day2 = order(svc, qty=15_000, date=D2)                      # day 2 has used 0
    finally:
        svc.close()

    assert [o["venueStatus"] for o in day1] == ["FILLED"] * 3
    assert day2["venueStatus"] == "FILLED", "a new session was refused on yesterday's turnover"


def test_turnover_consumed_today_survives_a_restart(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    limits = RiskLimits(max_participation=1.0, max_single_name_weight=1.0,
                        max_industry_weight=1.0, max_daily_turnover=0.3)
    root = tmp_path / "p"
    svc = service(root, m, limits)
    first = order(svc, qty=15_000, date=D1)
    second = order(svc, qty=15_000, date=D1)
    svc.close()

    restarted = service(root, m, limits)
    try:
        after = order(restarted, qty=15_000, date=D1)
    finally:
        restarted.close()

    assert first["venueStatus"] == "FILLED"
    assert second["venueStatus"] == "REJECTED"   # 300k + fees > 30% of equity
    assert after["venueStatus"] == "REJECTED", "a restart forgot today's turnover"
    assert "daily_turnover" in after["reason"]


def test_peak_equity_survives_a_restart(tmp_path, order):
    m = Market()
    m.set(SYM, D1, 10.0, 10.0)
    m.set(SYM, D2, 11.0, 10.0)
    limits = RiskLimits(max_participation=1.0, max_single_name_weight=1.0,
                        max_industry_weight=1.0, max_order_notional=1e9)
    root = tmp_path / "p"
    svc = service(root, m, limits)
    order(svc, qty=50_000, date=D1)
    order(svc, qty=100, px=11.0, date=D2)  # the venue marks the book at 11.0
    peak = svc.broker.risk_engine.peak_equity
    svc.close()

    restarted = service(root, m, limits)
    try:
        restored = restarted.broker.risk_engine.peak_equity
    finally:
        restarted.close()

    assert peak > 1_040_000.0
    assert restored == pytest.approx(peak)
