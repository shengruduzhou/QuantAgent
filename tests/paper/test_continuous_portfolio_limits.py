"""Portfolio limits on the continuous paper loop: evaluated, latched, reduce-only.

The continuous loop built a fresh `RiskEngine` per session and nothing ever
called `check_portfolio`, so the drawdown and daily-loss limits never latched
and the peak equity was forgotten between sessions. The session-start check
now values the recovered book at the session's prices against the persisted
peak and the previous close; a breach latches the PORTFOLIO switch, which
refuses BUYs and still lets the book be sold down.
"""

from __future__ import annotations

import pandas as pd

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper import ledger as lg
from quantagent.paper.continuous_execution import execute_pending_for_session
from quantagent.paper.risk import RiskLimits
from tests.paper.test_continuous_multi_name_book import (
    BANKS,
    FRIDAY,
    MONDAY,
    OTHER,
    SESSIONS,
    TUESDAY,
    config,
    record_target,
    sector_map,
)

HELD, NEW = BANKS[0], OTHER


def market_with_drop(drop_to: float) -> pd.DataFrame:
    rows = []
    for date in SESSIONS:
        for symbol in (HELD, NEW):
            close = drop_to if (symbol == HELD and date == TUESDAY) else 10.0
            rows.append({
                "trade_date": date, "symbol": symbol, "open": close, "high": close,
                "low": close, "close": close, "volume": 10_000_000.0,
                "amount": 100_000_000.0, "is_suspended": False, "is_st": False,
                "price_adjustment": "raw", "execution_eligible": True,
            })
    return pd.DataFrame(rows)


def limits() -> RiskLimits:
    # A concentrated single name is the cheapest way to move the whole book;
    # single-name and order-size limits are relaxed so only the portfolio
    # limits under test can object.
    return RiskLimits(max_single_name_weight=1.0, max_order_notional=1e9,
                      max_participation=1.0, max_industry_weight=1.0,
                      max_drawdown=0.03, max_daily_loss=1e12)


def run(tmp_path, date, frame):
    return execute_pending_for_session(
        date, frame,
        config=config(tmp_path, risk_limits=limits(), sector_map_path=sector_map(tmp_path)),
        authoritative_sessions=SESSIONS,
    )


def test_a_drawdown_breach_at_session_start_is_reduce_only(tmp_path) -> None:
    frame = market_with_drop(9.2)  # -8% on a 50% name: about -4% on the book
    record_target(tmp_path, FRIDAY, {HELD: 0.5})
    assert run(tmp_path, MONDAY, frame)[0].fill_count == 1

    record_target(tmp_path, MONDAY, {HELD: 0.25, NEW: 0.10})
    result = run(tmp_path, TUESDAY, frame)[0]

    book = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book()
    tuesday = {o.symbol: o for o in book.orders() if o.trade_date == TUESDAY}
    assert tuesday[HELD].side.value == "SELL" and tuesday[HELD].status.value == "FILLED"
    assert tuesday[NEW].status.value == "REJECTED"
    assert "kill switch" in (tuesday[NEW].reason or "")
    assert result.fill_count == 1

    triggered = [
        e.payload for e in lg.EventLedger(tmp_path / "operational.jsonl").read()
        if e.event_type == lg.KILL_SWITCH_TRIGGERED
    ]
    assert [p["scope"] for p in triggered] == ["PORTFOLIO"]
    assert "drawdown" in triggered[0]["reason"]


def test_the_peak_and_the_latched_switch_are_rebuilt_from_the_ledger(tmp_path) -> None:
    frame = market_with_drop(9.2)
    record_target(tmp_path, FRIDAY, {HELD: 0.5})
    run(tmp_path, MONDAY, frame)
    record_target(tmp_path, MONDAY, {HELD: 0.25, NEW: 0.10})
    run(tmp_path, TUESDAY, frame)

    from quantagent.paper.risk import RiskEngine

    engine = RiskEngine(limits(), state_ledger=lg.EventLedger(tmp_path / "operational.jsonl"))

    assert engine.peak_equity is not None and engine.peak_equity >= 999_000.0
    assert [s["scope"] for s in engine.kill_switch.active()] == ["PORTFOLIO"]


def test_no_breach_no_switch(tmp_path) -> None:
    frame = market_with_drop(10.0)
    record_target(tmp_path, FRIDAY, {HELD: 0.5})
    run(tmp_path, MONDAY, frame)
    record_target(tmp_path, MONDAY, {HELD: 0.25, NEW: 0.10})

    result = run(tmp_path, TUESDAY, frame)[0]

    assert result.fill_count == 2
    assert not [
        e for e in lg.EventLedger(tmp_path / "operational.jsonl").read()
        if e.event_type == lg.KILL_SWITCH_TRIGGERED
    ]
