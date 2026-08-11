from __future__ import annotations

import pandas as pd
import pytest

from quantagent.domain.lineage import Lineage
from quantagent.paper import ledger as operational_ledger
from quantagent.paper.account_target_state import (
    PaperAccountStateRefused,
    PaperAccountTargetState,
    recover_paper_account_target_state,
    reconcile_target_to_canonical_account,
)
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.orders import BUY, Order
from quantagent.paper.portfolio import Portfolio
from quantagent.portfolio.v7_target_weights import V7TargetWeightsResult


SYMBOL = "600000.SH"
DATE = "2026-08-10"
INITIAL = 100_000.0


def _market_panel(*, adjustment: str = "raw") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(DATE),
                "symbol": SYMBOL,
                "close": 10.0,
                "amount": 100_000.0,
                "price_adjustment": adjustment,
                "execution_eligible": adjustment == "raw",
            }
        ]
    )


def _partial_fill_ledger(tmp_path):
    canonical_path = tmp_path / "canonical.jsonl"
    portfolio = Portfolio(portfolio_id="v7-paper", cash=INITIAL, initial_cash=INITIAL)
    broker = PaperBroker(
        portfolio,
        operational_ledger.EventLedger(tmp_path / "operational.jsonl"),
        run_id="partial-fill-test",
        config=BrokerConfig(participation_cap=0.02, slippage_bps=0.0, impact_coefficient=0.0),
        canonical_ledger_path=str(canonical_path),
        lineage=Lineage(run_id="partial-fill-test", strategy_version_id="test"),
    )
    market = MarketSnapshot(
        symbol=SYMBOL,
        trade_date=DATE,
        last_price=10.0,
        previous_close=10.0,
        session_volume=10_000.0,
        clock="14:59:00",
    )
    order = broker.submit(
        Order(symbol=SYMBOL, side=BUY, quantity=1_000, limit_price=10.5),
        market,
    )
    assert 0 < order.filled_quantity < 1_000
    broker.cancel(order.order_id, market)
    assert not order.is_open
    return canonical_path, float(order.filled_quantity)


def test_empty_valid_canonical_ledger_is_known_cash_only_first_run(tmp_path) -> None:
    state = recover_paper_account_target_state(
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        market_panel=pd.DataFrame(),
        as_of_date=DATE,
        portfolio_id="v7-paper",
        initial_cash=INITIAL,
    )
    assert state.current_weights.empty
    assert state.quantities.empty
    assert state.cash == pytest.approx(INITIAL)
    assert state.nav == pytest.approx(INITIAL)
    assert state.canonical_records == 0
    assert state.canonical_head_hash == "0" * 64
    assert len(state.account_state_sha256) == 64


def test_partial_fill_not_desired_order_size_defines_current_weight(tmp_path) -> None:
    canonical_path, filled = _partial_fill_ledger(tmp_path)
    state = recover_paper_account_target_state(
        canonical_ledger_path=str(canonical_path),
        market_panel=_market_panel(),
        as_of_date=DATE,
        portfolio_id="v7-paper",
        initial_cash=INITIAL,
    )
    assert state.quantities[SYMBOL] == pytest.approx(filled)
    assert filled < 1_000
    expected_market_value = filled * 10.0
    expected_weight = expected_market_value / state.nav
    assert state.current_weights[SYMBOL] == pytest.approx(expected_weight)
    assert state.canonical_records > 0
    assert state.canonical_head_hash != "0" * 64


def test_adjusted_mark_for_actual_holding_fails_closed(tmp_path) -> None:
    canonical_path, _ = _partial_fill_ledger(tmp_path)
    with pytest.raises(PaperAccountStateRefused, match="raw/unadjusted"):
        recover_paper_account_target_state(
            canonical_ledger_path=str(canonical_path),
            market_panel=_market_panel(adjustment="qfq"),
            as_of_date=DATE,
            portfolio_id="v7-paper",
            initial_cash=INITIAL,
        )


def test_dropped_current_holding_counts_toward_turnover_union() -> None:
    state = PaperAccountTargetState(
        as_of_date=DATE,
        current_weights=pd.Series({"A": 0.20}, dtype=float),
        quantities=pd.Series({"A": 2_000.0}, dtype=float),
        cash=80_000.0,
        nav=100_000.0,
        canonical_records=7,
        canonical_head_hash="a" * 64,
        account_state_sha256="b" * 64,
    )
    desired = V7TargetWeightsResult(
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp(DATE), "B": 1.0}]
        ),
        diagnostics={"status": "passed"},
    )
    frozen = reconcile_target_to_canonical_account(
        desired,
        account_state=state,
        max_turnover=0.40,
    )
    row = frozen.target_weights.iloc[0]
    # Requested L1 churn is sell A .20 + buy B 1.00 = 1.20.  A 0.40
    # budget therefore scales both legs by one third; the old implementation
    # ignored A entirely because it reindexed previous holdings to target names.
    assert row["A"] == pytest.approx(0.20 * (2.0 / 3.0))
    assert row["B"] == pytest.approx(1.0 / 3.0)
    diag = frozen.diagnostics["canonical_account_reconciliation"]
    assert diag["requested_l1_weight_churn"] == pytest.approx(1.20)
    assert diag["applied_l1_weight_churn"] == pytest.approx(0.40)
    assert diag["dropped_current_symbols_counted"] == 1


def test_partial_fill_baseline_not_previous_desired_target_controls_next_freeze() -> None:
    state = PaperAccountTargetState(
        as_of_date=DATE,
        current_weights=pd.Series({"A": 0.20}, dtype=float),
        quantities=pd.Series({"A": 2_000.0}, dtype=float),
        cash=80_000.0,
        nav=100_000.0,
        canonical_records=9,
        canonical_head_hash="c" * 64,
        account_state_sha256="d" * 64,
    )
    desired = V7TargetWeightsResult(
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp(DATE), "A": 0.50}]
        ),
        diagnostics={},
    )
    frozen = reconcile_target_to_canonical_account(
        desired,
        account_state=state,
        max_turnover=0.10,
    )
    # Actual account is 20%, so only 10 percentage points may be added. If the
    # previous desired 50% had been reused as the baseline this would incorrectly
    # emit 50% again after a partial fill.
    assert frozen.target_weights.iloc[0]["A"] == pytest.approx(0.30)
    assert frozen.diagnostics["canonical_account_reconciliation"]["applied_l1_weight_churn"] == pytest.approx(0.10)
