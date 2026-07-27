"""A-share rules and fidelity-enforced simulation.

Two families of test: the exchange rulebook (which must match the market as it
is *on a given date*, not as it once was), and the simulator's refusal to make
claims its data cannot support.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantagent.backtest import ashare_rules as rules
from quantagent.backtest.microstructure_simulator import (
    BUY,
    MARKETABLE,
    PASSIVE,
    SELL,
    AShareMicrostructureSimulator,
    OrderIntent,
    SimulationRefused,
    simulator_for,
)
from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure import fidelity as fid
from quantagent.data.microstructure import integrity


# --- rulebook ---------------------------------------------------------------
class TestPriceLimits:
    @pytest.mark.parametrize("board,ratio", [
        (rules.SH_MAIN, 0.10), (rules.SZ_MAIN, 0.10),
        (rules.CHINEXT, 0.20), (rules.STAR, 0.20), (rules.BSE, 0.30),
    ])
    def test_ordinary_band_by_board(self, board, ratio):
        limits = rules.price_limits(
            board=board, previous_close=10.0, trade_date="2026-07-24"
        )
        assert limits.ratio == ratio
        assert limits.limit_up == pytest.approx(10.0 * (1 + ratio), abs=0.005)
        assert limits.limit_down == pytest.approx(10.0 * (1 - ratio), abs=0.005)

    def test_st_narrows_the_main_board_band(self):
        limits = rules.price_limits(
            board=rules.SH_MAIN, previous_close=10.0,
            trade_date="2026-07-24", is_st=True,
        )
        assert limits.ratio == 0.05
        assert limits.regime == "ST_BAND"

    def test_st_does_not_narrow_chinext(self):
        limits = rules.price_limits(
            board=rules.CHINEXT, previous_close=10.0,
            trade_date="2026-07-24", is_st=True,
        )
        assert limits.ratio == 0.20

    def test_registration_system_ipo_has_no_limit(self):
        limits = rules.price_limits(
            board=rules.SH_MAIN, previous_close=10.0, trade_date="2026-07-24",
            sessions_since_listing=0,
        )
        assert limits.unlimited
        assert limits.regime == "IPO_NO_LIMIT_WINDOW"

    def test_pre_reform_main_board_ipo_used_the_legacy_band(self):
        """Before 2023-04-10 a main-board IPO was capped at +44%/-36%."""
        limits = rules.price_limits(
            board=rules.SH_MAIN, previous_close=10.0, trade_date="2022-06-01",
            sessions_since_listing=0,
        )
        assert limits.regime == "IPO_LEGACY_APPROVAL_SYSTEM"
        assert limits.limit_up == pytest.approx(14.40, abs=0.005)
        assert limits.limit_down == pytest.approx(6.40, abs=0.005)

    def test_ipo_window_closes_after_five_sessions(self):
        limits = rules.price_limits(
            board=rules.SH_MAIN, previous_close=10.0, trade_date="2026-07-24",
            sessions_since_listing=5,
        )
        assert not limits.unlimited
        assert limits.ratio == 0.10

    def test_bse_ipo_window_is_one_session(self):
        assert rules.price_limits(
            board=rules.BSE, previous_close=10.0, trade_date="2026-07-24",
            sessions_since_listing=0,
        ).unlimited
        assert not rules.price_limits(
            board=rules.BSE, previous_close=10.0, trade_date="2026-07-24",
            sessions_since_listing=1,
        ).unlimited

    def test_session_count_is_never_guessed_from_calendar_days(self):
        """A calendar approximation would mis-band names around holidays."""
        limits = rules.price_limits(
            board=rules.SH_MAIN, previous_close=10.0, trade_date="2026-07-24",
            listing_date="2026-07-22", sessions_since_listing=None,
        )
        assert limits.regime == "ORDINARY"


class TestCosts:
    def test_stamp_duty_is_sell_side_only(self):
        buy = rules.trading_costs(notional_cny=100_000, side=BUY, trade_date="2026-07-24")
        sell = rules.trading_costs(notional_cny=100_000, side=SELL, trade_date="2026-07-24")
        assert buy.stamp_duty == 0.0
        assert sell.stamp_duty == pytest.approx(50.0)

    def test_stamp_duty_halved_on_2023_08_28(self):
        assert rules.stamp_duty_rate("2023-08-27") == pytest.approx(0.0010)
        assert rules.stamp_duty_rate("2023-08-28") == pytest.approx(0.0005)

    def test_transfer_fee_is_charged_both_sides(self):
        for side in (BUY, SELL):
            costs = rules.trading_costs(
                notional_cny=100_000, side=side, trade_date="2026-07-24"
            )
            assert costs.transfer_fee == pytest.approx(1.0)

    def test_commission_floor_applies_to_small_orders(self):
        costs = rules.trading_costs(notional_cny=1_000, side=BUY, trade_date="2026-07-24")
        assert costs.commission == pytest.approx(rules.COMMISSION_MINIMUM_CNY)


class TestLotRules:
    def test_main_board_buys_round_down_to_whole_lots(self):
        assert rules.round_to_lot(350, board=rules.SH_MAIN, side=BUY) == 300

    def test_buy_below_the_minimum_is_not_tradable(self):
        assert rules.round_to_lot(50, board=rules.SH_MAIN, side=BUY) == 0

    def test_star_requires_two_hundred_then_single_shares(self):
        assert rules.round_to_lot(150, board=rules.STAR, side=BUY) == 0
        assert rules.round_to_lot(237, board=rules.STAR, side=BUY) == 237

    def test_odd_lot_sell_only_when_liquidating_in_full(self):
        assert rules.round_to_lot(
            137, board=rules.SH_MAIN, side=SELL, is_full_liquidation=True
        ) == 137
        assert rules.round_to_lot(137, board=rules.SH_MAIN, side=SELL) == 100


class TestTradability:
    def test_cannot_buy_a_locked_limit_up(self):
        verdict = rules.tradability(at_limit_up=True)
        assert verdict.can_buy is False
        assert verdict.can_sell is True

    def test_cannot_sell_a_locked_limit_down(self):
        verdict = rules.tradability(at_limit_down=True)
        assert verdict.can_sell is False
        assert verdict.can_buy is True

    def test_t_plus_one_blocks_same_day_sale(self):
        verdict = rules.tradability(holding_acquired_today=True)
        assert verdict.can_sell is False
        assert any("T+1" in r for r in verdict.reasons)

    def test_suspension_blocks_both_sides(self):
        verdict = rules.tradability(is_suspended=True)
        assert not verdict.can_buy and not verdict.can_sell

    def test_delisting_period_blocks_entry_but_allows_exit(self):
        verdict = rules.tradability(is_delisting_period=True)
        assert verdict.can_buy is False
        assert verdict.can_sell is True


# --- simulator --------------------------------------------------------------
def _tick_events(symbol="600000.SH", n=10, price=9.0, volume=100_000.0):
    return pd.DataFrame({
        "symbol": [symbol] * n,
        "exchange_time": pd.to_datetime(
            [f"2026-07-24 10:00:{i:02d}" for i in range(n)]
        ),
        "ingest_sequence": list(range(n)),
        "price": [price] * n,
        "volume_shares": [volume] * n,
        "data_class": [mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE] * n,
    })


def _level_c() -> fid.FidelityDecision:
    return fid.decide_fidelity(data_classes=[mc.TRADE_TICK])


def _level_b() -> fid.FidelityDecision:
    return fid.decide_fidelity(data_classes=[mc.LEVEL2_SNAPSHOT])


def _level_a() -> fid.FidelityDecision:
    clean = integrity.IntegrityReport(
        family=mc.FAMILY_ORDER, data_class=mc.EXCHANGE_ORDER_EVENT, rows=1, symbols=1,
        checks=[integrity.CheckResult("all", integrity.PASS, "ok")],
    )
    return fid.decide_fidelity(
        data_classes=[mc.EXCHANGE_ORDER_EVENT], integrity_reports=[clean]
    )


class TestFidelityEnforcement:
    def test_not_simulatable_data_is_refused_at_construction(self):
        decision = fid.decide_fidelity(data_classes=[mc.UNKNOWN_SEMANTICS])
        with pytest.raises(SimulationRefused, match="no simulation at all"):
            AShareMicrostructureSimulator(decision)

    def test_level_b_never_populates_queue_position(self):
        """Snapshot depth cannot observe where an order sits in the queue."""
        simulator = AShareMicrostructureSimulator(_level_b())
        book = pd.DataFrame({
            "symbol": ["600000.SH"] * 3,
            "exchange_time": pd.to_datetime([f"2026-07-24 10:00:0{i}" for i in range(3)]),
            "ingest_sequence": [0, 1, 2],
            "ask_price": [9.01, 9.01, 9.02],
            "ask_volume_shares": [5_000.0, 5_000.0, 5_000.0],
            "bid_price": [9.00, 9.00, 9.01],
            "bid_volume_shares": [5_000.0, 5_000.0, 5_000.0],
        })
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 8_000, release_time="09:30:00")],
            book, trade_date="2026-07-24",
        )
        assert result.fills
        assert all(f.queue_position_shares is None for f in result.fills)
        assert "queue_position" not in result.permitted_claims

    def test_level_a_populates_queue_position(self):
        simulator = AShareMicrostructureSimulator(_level_a())
        assert "queue_position" in simulator.decision.permitted_claims

    def test_aggregated_ticks_carry_their_downgrade_into_the_result(self):
        decision = fid.decide_fidelity(
            data_classes=[mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE]
        )
        simulator = AShareMicrostructureSimulator(decision)
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24",
        )
        assert any("3-second aggregates" in d for d in result.downgrades)


class TestSimulatorRules:
    def test_t_plus_one_rejects_the_same_day_sale(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        result = simulator.simulate(
            [OrderIntent("600000.SH", SELL, 1_000, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24",
            state={"600000.SH": {"holding_acquired_today": True}},
        )
        assert result.fills == []
        assert "T+1" in result.rejected[0].reason

    def test_limit_up_blocks_the_buy_but_not_the_sell(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        state = {"600000.SH": {"at_limit_up": True}}
        buy = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24", state=state,
        )
        sell = simulator.simulate(
            [OrderIntent("600000.SH", SELL, 1_000, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24", state=state,
        )
        assert buy.fills == [] and buy.rejected
        assert sell.fills

    def test_suspension_rejects_everything(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24",
            state={"600000.SH": {"is_suspended": True}},
        )
        assert result.rejected and "suspended" in result.rejected[0].reason

    def test_sub_lot_order_is_rejected_not_silently_rounded_up(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 50, release_time="09:30:00")],
            _tick_events(), trade_date="2026-07-24",
        )
        assert result.fills == []
        assert "minimum lot" in result.rejected[0].reason

    def test_participation_cap_prevents_filling_more_than_the_market_traded(self):
        simulator = AShareMicrostructureSimulator(_level_c(), participation_cap=0.10)
        events = _tick_events(n=2, volume=10_000.0)  # 20,000 shares traded
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 100_000, release_time="09:30:00")],
            events, trade_date="2026-07-24",
        )
        assert result.filled_shares == pytest.approx(2_000.0)  # 10% of 20,000
        assert result.unfilled_shares > 0

    def test_latency_delays_participation(self):
        """An order released at 10:00:05 cannot fill on the 10:00:00 print."""
        simulator = AShareMicrostructureSimulator(_level_c(), latency_ms=0)
        events = _tick_events(n=10)
        early = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="10:00:00")],
            events, trade_date="2026-07-24",
        )
        late = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="10:00:08")],
            events, trade_date="2026-07-24",
        )
        assert len(early.fills) >= len(late.fills)
        for fill in late.fills:
            assert str(fill.fill_time) >= "2026-07-24 10:00:08"

    def test_orders_do_not_fill_during_the_lunch_break(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        events = _tick_events(n=3)
        events["exchange_time"] = pd.to_datetime(["2026-07-24 12:00:00"] * 3)
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, release_time="09:30:00")],
            events, trade_date="2026-07-24",
        )
        assert result.fills == []

    def test_sell_pays_stamp_duty_and_buy_does_not(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        events = _tick_events(n=5, volume=1_000_000.0)
        buy = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 10_000, release_time="09:30:00")],
            events, trade_date="2026-07-24",
        )
        sell = simulator.simulate(
            [OrderIntent("600000.SH", SELL, 10_000, release_time="09:30:00")],
            events, trade_date="2026-07-24",
        )
        assert sum(f.costs["stamp_duty"] for f in buy.fills) == 0.0
        assert sum(f.costs["stamp_duty"] for f in sell.fills) > 0.0

    def test_passive_limit_is_respected(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        events = _tick_events(n=5, price=9.10, volume=1_000_000.0)
        result = simulator.simulate(
            [OrderIntent("600000.SH", BUY, 1_000, style=PASSIVE,
                         limit_price=9.00, release_time="09:30:00")],
            events, trade_date="2026-07-24",
        )
        assert result.fills == []

    def test_symbol_without_events_is_rejected_not_filled(self):
        simulator = AShareMicrostructureSimulator(_level_c())
        result = simulator.simulate(
            [OrderIntent("000001.SZ", BUY, 1_000, board=rules.SZ_MAIN)],
            _tick_events(), trade_date="2026-07-24",
        )
        assert result.fills == []
        assert "no market events" in result.rejected[0].reason


class TestSimulatorFactory:
    def test_fidelity_is_derived_from_the_data_not_chosen(self):
        simulator = simulator_for({mc.LEVEL2_SNAPSHOT: pd.DataFrame()})
        assert simulator.decision.level == fid.LEVEL_B

    def test_bar_only_data_gives_level_d(self):
        simulator = simulator_for({}, has_bars=True)
        assert simulator.decision.level == fid.LEVEL_D
        assert "queue_position" not in simulator.decision.permitted_claims
