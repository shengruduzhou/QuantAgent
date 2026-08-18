"""Declared friction and applied friction are one number, and impact is a law.

Round 22 / R1 (backtest), closing F-04 and F-05 from
`docs/audits/round21/01_backtest.md`.

**F-04.** `CostModelConfig.slippage_bps` (5.0) sat inside every `BacktestConfig`,
was serialised with the run and quoted by reports, while the engine actually
moved the fill price by a private `FillModelConfig.slippage_bps` of 2.0. Every
report that said "5 bps slippage" described a run that charged 2 — an
optimistic gap of 3 bps per side that no output revealed. The fill model now
reads `CostModelConfig`; the second field does not exist.

The repository has previously been burned in the *opposite* direction: the
engine once charged `cost.slippage_bps` again as a fee on top of the already
moved fill price, an effective ~7 bps. So it is not enough to check the number
went up. `test_slippage_is_a_price_move_and_never_also_a_fee` reconstructs the
cash movement from the fill price and the three explicit fees and asserts there
is nothing left over.

**F-05.** The engine's impact term was `impact_bps * (filled / volume)` —
linear, with `impact_bps = 1.0`. Under the shipped participation cap of 5% that
can never exceed 0.05 bps, so capacity effects did not exist in the fast engine
at all, while `trusted_cost_model_config()` published a square-root model with
`impact_alpha_bps = 10.0`. Both now evaluate the same square-root law.
"""

from __future__ import annotations

from math import sqrt

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester
from quantagent.backtest.fill_model import AShareFillModel, FillModelConfig
from quantagent.execution.broker_base import OrderSide
from quantagent.execution.cost_model import AShareCostModel
from quantagent.quant_math.transaction_cost import (
    CostModelConfig,
    square_root_impact_bps,
)

SYMBOL = "600000.SH"


# ---------------------------------------------------------------------------
# F-04: one source
# ---------------------------------------------------------------------------
def test_fill_model_has_no_slippage_field_of_its_own() -> None:
    """A second field is a second answer. There must not be one to diverge."""
    assert not hasattr(FillModelConfig(), "slippage_bps")
    assert not hasattr(FillModelConfig(), "impact_bps")


def test_the_applied_slippage_is_the_declared_slippage() -> None:
    """What `BacktestConfig.cost` says is what the fill price does."""
    cost = CostModelConfig()
    model = AShareFillModel(FillModelConfig(), cost)
    # volume so large that impact is below the assertion tolerance.
    result = model.fill("buy", 1_000, 10.0, 1e18)

    applied_bps = (result.fill_price / 10.0 - 1.0) * 10_000.0
    assert applied_bps == pytest.approx(cost.slippage_bps, abs=1e-6)
    assert result.slippage_bps == pytest.approx(cost.slippage_bps)


def test_changing_the_declared_slippage_changes_the_fill_price() -> None:
    """The link is live, not a coincidence of two equal defaults."""
    for declared in (0.0, 3.0, 12.5):
        model = AShareFillModel(FillModelConfig(), CostModelConfig(slippage_bps=declared))
        result = model.fill("sell", 1_000, 10.0, 1e18)
        applied_bps = (1.0 - result.fill_price / 10.0) * 10_000.0
        assert applied_bps == pytest.approx(declared, abs=1e-6)


def test_the_engine_fills_at_the_slippage_its_own_config_declares() -> None:
    """End to end: the number in the config is the number in the trade log."""
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    prices = pd.DataFrame(
        [
            {
                "trade_date": d, "symbol": SYMBOL, "open": 10.0, "close": 10.0,
                "high": 10.0, "low": 10.0, "pre_close": 10.0,
                "volume": 1e15, "amount": 1e16,
            }
            for d in dates
        ]
    )
    weights = pd.DataFrame({SYMBOL: [0.10, 0.10, 0.10]}, index=dates)
    config = BacktestConfig(initial_nav=1_000_000.0, cost=CostModelConfig(slippage_bps=9.0))
    result = EventDrivenBacktester(config).run(weights, prices)

    trade = result.trades.iloc[0]
    assert (float(trade["price"]) / 10.0 - 1.0) * 10_000.0 == pytest.approx(9.0, abs=1e-4)


def test_slippage_is_a_price_move_and_never_also_a_fee() -> None:
    """Guard against re-introducing the double charge that was fixed earlier.

    Cash out must be exactly `shares x fill_price + commission + transfer`.
    Any additional slippage deduction shows up as a residual.
    """
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    prices = pd.DataFrame(
        [
            {
                "trade_date": d, "symbol": SYMBOL, "open": 10.0, "close": 10.0,
                "high": 10.0, "low": 10.0, "pre_close": 10.0,
                "volume": 1e8, "amount": 1e9,
            }
            for d in dates
        ]
    )
    weights = pd.DataFrame({SYMBOL: [0.10, 0.10, 0.10]}, index=dates)
    result = EventDrivenBacktester(BacktestConfig(initial_nav=1_000_000.0)).run(weights, prices)

    trade = result.trades.iloc[0]
    shares = float(trade["shares"])
    expected_cash_out = (
        shares * float(trade["price"])
        + float(trade["commission"])
        + float(trade["transfer_fee"])
    )
    # NAV on the fill bar = remaining cash + shares marked at that bar's close.
    nav_on_fill_bar = float(result.nav_curve.loc[dates[1]])
    cash_after = nav_on_fill_bar - shares * 10.0
    charged = 1_000_000.0 - cash_after
    assert charged == pytest.approx(expected_cash_out, abs=1e-6), (
        "cash left the account beyond price + commission + transfer: "
        "slippage is being charged twice"
    )


# ---------------------------------------------------------------------------
# F-05: impact is the square-root law, and it is the same one
# ---------------------------------------------------------------------------
def test_impact_follows_the_square_root_law_not_a_linear_one() -> None:
    """Quadrupling participation must double impact, not quadruple it."""
    cost = CostModelConfig()
    small = square_root_impact_bps(0.0025, cost)
    large = square_root_impact_bps(0.01, cost)
    assert large / small == pytest.approx(2.0, rel=1e-12)
    assert small == pytest.approx(10.0 * sqrt(0.0025), rel=1e-12)


def test_the_fast_engine_and_the_venue_cost_model_charge_the_same_impact() -> None:
    """One law, two call sites. Divergence here is how a trust certificate lies."""
    venue = AShareCostModel()
    fast = CostModelConfig()
    assert fast.impact_alpha_bps == pytest.approx(venue.impact_alpha_bps)

    for participation in (1e-6, 1e-4, 0.01, 0.05, 0.10, 0.25):
        value = 1_000_000.0
        venue_bps = (
            venue.calculate(
                OrderSide.BUY, 100_000, 10.0, participation_rate=participation
            )["impact_cost"]
            / value
            * 10_000.0
        )
        assert square_root_impact_bps(participation, fast) == pytest.approx(
            venue_bps, rel=1e-12
        ), f"impact laws disagree at participation={participation}"


def test_impact_is_material_at_the_participation_cap() -> None:
    """The old linear term topped out at 0.05 bps: capacity did not exist.

    At the shipped 5% cap the square-root law charges 10 x sqrt(0.05) = 2.236
    bps, ~45x the ceiling of the term it replaced.
    """
    model = AShareFillModel()
    # Ask for far more than the cap so the fill is capped at 5% of volume.
    result = model.fill("buy", 10_000_000, 10.0, 1_000_000.0)

    assert result.participation_rate == pytest.approx(0.05)
    assert result.impact_bps == pytest.approx(10.0 * sqrt(0.05), rel=1e-9)
    assert result.impact_bps > 2.0
    linear_ceiling = 1.0 * 0.05  # the replaced term, at the same cap
    assert result.impact_bps / linear_ceiling > 40.0


def test_a_bigger_order_into_the_same_book_pays_a_worse_price() -> None:
    """Capacity, stated as a monotonicity the engine must exhibit."""
    model = AShareFillModel()
    thin = model.fill("buy", 50_000, 10.0, 1_000_000.0)
    thick = model.fill("buy", 50_000, 10.0, 100_000_000.0)
    assert thin.fill_price > thick.fill_price
    assert thin.impact_bps > thick.impact_bps


def test_impact_is_reported_alongside_the_fill_not_only_folded_into_the_price() -> None:
    """A charge nobody can read is a charge nobody can audit."""
    result = AShareFillModel().fill("buy", 100_000, 10.0, 10_000_000.0)
    reconstructed = 10.0 * (1.0 + (result.slippage_bps + result.impact_bps) / 10_000.0)
    assert result.fill_price == pytest.approx(reconstructed, rel=1e-12)
    assert result.participation_rate == pytest.approx(0.01)
