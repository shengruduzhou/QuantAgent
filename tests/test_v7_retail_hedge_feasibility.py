from quantagent.portfolio.hedge_instrument_selector import (
    HedgeAction,
    HedgeInstrument,
    HedgeRecommendation,
)
from quantagent.portfolio.retail_hedge_feasibility import (
    RetailAccountCapabilities,
    filter_recommendation_for_retail,
)


def _instruments() -> tuple[HedgeInstrument, ...]:
    return (
        HedgeInstrument("510300", "CSI300 ETF", "000300.SH", estimated_beta=1.0, short_via_etf_inverse=False),
        HedgeInstrument("inverse_csi300", "Inverse CSI300 ETF", "000300.SH", estimated_beta=-1.0, short_via_etf_inverse=True),
    )


def test_retail_blocks_inverse_etf_when_capability_disabled():
    recommendation = HedgeRecommendation(
        actions=(HedgeAction.BROAD_INDEX_HEDGE, HedgeAction.CASH_BUFFER),
        instrument_weights={"510300": 0.10, "inverse_csi300": 0.05},
        expected_beta_reduction=0.20,
        expected_cost_bps=10.0,
        rationale="test",
    )
    result = filter_recommendation_for_retail(
        recommendation,
        RetailAccountCapabilities(can_buy_etf_inverse=False),
        instrument_universe=_instruments(),
    )
    assert "inverse_csi300" not in result.recommendation.instrument_weights
    assert "510300" in result.recommendation.instrument_weights
    assert any("inverse_csi300_blocked" in note for note in result.audit_notes)


def test_retail_falls_back_to_cash_when_all_blocked():
    recommendation = HedgeRecommendation(
        actions=(HedgeAction.BROAD_INDEX_HEDGE,),
        instrument_weights={},
        expected_beta_reduction=0.0,
        expected_cost_bps=0.0,
        rationale="test",
    )
    result = filter_recommendation_for_retail(
        recommendation,
        RetailAccountCapabilities(
            can_short_index_futures=False,
            can_buy_etf_inverse=False,
            can_hold_cash=True,
        ),
    )
    assert HedgeAction.CASH_BUFFER in result.recommendation.actions
    assert "broad_index_hedge" in result.blocked_actions


def test_retail_keeps_cash_and_gross_reduction_for_default_account():
    recommendation = HedgeRecommendation(
        actions=(HedgeAction.CASH_BUFFER, HedgeAction.REDUCE_GROSS),
        instrument_weights={},
        expected_beta_reduction=0.0,
        expected_cost_bps=0.0,
        rationale="test",
    )
    result = filter_recommendation_for_retail(recommendation, RetailAccountCapabilities())
    assert HedgeAction.CASH_BUFFER in result.recommendation.actions
    assert HedgeAction.REDUCE_GROSS in result.recommendation.actions
    assert result.feasibility_score == 1.0
