import numpy as np
import pandas as pd

from quantagent.portfolio.hedge_instrument_selector import (
    HedgeAction,
    select_hedge,
)
from quantagent.portfolio.portfolio_beta_model import (
    PortfolioBeta,
    benchmark_returns_from_close,
    portfolio_beta,
    returns_panel_from_close,
)
from quantagent.themes.lifecycle_trading_rules import (
    apply_lifecycle_caps,
    lifecycle_rule,
)
from quantagent.v7.schemas import ThemeLifecycleStage


def test_lifecycle_rule_per_stage_caps_make_sense():
    assert lifecycle_rule(ThemeLifecycleStage.POLICY_SEED).allow_open is False
    assert lifecycle_rule(ThemeLifecycleStage.NARRATIVE_FORMATION).max_position_weight <= 0.03
    earnings = lifecycle_rule(ThemeLifecycleStage.EARNINGS_REALIZATION)
    assert earnings.max_position_weight >= 0.05
    assert earnings.allow_add is True
    bubble = lifecycle_rule(ThemeLifecycleStage.VALUATION_BUBBLE)
    assert bubble.require_trim is True
    assert bubble.allow_open is False
    invalidated = lifecycle_rule(ThemeLifecycleStage.INVALIDATED)
    assert invalidated.require_full_exit is True


def test_apply_lifecycle_caps_zeros_invalidated_and_trims_bubble():
    target_weights = {
        "AAA.SH": 0.06,
        "BBB.SH": 0.10,
        "CCC.SH": 0.02,
    }
    member_lifecycle = {
        "AAA.SH": ThemeLifecycleStage.EARNINGS_REALIZATION,
        "BBB.SH": ThemeLifecycleStage.VALUATION_BUBBLE,
        "CCC.SH": ThemeLifecycleStage.INVALIDATED,
    }
    member_theme = {"AAA.SH": "ai_compute", "BBB.SH": "ai_compute", "CCC.SH": "ai_compute"}
    capped, notes = apply_lifecycle_caps(target_weights, member_lifecycle, member_theme)
    # Invalidated → zero
    assert capped["CCC.SH"] == 0.0
    # Bubble → trimmed to the per-position cap
    assert capped["BBB.SH"] <= lifecycle_rule(ThemeLifecycleStage.VALUATION_BUBBLE).max_position_weight + 1e-9
    assert any("CCC.SH" in note for note in notes)


def test_portfolio_beta_estimates_market_exposure():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    benchmark_close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(dates))))
    benchmark_panel = pd.DataFrame(
        {"trade_date": dates, "symbol": "000300.SH", "close": benchmark_close}
    )
    symbols = ["A.SH", "B.SH"]
    rows = [benchmark_panel]
    for beta_target, symbol in zip([1.30, 0.40], symbols):
        bench_ret = pd.Series(benchmark_close).pct_change().fillna(0.0).to_numpy()
        prices = [10.0]
        for ret in bench_ret[1:]:
            prices.append(prices[-1] * (1.0 + beta_target * ret + rng.normal(0.0, 0.002)))
        rows.append(pd.DataFrame({"trade_date": dates, "symbol": symbol, "close": prices}))
    price_panel = pd.concat(rows, ignore_index=True)
    returns = returns_panel_from_close(price_panel)
    benchmark = benchmark_returns_from_close(price_panel, "000300.SH")
    pb = portfolio_beta({"A.SH": 0.6, "B.SH": 0.4}, returns, benchmark)
    assert isinstance(pb, PortfolioBeta)
    # Weighted beta of 0.6*1.30 + 0.4*0.40 ≈ 0.94
    assert 0.80 <= pb.portfolio_beta <= 1.05


def test_select_hedge_picks_sector_and_broad_etfs_when_beta_too_high():
    pb = PortfolioBeta(
        portfolio_beta=1.15,
        benchmark_symbol="000300.SH",
        sample_count=200,
        sector_betas={},
        diagnostics={},
    )
    rec = select_hedge(
        pb,
        target_beta=0.30,
        hedge_need_score=0.55,
        sector_exposure={"semi": 0.20},
    )
    assert HedgeAction.NONE not in rec.actions
    assert rec.expected_beta_reduction > 0.0
    assert sum(rec.instrument_weights.values()) <= 0.26


def test_select_hedge_returns_none_when_hedge_need_low_and_beta_under_target():
    pb = PortfolioBeta(
        portfolio_beta=0.20,
        benchmark_symbol="000300.SH",
        sample_count=200,
        sector_betas={},
        diagnostics={},
    )
    rec = select_hedge(pb, target_beta=0.40, hedge_need_score=0.10)
    assert HedgeAction.NONE in rec.actions
    assert rec.instrument_weights == {}
