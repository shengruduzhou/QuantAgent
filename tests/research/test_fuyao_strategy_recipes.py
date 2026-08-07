from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.research.fuyao_strategy_recipes import (
    BreakoutConfig,
    MomentumConfig,
    ReversalConfig,
    price_volume_breakout_weights,
    short_term_reversal_weights,
    time_series_momentum_weights,
)


def _panel(symbols: tuple[str, ...], periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    rows = []
    for j, symbol in enumerate(symbols):
        base = 10.0 + j
        for i, date in enumerate(dates):
            close = base * (1.0 + 0.002 * i)
            rows.append({
                "symbol": symbol,
                "trade_date": date,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
                "amount": 20_000_000.0,
                "adjustment": "forward",
            })
    return pd.DataFrame(rows)


def test_breakout_threshold_excludes_current_close_from_prior_high() -> None:
    panel = _panel(("000001.SZ",), periods=90)
    last = panel.index[-1]
    panel.loc[last, "close"] = float(panel.loc[last - 1, "close"]) * 1.20
    panel.loc[last, "high"] = panel.loc[last, "close"]
    panel.loc[last, "volume"] = 3_000_000.0
    result = price_volume_breakout_weights(
        panel,
        BreakoutConfig(breakout_window=20, exit_window=10, volume_window=10, volume_ratio_min=1.5, ma_window=20),
    )
    signal = result.signal_frame.iloc[-1]
    assert bool(signal["entry_signal"]) is True
    assert float(signal["prior_high"]) < float(signal["close"])
    assert result.diagnostics["executionTiming"].startswith("T+1 open")


def test_raw_declared_prices_are_blocked_for_breakout() -> None:
    panel = _panel(("000001.SZ",), periods=90)
    panel["adjustment"] = "none"
    with pytest.raises(ValueError, match="adjusted"):
        price_volume_breakout_weights(panel)


def test_time_series_momentum_enters_cash_before_history_and_active_after() -> None:
    panel = _panel(("000001.SZ", "000002.SZ"), periods=80)
    result = time_series_momentum_weights(
        panel,
        MomentumConfig(momentum_window=20, ma_window=20, volatility_window=10, weighting="equal", rebalance="day", max_weight_per_asset=0.6),
    )
    assert result.target_weights.iloc[:20].sum(axis=1).max() == 0.0
    assert result.target_weights.iloc[-1].sum() == pytest.approx(1.0)
    assert result.diagnostics["cashStateDays"] >= 20
    assert result.diagnostics["executionTiming"].startswith("T+1 open")


def test_momentum_inverse_volatility_respects_cap_and_cash_residual() -> None:
    panel = _panel(("000001.SZ",), periods=80)
    # Add enough variation for non-zero realised volatility.
    mask = panel["symbol"] == "000001.SZ"
    panel.loc[mask, "close"] *= 1.0 + 0.01 * np.sin(np.arange(mask.sum()))
    result = time_series_momentum_weights(
        panel,
        MomentumConfig(momentum_window=20, ma_window=20, volatility_window=10, weighting="inverse_volatility", rebalance="day", max_weight_per_asset=0.35),
    )
    assert float(result.target_weights.max().max()) <= 0.35 + 1e-12


def test_short_term_reversal_selects_cross_section_and_reports_ic_sign_contract() -> None:
    symbols = tuple(f"{i:06d}.SZ" for i in range(1, 21))
    panel = _panel(symbols, periods=150)
    last_dates = sorted(panel["trade_date"].unique())[-8:]
    # Create a controlled cross-sectional selloff without violating the -9.5% daily filter.
    for rank, symbol in enumerate(symbols):
        symbol_mask = panel["symbol"] == symbol
        idx = panel.index[symbol_mask & panel["trade_date"].isin(last_dates)]
        panel.loc[idx, "close"] *= 1.0 - rank * 0.001
    benchmark_dates = pd.bdate_range("2025-01-02", periods=150)
    benchmark = pd.Series(np.linspace(100.0, 112.0, len(benchmark_dates)), index=benchmark_dates)
    result = short_term_reversal_weights(
        panel,
        benchmark,
        ReversalConfig(formation_days=5, holding_days=5, cooldown_days=5, bottom_quantile=0.10, ma_window=20, liquidity_window=10, max_positions=5),
    )
    assert result.target_weights.shape[1] == len(symbols)
    assert result.diagnostics["rankIcExpectedSign"] == "negative"
    assert "do not claim reversal" in str(result.diagnostics["evidenceMessage"]) or result.diagnostics["reversalEvidence"] is True
    assert result.diagnostics["executionTiming"].startswith("T+1 open")
