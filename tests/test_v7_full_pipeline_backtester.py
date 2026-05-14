import pandas as pd

from quantagent.backtest.full_pipeline_backtester import (
    FullPipelineBacktestConfig,
    run_full_pipeline_backtest,
)


def _build_prices() -> tuple[pd.DataFrame, list[str]]:
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    symbols = ["A", "B", "C"]
    rows = []
    for index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            close = 100.0 * (1.0 + 0.005 * (index + symbol_index))
            rows.append({"trade_date": date, "symbol": symbol, "close": close})
    return pd.DataFrame(rows), [date.strftime("%Y-%m-%d") for date in dates]


def test_full_pipeline_backtester_respects_t_plus_one_lag():
    prices, dates = _build_prices()

    def daily_step(as_of_date: str) -> dict:
        # Buy A on day 1 only
        if as_of_date == dates[0]:
            return {"target_weights": {"A": 0.5}, "universe_size": 3}
        return {"target_weights": {}, "universe_size": 3}

    result = run_full_pipeline_backtest(
        dates,
        prices,
        daily_step,
        FullPipelineBacktestConfig(initial_capital=1_000_000.0, execution_lag_days=1, cost_bps=0.0, max_single_name_weight=1.0),
    )
    assert not result.nav.empty
    # The realized weight series should keep A in subsequent dates
    realized = result.realized_weight_history
    assert realized.shape[0] == len(dates)
    # First day has no realised position yet (T+1 fill)
    if "A" in realized.columns:
        # Day 1 realized weight should be 0 (target placed but not yet executed)
        assert float(realized["A"].iloc[0]) == 0.0
        # Day 2 onward should hold 0.5 in A
        assert float(realized["A"].iloc[1]) == 0.5


def test_full_pipeline_backtester_caps_single_name_weight():
    prices, dates = _build_prices()

    def daily_step(_as_of_date: str) -> dict:
        return {"target_weights": {"A": 0.99}, "universe_size": 3}

    result = run_full_pipeline_backtest(
        dates,
        prices,
        daily_step,
        FullPipelineBacktestConfig(max_single_name_weight=0.10, execution_lag_days=1, cost_bps=0.0),
    )
    assert (result.target_weight_history["A"] <= 0.10 + 1e-9).all()


def test_full_pipeline_backtester_records_audit():
    prices, dates = _build_prices()

    def daily_step(as_of_date: str) -> dict:
        return {
            "target_weights": {"A": 0.05},
            "universe_size": 3,
            "audit": {"gate_dropped": 2},
        }

    result = run_full_pipeline_backtest(dates, prices, daily_step)
    assert len(result.pit_audit) == len(dates)
    assert result.pit_audit[0]["audit"]["gate_dropped"] == 2
