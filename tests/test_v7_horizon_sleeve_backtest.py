from __future__ import annotations

import pandas as pd


def test_executable_backtest_uses_horizon_sleeves(tmp_path):
    from quantagent.training.v7_experiment import V7TrainingConfig, _compute_executable_backtest

    dates = pd.date_range("2025-01-02", periods=15, freq="B")
    symbols = [f"S{i:03d}" for i in range(8)]
    horizons = (1, 5, 20, 60, 120, 126)
    rows = []
    for date_idx, date in enumerate(dates):
        for sym_idx, symbol in enumerate(symbols):
            one_day_return = 0.001 + 0.0001 * sym_idx
            for horizon in horizons:
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "horizon": horizon,
                        "prediction": sym_idx * 0.01 + horizon * 0.00001,
                        "forward_return_1d": one_day_return,
                        f"forward_return_{horizon}d": one_day_return * max(1, horizon),
                        "sample_role": "validation",
                        "fold_id": 0,
                    }
                )
    predictions = pd.DataFrame(rows)
    result = _compute_executable_backtest(
        predictions,
        V7TrainingConfig(
            executable_strategy="horizon_sleeves",
            executable_base_gross=0.60,
            executable_max_weight_per_name=0.20,
            executable_max_turnover=1.0,
            target_max_drawdown=0.10,
            output_dir=str(tmp_path),
        ),
        tmp_path,
    )

    assert result["executable_backtest_status"] == "ok"
    assert result["executable_strategy"] == "horizon_sleeves"
    assert result["max_drawdown_target_passed"] is True
    assert result["average_gross_exposure"] <= 0.60 + 1e-9
    assert (tmp_path / "sleeve_config.json").exists()
