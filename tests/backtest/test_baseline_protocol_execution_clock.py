from __future__ import annotations

import pandas as pd

import scripts.baseline_protocol as baseline_protocol
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.backtest.strict_v8 import run_strict_backtest_v8


def _market() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=3)
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "trade_date": date,
                "symbol": "600000.SH",
                "close": 10.0 + index,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        )
    return pd.DataFrame(rows)


def test_baseline_targets_remain_on_signal_date_and_execute_once_next_session(tmp_path) -> None:
    market = _market()
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    signal_date = sessions[0]
    preds = pd.DataFrame(
        [
            {
                "trade_date": signal_date,
                "symbol": "600000.SH",
                "alpha_score": 1.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        ]
    )

    targets = baseline_protocol._target_weights(
        preds,
        "alpha_score",
        1,
        eligible_only=True,
    )

    # baseline_protocol must never move the target index to T+1 itself.
    assert list(targets.index) == [signal_date]
    assert not hasattr(baseline_protocol, "_apply_delay")

    result = run_strict_backtest_v8(
        targets,
        market,
        config=AShareExecutionSimulationConfig(
            initial_cash=100_000.0,
            slippage_bps=0.0,
            audit_log_dir=str(tmp_path / "audit"),
        ),
    )

    assert not result.trades.empty
    first = result.trades.iloc[0]
    assert pd.Timestamp(first["signal_date"]) == signal_date
    assert pd.Timestamp(first["execution_date"]) == sessions[1]
    assert pd.Timestamp(first["trade_date"]) == sessions[1]
    assert result.config["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS


def test_legacy_variant_name_does_not_reintroduce_a_manual_delay() -> None:
    source = baseline_protocol.evaluate.__code__.co_names

    assert "_apply_delay" not in source
    assert baseline_protocol.EXECUTION_TIMING_SEMANTICS
