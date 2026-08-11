from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    ExecutionTimingViolation,
    simulate_ashare_target_weights,
)
from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    validate_execution_trace,
)
from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights


def _market(*, periods: int = 4, limit_up_second: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
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
                "is_limit_up": bool(limit_up_second and index == 1),
                "is_limit_down": False,
            }
        )
    return pd.DataFrame(rows)


def test_signal_date_executes_only_on_next_market_session_close(tmp_path) -> None:
    market = _market(periods=4)
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    targets = pd.DataFrame(
        {"600000.SH": [0.10, 0.10, 0.10]},
        index=sessions[:3],
    )
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=str(tmp_path / "audit")),
    )

    schedules = result.execution_trace[result.execution_trace["record_type"] == "session_mapping"]
    assert len(schedules) == 3
    assert schedules["execution_timing_semantics"].eq(EXECUTION_TIMING_SEMANTICS).all()
    assert (pd.to_datetime(schedules["execution_date"]) > pd.to_datetime(schedules["signal_date"])).all()
    assert schedules.iloc[0]["signal_date"] == sessions[0]
    assert schedules.iloc[0]["execution_date"] == sessions[1]
    assert not result.order_audit.empty
    first = result.order_audit.iloc[0]
    assert first["signal_date"] == sessions[0]
    assert first["execution_date"] == sessions[1]
    assert first["trade_date"] == sessions[1]
    assert first["price_source"] == "close"
    assert float(first["reference_price"]) == 11.0
    assert result.config["execution_trace_ok"] is True
    assert result.config["target_input_index_semantics"] == "undeclared_legacy_signal_date"
    assert isinstance(result.config["execution_trace_sha256"], str)
    assert len(result.config["execution_trace_sha256"]) == 64


def test_signal_dated_hold_band_is_mapped_exactly_once(tmp_path) -> None:
    market = _market(periods=4)
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    predictions = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": [sessions[0]],
            "alpha_score": [1.0],
        }
    )
    targets = build_hold_band_weights(
        predictions,
        config=HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=1, delay_days=0),
        trade_dates=list(sessions),
    )
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=str(tmp_path / "audit")),
    )
    mapping = result.execution_trace[result.execution_trace["record_type"] == "session_mapping"].iloc[0]
    assert pd.Timestamp(mapping["signal_date"]) == sessions[0]
    assert pd.Timestamp(mapping["execution_date"]) == sessions[1]
    assert result.config["target_input_index_semantics"] == "signal_date"


def test_execution_dated_hold_band_is_rejected_before_second_delay(tmp_path) -> None:
    market = _market(periods=4)
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    predictions = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": [sessions[0]],
            "alpha_score": [1.0],
        }
    )
    targets = build_hold_band_weights(
        predictions,
        config=HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=1, delay_days=1),
        trade_dates=list(sessions),
    )
    assert targets.index[0] == sessions[1]
    with pytest.raises(ExecutionTimingViolation, match="requires signal-dated target weights"):
        simulate_ashare_target_weights(
            targets,
            market,
            AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=str(tmp_path / "audit")),
        )


def test_last_signal_without_next_session_is_explicitly_terminal_censored(tmp_path) -> None:
    market = _market(periods=3)
    last = pd.Timestamp(market["trade_date"].max())
    targets = pd.DataFrame({"600000.SH": [0.10]}, index=[last])
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
    )
    validation = validate_execution_trace(result.execution_trace)
    assert validation.ok is True, validation.reasons
    assert validation.mapped_signal_days == 0
    assert validation.terminal_censored_signal_days == 1
    schedule = result.execution_trace.iloc[0]
    assert schedule["record_type"] == "session_mapping"
    assert schedule["status"] == "unmapped"
    assert schedule["reason"] == "no_next_market_session"
    assert pd.isna(schedule["execution_date"])
    assert result.order_audit.empty


def test_non_session_signal_date_fails_closed(tmp_path) -> None:
    market = _market(periods=4)
    targets = pd.DataFrame({"600000.SH": [0.10]}, index=[pd.Timestamp("2024-01-06")])
    with pytest.raises(ExecutionTimingViolation, match="signal_date_not_market_session"):
        simulate_ashare_target_weights(
            targets,
            market,
            AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
        )


def test_missing_required_next_session_bar_fails_closed(tmp_path) -> None:
    market = _market(periods=3)
    first = pd.Timestamp(market["trade_date"].min())
    targets = pd.DataFrame(
        {"600000.SH": [0.05], "000001.SZ": [0.05]},
        index=[first],
    )
    with pytest.raises(ExecutionTimingViolation, match="missing_execution_bar"):
        simulate_ashare_target_weights(
            targets,
            market,
            AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
        )


def test_next_session_limit_up_rule_is_preserved(tmp_path) -> None:
    market = _market(periods=3, limit_up_second=True)
    first = pd.Timestamp(market["trade_date"].min())
    second = sorted(pd.to_datetime(market["trade_date"].unique()))[1]
    targets = pd.DataFrame({"600000.SH": [0.10]}, index=[first])
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=str(tmp_path / "audit")),
    )
    assert len(result.order_audit) == 1
    order = result.order_audit.iloc[0]
    assert order["status"] == "rejected"
    assert order["last_message"] == "limit_up_no_buy"
    assert pd.Timestamp(order["execution_date"]) == pd.Timestamp(second)


def test_trace_validator_rejects_same_session_tamper(tmp_path) -> None:
    market = _market(periods=3)
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    targets = pd.DataFrame({"600000.SH": [0.10]}, index=[sessions[0]])
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
    )
    tampered = result.execution_trace.copy()
    schedule_index = tampered.index[tampered["record_type"] == "session_mapping"][0]
    tampered.loc[schedule_index, "execution_date"] = tampered.loc[schedule_index, "signal_date"]
    report = validate_execution_trace(tampered)
    assert report.ok is False
    assert "execution_trace_same_or_prior_session_execution" in report.reasons


def test_non_terminal_unmapped_signal_still_fails_closed(tmp_path) -> None:
    market = _market(periods=4)
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    targets = pd.DataFrame({"600000.SH": [0.10, 0.10]}, index=sessions[:2])
    result = simulate_ashare_target_weights(
        targets,
        market,
        AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
    )
    tampered = result.execution_trace.copy()
    idx = tampered.index[tampered["record_type"] == "session_mapping"][0]
    tampered.loc[idx, "status"] = "unmapped"
    tampered.loc[idx, "reason"] = "no_next_market_session"
    tampered.loc[idx, "execution_date"] = pd.NaT
    validation = validate_execution_trace(tampered)
    assert validation.ok is False
    assert any(reason.startswith("execution_trace_unmapped_signal") for reason in validation.reasons)


def test_empty_target_result_keeps_config_out_of_risk_events(tmp_path) -> None:
    result = simulate_ashare_target_weights(
        pd.DataFrame(),
        _market(periods=2),
        AShareExecutionSimulationConfig(audit_log_dir=str(tmp_path / "audit")),
    )
    assert result.risk_events == []
    assert result.config["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS
    assert result.config["execution_trace_ok"] is True
