from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.evaluation import forward_return_labels
from quantagent.factors.executable_labels import (
    FACTOR_LABEL_SCHEMA_VERSION,
    build_executable_forward_returns,
    canonical_market_sessions,
    executable_factor_decay_curve,
    market_session_schedule_sha256,
)
from quantagent.factors.lifecycle import (
    UNVERIFIED_RETURN_SEMANTICS,
    build_factor_lifecycle_report,
)


def _panel(days: int = 8, symbols: int = 24) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2026-01-05", periods=days)
    for day_idx, date in enumerate(dates):
        for symbol_idx in range(symbols):
            close = 10.0 + day_idx * (1.0 + symbol_idx / 100.0)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"S{symbol_idx:03d}",
                    "close": close,
                    "amount": 20_000_000.0,
                    "factor": float(symbol_idx),
                }
            )
    return pd.DataFrame(rows)


def _sessions(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]).drop_duplicates().sort_values())


def test_executable_label_starts_at_next_global_session_not_signal_close() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2026-01-05", periods=4),
            "symbol": ["600000.SH"] * 4,
            "close": [10.0, 20.0, 30.0, 60.0],
        }
    )
    sessions = _sessions(frame)
    legacy = forward_return_labels(frame, horizons=(1,))
    executable = build_executable_forward_returns(
        frame,
        horizons=(1,),
        market_sessions=sessions,
    ).frame

    assert legacy.loc[0, "forward_return_1d"] == pytest.approx(1.0)
    assert executable.loc[0, "forward_executable_return_1d"] == pytest.approx(0.5)
    assert executable.loc[0, "factor_label_entry_date"] == frame.loc[1, "trade_date"]
    assert executable.loc[0, "factor_label_end_1d"] == frame.loc[2, "trade_date"]
    assert bool(executable.loc[0, "factor_label_entry_observed"]) is True


def test_executable_factor_label_schema_binds_strict_execution_and_calendar() -> None:
    frame = _panel()
    sessions = _sessions(frame)
    result = build_executable_forward_returns(
        frame,
        horizons=(1, 3),
        market_sessions=sessions,
    )
    assert result.schema["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS
    assert result.schema["schema_version"] == FACTOR_LABEL_SCHEMA_VERSION
    assert result.schema["entry_delay_sessions"] == 1
    assert result.schema["market_session_schedule_sha256"] == market_session_schedule_sha256(sessions)


def test_governed_labels_require_explicit_market_calendar() -> None:
    with pytest.raises(ValueError, match="explicit market_sessions"):
        build_executable_forward_returns(_panel(), horizons=(1,))


def test_zero_delay_cannot_masquerade_as_executable_factor_label() -> None:
    frame = _panel()
    with pytest.raises(ValueError, match="entry_delay_sessions"):
        build_executable_forward_returns(
            frame,
            horizons=(1,),
            entry_delay_sessions=0,
            market_sessions=_sessions(frame),
        )


def test_missing_symbol_bar_on_global_t_plus_one_does_not_shift_entry() -> None:
    sessions = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    frame = pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "symbol": "A.SZ", "close": 10.0},
            {"trade_date": "2026-01-07", "symbol": "A.SZ", "close": 11.0},
            {"trade_date": "2026-01-08", "symbol": "A.SZ", "close": 12.0},
            {"trade_date": "2026-01-05", "symbol": "B.SZ", "close": 20.0},
            {"trade_date": "2026-01-06", "symbol": "B.SZ", "close": 21.0},
            {"trade_date": "2026-01-07", "symbol": "B.SZ", "close": 22.0},
            {"trade_date": "2026-01-08", "symbol": "B.SZ", "close": 23.0},
        ]
    )
    built = build_executable_forward_returns(frame, horizons=(1,), market_sessions=sessions).frame
    a_t = built[(built["symbol"] == "A.SZ") & (built["trade_date"] == pd.Timestamp("2026-01-05"))].iloc[0]
    b_t = built[(built["symbol"] == "B.SZ") & (built["trade_date"] == pd.Timestamp("2026-01-05"))].iloc[0]

    assert a_t["factor_label_entry_date"] == pd.Timestamp("2026-01-06")
    assert bool(a_t["factor_label_entry_observed"]) is False
    assert pd.isna(a_t["forward_executable_return_1d"])
    assert b_t["factor_label_end_1d"] == pd.Timestamp("2026-01-07")
    assert b_t["forward_executable_return_1d"] == pytest.approx(22.0 / 21.0 - 1.0)


def test_missing_symbol_bar_on_exact_exit_does_not_stretch_horizon() -> None:
    sessions = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    frame = pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "symbol": "A.SZ", "close": 10.0},
            {"trade_date": "2026-01-06", "symbol": "A.SZ", "close": 10.5},
            {"trade_date": "2026-01-08", "symbol": "A.SZ", "close": 12.0},
        ]
    )
    built = build_executable_forward_returns(frame, horizons=(1,), market_sessions=sessions).frame
    row = built.iloc[0]
    assert row["factor_label_end_1d"] == pd.Timestamp("2026-01-07")
    assert bool(row["factor_label_end_observed_1d"]) is False
    assert pd.isna(row["forward_executable_return_1d"])


def test_calendar_validation_rejects_unsorted_and_duplicate_sessions() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        canonical_market_sessions(["2026-01-06", "2026-01-05"])
    with pytest.raises(ValueError, match="duplicate"):
        canonical_market_sessions(["2026-01-05", "2026-01-05"])


def test_factor_date_outside_explicit_calendar_is_rejected() -> None:
    frame = _panel()
    with pytest.raises(ValueError, match="absent from market_sessions"):
        build_executable_forward_returns(
            frame,
            horizons=(1,),
            market_sessions=_sessions(frame)[1:],
        )


def test_decay_curve_uses_same_explicit_calendar() -> None:
    frame = _panel()
    decay = executable_factor_decay_curve(
        frame,
        "factor",
        horizons=(1, 3),
        market_sessions=_sessions(frame),
    )
    assert tuple(decay.horizon_days) == (1, 3)
    assert set(decay.rank_ic.index) == {1, 3}


def test_lifecycle_decay_is_bound_to_executable_calendar() -> None:
    frame = _panel(days=70, symbols=24)
    sessions = _sessions(frame)
    labeled = build_executable_forward_returns(
        frame,
        horizons=(1,),
        market_sessions=sessions,
    ).frame
    report = build_factor_lifecycle_report(
        labeled,
        "factor",
        "forward_executable_return_1d",
        market_sessions=sessions,
    )
    assert report.label_semantics == EXECUTION_TIMING_SEMANTICS
    assert report.market_session_schedule_sha256 == market_session_schedule_sha256(sessions)
    assert report.recommended_status != "active"


def test_lifecycle_with_prices_refuses_missing_calendar() -> None:
    frame = _panel(days=70, symbols=24)
    labeled = build_executable_forward_returns(
        frame,
        horizons=(1,),
        market_sessions=_sessions(frame),
    ).frame
    with pytest.raises(ValueError, match="explicit market_sessions"):
        build_factor_lifecycle_report(
            labeled,
            "factor",
            "forward_executable_return_1d",
        )


def test_lifecycle_without_prices_marks_return_semantics_unverified() -> None:
    frame = _panel(days=70, symbols=24).drop(columns=["close"])
    frame["caller_return"] = np.linspace(0.0, 0.01, len(frame))
    report = build_factor_lifecycle_report(frame, "factor", "caller_return")
    assert report.label_semantics == UNVERIFIED_RETURN_SEMANTICS
    assert report.market_session_schedule_sha256 is None
