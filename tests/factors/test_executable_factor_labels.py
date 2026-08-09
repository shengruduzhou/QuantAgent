from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.evaluation import forward_return_labels
from quantagent.factors.executable_labels import (
    FACTOR_LABEL_SEMANTICS,
    build_executable_forward_returns,
)
from quantagent.factors.lifecycle import build_factor_lifecycle_report


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


def test_executable_label_starts_at_next_session_not_signal_close() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2026-01-05", periods=4),
            "symbol": ["600000.SH"] * 4,
            "close": [10.0, 20.0, 30.0, 60.0],
        }
    )
    legacy = forward_return_labels(frame, horizons=(1,))
    executable = build_executable_forward_returns(frame, horizons=(1,)).frame

    assert legacy.loc[0, "forward_return_1d"] == pytest.approx(1.0)
    assert executable.loc[0, "forward_executable_return_1d"] == pytest.approx(0.5)
    assert executable.loc[0, "factor_label_entry_date"] == frame.loc[1, "trade_date"]
    assert executable.loc[0, "factor_label_end_1d"] == frame.loc[2, "trade_date"]


def test_executable_factor_label_semantics_matches_strict_execution() -> None:
    result = build_executable_forward_returns(_panel(), horizons=(1, 3))
    assert FACTOR_LABEL_SEMANTICS == EXECUTION_TIMING_SEMANTICS
    assert result.schema["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS
    assert result.schema["entry_delay_sessions"] == 1


def test_zero_delay_cannot_masquerade_as_executable_factor_label() -> None:
    with pytest.raises(ValueError, match="entry_delay_sessions"):
        build_executable_forward_returns(_panel(), horizons=(1,), entry_delay_sessions=0)


def test_lifecycle_decay_is_stamped_with_executable_semantics() -> None:
    frame = _panel(days=70, symbols=24)
    # Caller supplied target for the core diagnostic; decay itself is rebuilt
    # from close using the governed next-session entry contract.
    labeled = build_executable_forward_returns(frame, horizons=(1,)).frame
    report = build_factor_lifecycle_report(
        labeled,
        "factor",
        "forward_executable_return_1d",
    )
    assert report.label_semantics == EXECUTION_TIMING_SEMANTICS
    assert report.recommended_status != "active"
