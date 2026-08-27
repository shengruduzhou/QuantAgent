from __future__ import annotations

import pandas as pd
import pytest

from quantagent.cli.v7_train import _restrict_market_for_paper


def _market() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=7)
    return pd.DataFrame(
        {
            "trade_date": list(dates) * 2,
            "symbol": ["A"] * len(dates) + ["BENCH"] * len(dates),
            "close": [10.0] * len(dates) + [100.0] * len(dates),
        }
    )


def test_paper_market_keeps_intervening_and_final_execution_session() -> None:
    dates = pd.bdate_range("2026-01-05", periods=7)
    weights = pd.DataFrame({"A": [1.0, 0.0]}, index=[dates[0], dates[4]])

    result = _restrict_market_for_paper(_market(), weights, "BENCH")

    assert sorted(result["trade_date"].unique()) == list(dates[:6])
    assert set(result["symbol"]) == {"A", "BENCH"}


def test_paper_market_fails_without_session_after_last_signal() -> None:
    dates = pd.bdate_range("2026-01-05", periods=7)
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[-1]])

    with pytest.raises(ValueError, match="next session"):
        _restrict_market_for_paper(_market(), weights)
