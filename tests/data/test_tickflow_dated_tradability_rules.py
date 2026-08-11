from __future__ import annotations

import pandas as pd

from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.tickflow_provider import TickflowProvider


def test_current_snapshot_main_board_st_uses_current_ten_percent_band(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "600000.SH"],
            "trade_date": pd.to_datetime(["2026-08-05", "2026-08-06", "2026-08-07"]),
            "open": [10.0, 10.0, 10.5],
            "high": [10.0, 10.5, 11.55],
            "low": [10.0, 10.0, 10.5],
            "close": [10.0, 10.5, 11.55],
            "volume": [1_000_000, 1_000_000, 1_000_000],
            "amount": [10_000_000, 10_500_000, 11_550_000],
        }
    )
    provider = TickflowProvider()
    monkeypatch.setattr(provider, "_call_tickflow_daily", lambda request: raw.copy())
    monkeypatch.setattr(
        provider,
        "_ensure_all_instruments",
        lambda: [{"symbol": "600000.SH", "name": "ST测试", "type": "stock"}],
    )

    result = provider._call_tickflow_current_snapshot_tradability(
        ProviderRequest(
            symbols=("600000.SH",),
            start_date="2026-08-05",
            end_date="2026-08-07",
        )
    )

    assert bool(result.loc[1, "is_st"]) is True
    assert bool(result.loc[1, "is_limit_up"]) is False
    assert bool(result.loc[2, "is_limit_up"]) is True


def test_current_snapshot_chinext_flags_are_not_flattened_to_main_board_band(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["300001.SZ", "300001.SZ"],
            "trade_date": pd.to_datetime(["2026-08-06", "2026-08-07"]),
            "open": [10.0, 10.0],
            "high": [10.0, 11.0],
            "low": [10.0, 10.0],
            "close": [10.0, 11.0],
            "volume": [1_000_000, 1_000_000],
            "amount": [10_000_000, 11_000_000],
        }
    )
    provider = TickflowProvider()
    monkeypatch.setattr(provider, "_call_tickflow_daily", lambda request: raw.copy())
    monkeypatch.setattr(
        provider,
        "_ensure_all_instruments",
        lambda: [{"symbol": "300001.SZ", "name": "测试股份", "type": "stock"}],
    )

    result = provider._call_tickflow_current_snapshot_tradability(
        ProviderRequest(
            symbols=("300001.SZ",),
            start_date="2026-08-06",
            end_date="2026-08-07",
        )
    )

    assert bool(result.loc[1, "is_limit_up"]) is False
