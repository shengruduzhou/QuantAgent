from __future__ import annotations

import pytest

from quantagent.data.integrity import DailyOHLCVIntegrityPolicy, expected_daily_keys
from quantagent.data.providers.base import ProviderRequest


def test_multi_symbol_global_expected_dates_are_rejected_as_ambiguous() -> None:
    request = ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH", "000001.SZ"),
    )
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06")
    )
    with pytest.raises(ValueError, match="multi-symbol coverage requires explicit expected_symbol_trade_dates"):
        expected_daily_keys(request, policy)


def test_explicit_per_symbol_keys_can_encode_suspension_filtered_expectations() -> None:
    request = ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH", "000001.SZ"),
    )
    # 000001.SZ is intentionally not expected on Jan-06 (for example after an
    # upstream PIT tradability/suspension filter). The integrity layer must not
    # re-invent the missing cross-product key.
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_symbol_trade_dates=(
            ("600000.SH", "2026-01-05"),
            ("600000.SH", "2026-01-06"),
            ("000001.SZ", "2026-01-05"),
        )
    )
    assert expected_daily_keys(request, policy) == {
        ("600000.SH", "2026-01-05"),
        ("600000.SH", "2026-01-06"),
        ("000001.SZ", "2026-01-05"),
    }


def test_per_symbol_expected_key_must_reference_requested_symbol_and_window() -> None:
    request = ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH",),
    )
    outside_symbol = DailyOHLCVIntegrityPolicy.production(
        expected_symbol_trade_dates=(("000001.SZ", "2026-01-05"),)
    )
    with pytest.raises(ValueError, match="outside the provider request"):
        expected_daily_keys(request, outside_symbol)

    outside_window = DailyOHLCVIntegrityPolicy.production(
        expected_symbol_trade_dates=(("600000.SH", "2026-01-07"),)
    )
    with pytest.raises(ValueError, match="inside the provider request window"):
        expected_daily_keys(request, outside_window)
