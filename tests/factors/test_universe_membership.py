from __future__ import annotations

import pandas as pd
import pytest

from quantagent.factors.universe_membership import (
    UniverseMembershipError,
    filter_market_by_membership,
    load_universe_membership,
    symbols_for_window,
)


def _market() -> pd.DataFrame:
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        [
            {"trade_date": day, "symbol": symbol, "close": 10.0, "amount": 1_000_000.0}
            for day in dates
            for symbol in ("OLD.SH", "NEW.SH")
        ]
    )


def test_entry_and_delisting_intervals_change_historical_cross_section(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "symbol": ["OLD.SH", "NEW.SH"],
            "effective_from": ["2020-01-01", "2024-01-02"],
            "effective_to": ["2023-12-29", None],
            "universe_id": ["research_all", "research_all"],
            "source": ["test_master", "test_master"],
            "source_version": ["v1", "v1"],
            "available_at": ["2019-12-31T18:00:00+08:00", "2024-01-01T18:00:00+08:00"],
            "membership_reason": ["listed_until_delisting", "listing_effective"],
        }
    ).to_csv(path, index=False)

    evidence = load_universe_membership(path)
    filtered = filter_market_by_membership(_market(), evidence)

    assert set(filtered.loc[filtered["trade_date"] == pd.Timestamp("2023-12-29"), "symbol"]) == {"OLD.SH"}
    assert set(filtered.loc[filtered["trade_date"] == pd.Timestamp("2024-01-02"), "symbol"]) == {"NEW.SH"}
    assert "OLD.SH" in symbols_for_window(
        evidence, start_date="2023-01-01", end_date="2023-12-31"
    )
    assert "NEW.SH" not in symbols_for_window(
        evidence, start_date="2023-01-01", end_date="2023-12-31"
    )


def test_membership_available_after_effective_preopen_is_rejected(tmp_path) -> None:
    path = tmp_path / "future_membership.csv"
    pd.DataFrame(
        {
            "symbol": ["NEW.SH"],
            "effective_from": ["2024-01-02"],
            "effective_to": [None],
            "universe_id": ["research_all"],
            "source": ["test_master"],
            "source_version": ["v1"],
            "available_at": ["2024-01-02T10:00:00+08:00"],
            "membership_reason": ["late_announcement"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(UniverseMembershipError, match="09:25"):
        load_universe_membership(path)


def test_overlapping_intervals_are_rejected(tmp_path) -> None:
    path = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "symbol": ["A.SH", "A.SH"],
            "effective_from": ["2024-01-01", "2024-01-10"],
            "effective_to": ["2024-01-15", "2024-01-20"],
            "universe_id": ["u", "u"],
            "source": ["test", "test"],
            "source_version": ["v1", "v1"],
            "available_at": ["2023-12-31T18:00:00+08:00", "2024-01-09T18:00:00+08:00"],
            "membership_reason": ["member", "member"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(UniverseMembershipError, match="overlapping"):
        load_universe_membership(path)


def test_row_digest_mismatch_blocks_rehashed_current_member_file(tmp_path) -> None:
    path = tmp_path / "digest.csv"
    pd.DataFrame(
        {
            "symbol": ["A.SH"],
            "effective_from": ["2024-01-01"],
            "effective_to": [None],
            "universe_id": ["u"],
            "source": ["test"],
            "source_version": ["v1"],
            "available_at": ["2023-12-31T18:00:00+08:00"],
            "membership_reason": ["member"],
            "row_sha256": ["0" * 64],
        }
    ).to_csv(path, index=False)

    with pytest.raises(UniverseMembershipError, match="row_sha256"):
        load_universe_membership(path)
