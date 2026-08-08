from __future__ import annotations

import pandas as pd

from quantagent.data.fundamental import (
    FundamentalRankerBuilder,
    FundamentalRankerConfig,
    build_fundamental_ranker,
    pit_safe_sector_map,
)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": f"60000{i}.SH",
                "available_at": pd.Timestamp("2024-01-15"),
                "pe_ttm": 5.0 + i,
                "pb": 0.8 + 0.1 * i,
                "ps_ttm": 1.0 + 0.1 * i,
                "roe": 0.15 - i * 0.005,
                "gross_margin": 0.35 - i * 0.005,
                "operating_cf_to_net_income": 1.1 - i * 0.02,
                "revenue_yoy": 0.12 - i * 0.004,
                "net_income_yoy": 0.10 - i * 0.004,
            }
            for i in range(8)
        ]
    )


def _current_snapshot_sector() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": f"60000{i}.SH",
                "sector_level_1": "CurrentSector",
                "coverage_status": "current_snapshot",
            }
            for i in range(8)
        ]
    )


def test_current_snapshot_without_available_at_is_not_pit_eligible() -> None:
    assert pit_safe_sector_map(_current_snapshot_sector()) is None


def test_historical_ranker_falls_back_to_board_proxy_for_current_snapshot() -> None:
    result = build_fundamental_ranker(
        _metrics(),
        as_of_dates=["2024-02-01"],
        sector_map=_current_snapshot_sector(),
    )
    assert not result.frame.empty
    assert set(result.frame["rank_bucket_kind"]) == {"board_proxy"}
    assert "CurrentSector" not in set(result.frame["rank_bucket"])


def test_current_snapshot_cannot_open_fundamental_overlay_gate() -> None:
    builder = FundamentalRankerBuilder(FundamentalRankerConfig())
    result = builder.build(
        _metrics(),
        as_of_dates=["2024-02-01"],
        sector_map=_current_snapshot_sector(),
    )
    assert result.coverage["real_sector_share"] == 0.0
    gate = result.coverage["gate"]
    assert gate["fundamental_ranker_usable_for_overlay"] is False
    assert "real_sector_share_below_threshold" in gate["reason"]


def test_dated_sector_map_remains_eligible() -> None:
    sector = _current_snapshot_sector().drop(columns=["coverage_status"])
    sector["available_at"] = pd.Timestamp("2024-01-01")
    result = build_fundamental_ranker(
        _metrics(),
        as_of_dates=["2024-02-01"],
        sector_map=sector,
    )
    assert set(result.frame["rank_bucket_kind"]) == {"sector_level_1"}
    assert set(result.frame["rank_bucket"]) == {"CurrentSector"}
