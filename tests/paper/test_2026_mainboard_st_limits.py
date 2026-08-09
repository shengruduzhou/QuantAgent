from __future__ import annotations

import pytest

from quantagent.backtest import ashare_rules as rules
from quantagent.paper.broker import MarketSnapshot


@pytest.mark.parametrize("board", [rules.SH_MAIN, rules.SZ_MAIN])
def test_august_2026_paper_snapshot_uses_ten_percent_main_board_st_band(board: str) -> None:
    snapshot = MarketSnapshot(
        symbol="600000.SH" if board == rules.SH_MAIN else "000001.SZ",
        trade_date="2026-08-07",
        last_price=10.50,
        previous_close=10.00,
        session_volume=1_000_000,
        board=board,
        is_st=True,
    )

    limits = snapshot.limits()
    assert limits.ratio == pytest.approx(0.10)
    assert limits.limit_up == pytest.approx(11.00)
    assert limits.limit_down == pytest.approx(9.00)
    assert snapshot.at_limit_up is False


def test_pre_reform_paper_snapshot_preserves_five_percent_main_board_st_band() -> None:
    snapshot = MarketSnapshot(
        symbol="600000.SH",
        trade_date="2026-07-03",
        last_price=10.50,
        previous_close=10.00,
        session_volume=1_000_000,
        board=rules.SH_MAIN,
        is_st=True,
    )

    assert snapshot.limits().ratio == pytest.approx(0.05)
    assert snapshot.at_limit_up is True
