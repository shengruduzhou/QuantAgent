from __future__ import annotations

import pandas as pd

from quantagent.data.providers.akshare_calendar import load_akshare_research_calendar


class _FakeAkShare:
    __version__ = "test"

    def __init__(self, dates: list[object]) -> None:
        self._dates = dates

    def tool_trade_date_hist_sina(self) -> pd.DataFrame:
        return pd.DataFrame({"trade_date": self._dates})


def test_research_calendar_rejects_partially_invalid_upstream_response() -> None:
    evidence = load_akshare_research_calendar(
        allow_network=True,
        ak_module=_FakeAkShare(["2025-01-02", "not-a-date", "2025-01-03"]),
    )

    assert evidence.usable is False
    assert evidence.metadata["status"] == "invalid_sessions"
    assert evidence.metadata["invalid_session_count"] == 1
    assert evidence.warnings == ("akshare_calendar_invalid_sessions:1",)


def test_research_calendar_accepts_complete_monotonic_session_set() -> None:
    evidence = load_akshare_research_calendar(
        allow_network=True,
        ak_module=_FakeAkShare(["2025-01-02", "2025-01-03"]),
    )

    assert evidence.usable is True
    assert evidence.metadata["status"] == "passed"
    assert evidence.metadata["session_count"] == 2
