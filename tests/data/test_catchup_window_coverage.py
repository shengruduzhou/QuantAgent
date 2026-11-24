"""Cycle 32: the FRESH catch-up must not write a panel with an interior hole.

The vendor returns the latest ``count`` bars; with the old hard-coded ``count=40``
a 73-session window (2026-08-06..2026-11-24) would have staged only
2026-09-22..2026-11-24, and the supervisor's ``max(trade_date)`` check would have
reported the panel CURRENT.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catchup = _load("catchup_panel_chunked")
repair = _load("repair_fresh_window_20260704")


def _calendar(start: str, end: str) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"trade_date": days, "is_open": True})


def _rows(dates, symbols) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": s, "trade_date": d, "close": 10.0} for d in dates for s in symbols]
    )


SEEDS = [f"{i:06d}.SZ" for i in range(100)]


def test_full_window_passes():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-11-24")
    counts = catchup.require_window_coverage(_rows(sessions, SEEDS), sessions, len(SEEDS))
    assert len(counts) == len(sessions)


def test_latest_40_bars_hole_is_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-11-24")
    assert len(sessions) > 40
    latest_40 = sessions[-40:]  # what count=40 returns
    with pytest.raises(catchup.WindowCoverageError, match="sessions below"):
        catchup.require_window_coverage(_rows(latest_40, SEEDS), sessions, len(SEEDS))


def test_thin_session_is_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-08-20")
    rows = _rows(sessions, SEEDS)
    rows = rows[~((rows["trade_date"] == sessions[3]) & (rows["symbol"].isin(SEEDS[:10])))]
    with pytest.raises(catchup.WindowCoverageError):
        catchup.require_window_coverage(rows, sessions, len(SEEDS))
    # 92/100 < 93% is refused; 93/100 passes
    rows93 = _rows(sessions, SEEDS)
    rows93 = rows93[~((rows93["trade_date"] == sessions[3]) & (rows93["symbol"].isin(SEEDS[:7])))]
    catchup.require_window_coverage(rows93, sessions, len(SEEDS))


def test_off_calendar_rows_are_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-08-20")
    rows = pd.concat([_rows(sessions, SEEDS), _rows([pd.Timestamp("2026-08-08")], SEEDS)])  # a Saturday
    with pytest.raises(catchup.WindowCoverageError, match="not calendar sessions"):
        catchup.require_window_coverage(rows, sessions, len(SEEDS))


def test_expired_calendar_is_refused():
    with pytest.raises(catchup.WindowCoverageError, match="calendar ends"):
        catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-12-28", "2027-01-05")


def test_closed_days_are_not_sessions():
    calendar = _calendar("2026-09-28", "2026-10-12")
    calendar.loc[calendar["trade_date"].between("2026-10-01", "2026-10-08"), "is_open"] = False
    sessions = catchup.window_sessions(calendar, "2026-09-28", "2026-10-12")
    assert pd.Timestamp("2026-10-05") not in sessions
    assert pd.Timestamp("2026-10-09") in sessions


class _FakeKlines:
    def __init__(self):
        self.calls = []

    def get(self, sym, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame()


class _FakeTF:
    def __init__(self):
        self.klines = _FakeKlines()


def test_fetch_passes_requested_count():
    tf = _FakeTF()
    repair.fetch_with_retry(tf, "000001.SZ", count=83)
    assert tf.klines.calls[-1]["count"] == 83
    repair.fetch_with_retry(tf, "000001.SZ")
    assert tf.klines.calls[-1]["count"] == 40  # legacy default for other callers
    with pytest.raises(ValueError):
        repair.fetch_with_retry(tf, "000001.SZ", count=0)


def test_staging_lock_is_exclusive(tmp_path):
    first = catchup._acquire_staging_lock(tmp_path)
    assert first is not None
    assert catchup._acquire_staging_lock(tmp_path) is None
    first.close()
    again = catchup._acquire_staging_lock(tmp_path)
    assert again is not None
    again.close()
