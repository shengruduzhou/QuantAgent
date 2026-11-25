"""Suspension representation for the traded-sessions-only silver panel (Cycle 32)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import ExecutionTimingViolation
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.data.ashare.session_gaps import (
    GAP_DELISTED,
    GAP_SUSPENDED,
    GAP_UNEXPLAINED,
    SessionGapError,
    augment_with_suspended_rows,
    classify_session_gaps,
    find_session_gaps,
    fold_suspension_snapshots,
    load_suspension_evidence,
)

SESSIONS = list(pd.bdate_range("2025-09-01", periods=8))
HELD = "600000.SH"
OTHER = "000001.SZ"


def _bar(symbol: str, day: pd.Timestamp, close: float) -> dict:
    return {"symbol": symbol, "trade_date": day, "open": close, "high": close, "low": close,
            "close": close, "volume": 1_000_000.0, "amount": close * 1_000_000.0,
            "available_at": day + pd.Timedelta(hours=15), "is_suspended": 0.0,
            "is_st": False, "is_limit_up": 0.0, "is_limit_down": 0.0}


def _panel(hole_days: tuple[int, ...] = (3, 4)) -> pd.DataFrame:
    rows = [_bar(OTHER, d, 10.0 + i) for i, d in enumerate(SESSIONS)]
    rows += [_bar(HELD, d, 20.0 + i) for i, d in enumerate(SESSIONS) if i not in hole_days]
    return pd.DataFrame(rows)


def _halt(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    return pd.DataFrame({"symbol": [symbol], "effective_start": [start],
                         "effective_end": [end], "suspension_reason": ["重大事项"]})


def test_gaps_come_from_the_calendar_not_the_panel_dates() -> None:
    panel = _panel()
    gaps = find_session_gaps(panel, SESSIONS)
    assert list(zip(gaps["symbol"], gaps["trade_date"])) == [
        (HELD, SESSIONS[3]), (HELD, SESSIONS[4])]
    assert set(gaps["gap_kind"]) == {"interior"}
    # a symbol that stops printing before the panel max has trailing gaps
    trailing = find_session_gaps(_panel(hole_days=(6, 7)), SESSIONS)
    assert list(trailing["gap_kind"]) == ["trailing", "trailing"]


def test_panel_date_off_calendar_is_refused() -> None:
    with pytest.raises(SessionGapError):
        find_session_gaps(_panel(), SESSIONS[:3] + SESSIONS[4:])


def test_classification_requires_interval_and_snapshot_coverage() -> None:
    gaps = find_session_gaps(_panel(), SESSIONS)
    halts = _halt(HELD, SESSIONS[3], SESSIONS[4])
    # snapshot only for the first gap day: the second stays unexplained even
    # though the interval spans it (no snapshot = no evidence)
    out = classify_session_gaps(gaps, suspension_intervals=halts, snapshot_dates=[SESSIONS[3]])
    assert list(out["classification"]) == [GAP_SUSPENDED, GAP_UNEXPLAINED]
    assert list(out["evidence"]) == ["eastmoney_suspension_snapshot",
                                     "outside_halt_snapshot_window"]
    none = classify_session_gaps(gaps, suspension_intervals=None, snapshot_dates=SESSIONS)
    assert set(none["classification"]) == {GAP_UNEXPLAINED}
    delisted = classify_session_gaps(gaps, suspension_intervals=None, snapshot_dates=SESSIONS,
                                     delisting_dates={HELD: SESSIONS[2]})
    assert set(delisted["classification"]) == {GAP_DELISTED}


def test_open_ended_halt_is_closed_at_last_snapshot() -> None:
    snaps = pd.DataFrame({
        "code": ["600000", "600000"], "name": ["x", "x"],
        "suspend_start": [SESSIONS[3], SESSIONS[3]], "suspend_end": [pd.NaT, pd.NaT],
        "suspension_reason": ["r", "r"], "snapshot_date": [SESSIONS[3], SESSIONS[4]],
    })
    iv = fold_suspension_snapshots(snaps, retrieved_at="2026-11-25T00:00:00+00:00")
    assert len(iv) == 1 and iv.loc[0, "symbol"] == HELD
    assert iv.loc[0, "effective_end"] == SESSIONS[4]
    for column in ("source", "source_endpoint", "retrieved_at", "available_at", "quality_status"):
        assert iv.loc[0, column]


def test_suspended_rows_are_untradeable_marks_at_the_carried_close() -> None:
    panel = _panel()
    out, classified = augment_with_suspended_rows(
        panel, sessions=SESSIONS, suspension_intervals=_halt(HELD, SESSIONS[3], SESSIONS[4]),
        snapshot_dates=SESSIONS)
    assert len(out) == len(panel) + 2
    rows = out[(out["symbol"] == HELD) & out["trade_date"].isin(SESSIONS[3:5])]
    assert (rows["close"] == 22.0).all()          # last traded raw close (session 2)
    assert (rows[["open", "high", "low"]].to_numpy() == 22.0).all()
    assert (rows["volume"] == 0).all() and (rows["amount"] == 0).all()
    assert (rows["is_suspended"] == 1).all()
    assert out.dtypes.equals(panel.dtypes)
    assert not out.duplicated(["symbol", "trade_date"]).any()


def test_unexplained_hole_is_not_filled() -> None:
    panel = _panel()
    out, classified = augment_with_suspended_rows(
        panel, sessions=SESSIONS, suspension_intervals=None, snapshot_dates=SESSIONS)
    assert len(out) == len(panel)
    assert set(classified["classification"]) == {GAP_UNEXPLAINED}


def _targets() -> pd.DataFrame:
    # hold HELD from session 0 onwards (signal-dated)
    return pd.DataFrame({HELD: [0.5] * 6, OTHER: [0.3] * 6}, index=SESSIONS[:6])


def test_held_name_with_omitted_suspension_row_fails_the_strict_simulator() -> None:
    """The FRESH books failure: the panel omits the halted session."""
    with pytest.raises(ExecutionTimingViolation, match="missing_execution_bar"):
        run_strict_backtest_v8(_targets(), _panel())


def test_held_name_through_an_evidenced_halt_is_valued_not_traded() -> None:
    out, _ = augment_with_suspended_rows(
        _panel(), sessions=SESSIONS, suspension_intervals=_halt(HELD, SESSIONS[3], SESSIONS[4]),
        snapshot_dates=SESSIONS)
    result = run_strict_backtest_v8(_targets(), out)
    nav = result.nav
    assert set(SESSIONS[3:5]) <= set(pd.to_datetime(nav.index))   # valued through the halt
    assert nav.notna().all()
    trades = result.trades
    on_halt = trades[(trades["symbol"].astype(str) == HELD)
                     & pd.to_datetime(trades["trade_date"]).isin(SESSIONS[3:5])]
    assert len(on_halt) and (on_halt["status"] == "rejected").all()
    assert set(on_halt["last_message"]) == {"suspended"}


def test_unexplained_hole_still_fails_closed_in_the_simulator() -> None:
    out, _ = augment_with_suspended_rows(
        _panel(), sessions=SESSIONS, suspension_intervals=None, snapshot_dates=SESSIONS)
    with pytest.raises(ExecutionTimingViolation, match="missing_execution_bar"):
        run_strict_backtest_v8(_targets(), out)


def test_load_evidence_counts_empty_snapshot_files_as_covered(tmp_path) -> None:
    staging = tmp_path / "pit" / "_suspension"
    staging.mkdir(parents=True)
    pd.DataFrame(columns=["code", "snapshot_date"]).to_parquet(staging / "20250904.parquet")
    _halt(HELD, SESSIONS[3], SESSIONS[4]).to_parquet(tmp_path / "pit" / "suspension_intervals.parquet")
    intervals, covered = load_suspension_evidence([tmp_path / "pit", tmp_path / "missing"])
    assert covered == [pd.Timestamp("2025-09-04")]
    assert len(intervals) == 1


def test_fresh_books_panel_gets_evidenced_suspension_rows(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "fresh_blind_daily.py"
    spec = importlib.util.spec_from_file_location("fresh_blind_daily_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"trade_date": SESSIONS, "is_open": True}).to_parquet(calendar)
    pit = tmp_path / "fresh"
    (pit / "_suspension").mkdir(parents=True)
    for day in SESSIONS:
        pd.DataFrame(columns=["code", "snapshot_date"]).to_parquet(
            pit / "_suspension" / f"{day:%Y%m%d}.parquet")
    _halt(HELD, SESSIONS[3], SESSIONS[3]).to_parquet(pit / "suspension_intervals.parquet")
    monkeypatch.setattr(module, "CALENDAR", calendar)
    monkeypatch.setattr(module, "SUSPENSION_DIRS", [pit])
    monkeypatch.setattr(module, "DELISTINGS", tmp_path / "absent.parquet")

    out, report = module.with_suspension_rows(_panel(), SESSIONS[0])
    assert len(out) == len(_panel()) + 1
    assert report["by_classification"] == {"SUSPENDED": 1, "MISSING_UNEXPLAINED": 1}
    assert report["unexplained"] == [f"{HELD}@{SESSIONS[4].date()}"]
