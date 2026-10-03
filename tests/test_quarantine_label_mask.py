"""Label windows, not just signal dates, must avoid quarantined holdouts (round-29 R3-F02)."""

import pandas as pd

from quantagent.backtest.quarantine import QuarantineWindow, _BUILTIN_WINDOWS, clean_label_mask


def _window(start, end):
    return QuarantineWindow(pd.Timestamp(start), pd.Timestamp(end), "test", "test")


def test_clean_signal_date_with_label_reaching_into_holdout_is_dropped():
    sessions = pd.bdate_range("2025-08-20", "2025-09-10")
    dates = pd.Series(sessions)
    mask = clean_label_mask(
        dates, horizon_sessions=5, windows=[_window("2025-09-01", "2026-05-18")]
    )
    kept = dates[mask]
    # 2025-08-21: entry 08-22, exit 6 sessions later = 08-29 (clean);
    # 2025-08-22: exit = 09-01, inside the holdout.
    assert kept.max() == pd.Timestamp("2025-08-21")
    assert not mask[dates >= pd.Timestamp("2025-09-01")].any()


def test_rows_beyond_the_known_calendar_are_not_clean():
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=10))
    mask = clean_label_mask(dates, horizon_sessions=5, windows=[])
    assert mask.sum() == 4  # only rows with 6 later sessions available


def test_builtin_fallback_protects_the_frozen_fresh_window_too():
    starts = {w["start"] for w in _BUILTIN_WINDOWS}
    assert {"2025-09-01", "2026-05-19"} <= starts


def test_a_suspension_gap_pushes_the_label_exit_into_the_holdout():
    """Gold labels exit on the symbol's own (h+1)-th next row: a gap makes the
    real exit later than the calendar exit (round-29 ENG_FUSION / R10)."""
    sessions = pd.bdate_range("2025-08-18", "2025-09-12")
    other = pd.DataFrame({"trade_date": sessions, "symbol": "B"})
    gapped = pd.DataFrame({"trade_date": [d for d in sessions if not ("2025-08-22" <= str(d.date()) <= "2025-08-27")],
                           "symbol": "A"})
    frame = pd.concat([other, gapped], ignore_index=True)
    window = [_window("2025-09-01", "2026-05-18")]
    calendar_only = clean_label_mask(frame["trade_date"], horizon_sessions=2, windows=window)
    per_symbol = clean_label_mask(frame["trade_date"], horizon_sessions=2, windows=window,
                                  symbols=frame["symbol"])
    row = (frame["symbol"] == "A") & (frame["trade_date"] == pd.Timestamp("2025-08-21"))
    # calendar exit = 08-26 (clean); A's own 3rd next row is 09-01 (holdout)
    assert bool(calendar_only[row].iloc[0]) is True
    assert bool(per_symbol[row].iloc[0]) is False
    assert per_symbol[frame["symbol"] == "B"].equals(calendar_only[frame["symbol"] == "B"])
