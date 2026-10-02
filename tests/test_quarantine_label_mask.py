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
