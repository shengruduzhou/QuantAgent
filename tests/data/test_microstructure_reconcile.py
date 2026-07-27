"""Tick-to-daily reconciliation, including the aggregation blind spot.

The behaviour under test is the one that decides whether a tick feed may be
trusted for a day: does the feed rebuild the exchange bar, and when it does not,
is the shortfall the kind aggregation provably causes or a real disagreement?
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure import reconcile


def _events(prices, volumes, times, *, symbol="600000.SH",
            data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE) -> pd.DataFrame:
    rows = len(prices)
    return pd.DataFrame({
        "symbol": [symbol] * rows,
        "trade_date": ["2026-07-24"] * rows,
        "exchange_time": pd.to_datetime([f"2026-07-24 {t}" for t in times]),
        "ingest_sequence": list(range(rows)),
        "event_time_ns": [1_700_000_000_000_000_000 + i for i in range(rows)],
        "price": prices,
        "volume_shares": [float(v) for v in volumes],
        "amount_cny": [p * v for p, v in zip(prices, volumes)],
        "data_class": [data_class] * rows,
    })


def _panel(open_, high, low, close, volume, amount, symbol="600000.SH") -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": symbol, "trade_date": "2026-07-24", "open": open_, "high": high,
        "low": low, "close": close, "volume": volume, "amount": amount,
    }])


class TestDailyBarFromTrades:
    def test_ohlcv_aggregation(self):
        events = _events([10.0, 10.5, 9.8, 10.2], [100, 200, 300, 400],
                         ["09:30:00", "10:00:00", "11:00:00", "14:30:00"])
        bar = reconcile.daily_bar_from_trades(events)
        assert bar["open"] == 10.0
        assert bar["high"] == 10.5
        assert bar["low"] == 9.8
        assert bar["close"] == 10.2
        assert bar["volume"] == 1000.0

    def test_post_close_prints_are_excluded_from_the_bar(self):
        """Measured behaviour: post-close prints are absent from the daily bar."""
        events = _events([10.0, 10.2, 99.0], [100, 200, 500],
                         ["09:30:00", "14:30:00", "15:10:00"])
        bar = reconcile.daily_bar_from_trades(events)
        assert bar["close"] == 10.2
        assert bar["volume"] == 300.0
        assert bar["excluded_outside_session"] == 1
        assert bar["excluded_phase_counts"][mc.PHASE_POST_CLOSE] == 1

    def test_closing_auction_print_is_included(self):
        events = _events([10.0, 10.4], [100, 900], ["09:30:00", "15:00:03"])
        bar = reconcile.daily_bar_from_trades(events)
        assert bar["close"] == 10.4
        assert bar["volume"] == 1000.0

    def test_star_after_hours_excluded_for_star_symbols(self):
        events = _events([10.0, 10.4], [100, 900],
                         ["09:30:00", "15:10:00"], symbol="688008.SH")
        bar = reconcile.daily_bar_from_trades(events)
        assert bar["volume"] == 100.0
        assert bar["excluded_phase_counts"][mc.PHASE_AFTER_HOURS] == 1


class TestReconciliation:
    def test_exact_match(self):
        events = _events([10.0, 10.5, 9.8, 10.2], [100, 200, 300, 400],
                         ["09:30:00", "10:00:00", "11:00:00", "14:30:00"])
        panel = _panel(10.0, 10.5, 9.8, 10.2, 1000.0, 10130.0)
        report = reconcile.reconcile_days(events, panel)
        assert report["status_counts"] == {reconcile.MATCH: 1}
        assert report["match_rate_over_verifiable"] == 1.0

    def test_volume_disagreement_is_a_mismatch(self):
        events = _events([10.0, 10.2], [100, 200], ["09:30:00", "14:30:00"])
        panel = _panel(10.0, 10.2, 10.0, 10.2, 50_000.0, 510_000.0)
        report = reconcile.reconcile_days(events, panel)
        assert report["status_counts"] == {reconcile.MISMATCH: 1}
        assert "volume" in report["results"][0]["mismatched_fields"]

    def test_aggregation_narrows_the_range_and_is_recognised(self):
        """Derived range strictly inside the panel range = aggregation blind spot."""
        events = _events([10.0, 10.4, 10.2], [100, 200, 300],
                         ["09:30:00", "10:00:00", "14:30:00"])
        # Panel saw a wider range than any 3s bucket close revealed.
        panel = _panel(10.0, 10.6, 9.9, 10.2, 600.0, 6120.0)
        report = reconcile.reconcile_days(events, panel)
        result = report["results"][0]
        assert result["status"] == reconcile.MATCH_WITHIN_AGGREGATION_LIMITS
        assert sorted(result["mismatched_fields"]) == ["high", "low"]
        assert "aggregation_note" in result["detail"]
        # It still counts as verified for the match rate.
        assert report["match_rate_over_verifiable"] == 1.0

    def test_derived_high_above_panel_high_is_a_real_error(self):
        """Aggregation can only lose extremes, never invent them."""
        events = _events([10.0, 12.0], [100, 200], ["09:30:00", "14:30:00"])
        panel = _panel(10.0, 10.5, 10.0, 12.0, 300.0, 3400.0)
        report = reconcile.reconcile_days(events, panel)
        assert report["results"][0]["status"] == reconcile.MISMATCH

    def test_true_tick_class_gets_no_aggregation_allowance(self):
        events = _events([10.0, 10.4, 10.2], [100, 200, 300],
                         ["09:30:00", "10:00:00", "14:30:00"],
                         data_class=mc.EXCHANGE_TRADE_EVENT)
        panel = _panel(10.0, 10.6, 9.9, 10.2, 600.0, 6120.0)
        report = reconcile.reconcile_days(events, panel)
        assert report["results"][0]["status"] == reconcile.MISMATCH

    def test_missing_panel_row_is_unverified_not_passed(self):
        events = _events([10.0], [100], ["09:30:00"])
        report = reconcile.reconcile_days(events, pd.DataFrame(
            columns=["symbol", "trade_date", "open", "high", "low", "close",
                     "volume", "amount"]
        ))
        assert report["status_counts"] == {reconcile.NO_PANEL_ROW: 1}
        assert report["match_rate_over_verifiable"] is None
        assert report["unverifiable_symbol_days"] == 1

    def test_report_records_cohort_breadth(self):
        """A reconciliation must not let one symbol stand in for coverage."""
        events = pd.concat([
            _events([10.0, 10.2], [100, 200], ["09:30:00", "14:30:00"]),
            _events([20.0, 20.4], [300, 400], ["09:30:00", "14:30:00"],
                    symbol="000001.SZ"),
        ], ignore_index=True)
        panel = pd.concat([
            _panel(10.0, 10.2, 10.0, 10.2, 300.0, 3040.0),
            _panel(20.0, 20.4, 20.0, 20.4, 700.0, 14160.0, symbol="000001.SZ"),
        ], ignore_index=True)
        report = reconcile.reconcile_days(events, panel)
        assert report["symbol_days"] == 2
        assert report["distinct_symbols"] == 2
