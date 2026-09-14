"""Daily holding valuation without extra orders or expanded evaluation windows."""
import tempfile
import unittest
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    ExecutionTimingViolation,
    simulate_ashare_target_weights,
)
from quantagent.backtest.trace_proven_strict_v8 import run_trace_proven_strict_backtest_v8

SESSIONS = pd.bdate_range('2024-01-02', periods=5)
SYMBOL = '600000.SH'


def make_panel(prices):
    return pd.DataFrame([
        dict(trade_date=date, symbol=SYMBOL, close=price, volume=1e8,
             amount=price * 1e8, is_suspended=False, is_st=False,
             is_limit_up=False, is_limit_down=False)
        for date, price in zip(SESSIONS, prices)
    ])


def run_case(weights, prices):
    with tempfile.TemporaryDirectory(prefix='daily_nav_') as audit_dir:
        return run_trace_proven_strict_backtest_v8(
            weights, make_panel(prices),
            config=AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=audit_dir),
        )


class DailyMarkRegression(unittest.TestCase):
    def test_intermediate_drawdown_preserves_fill_endpoints(self):
        weights = pd.DataFrame({SYMBOL: [0.5, 0.0]}, index=SESSIONS[[0, 3]])
        flat = run_case(weights, [10., 10., 10., 10., 10.])
        dip = run_case(weights, [10., 10., 9., 10., 10.])
        columns = ['trade_date', 'symbol', 'side', 'filled_quantity', 'avg_price', 'total_cost']
        pd.testing.assert_frame_equal(flat.trades[columns], dip.trades[columns])
        self.assertEqual(len(dip.trades), 2, 'Valuation must not create orders')
        buy = dip.trades.loc[dip.trades.side.eq('buy')].iloc[0]
        expected_trough = 1_000_000. - float(buy.total_cost) - float(buy.filled_quantity)
        self.assertIn(SESSIONS[2], dip.nav.index, 'Held-session NAV is missing')
        self.assertAlmostEqual(float(dip.nav.loc[SESSIONS[2]]), expected_trough, places=7)
        self.assertAlmostEqual(dip.metrics.max_drawdown, 1 - expected_trough / 1_000_000., places=10)
        self.assertAlmostEqual(float(flat.nav.iloc[-1]), float(dip.nav.iloc[-1]), places=7)

    def test_window_does_not_expand_beyond_last_execution(self):
        weights = pd.DataFrame({SYMBOL: [0.5]}, index=SESSIONS[[0]])
        result = run_case(weights, [10., 10., 9., 8., 7.])
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.nav.index[-1], SESSIONS[1])

    def test_missing_held_bar_blocks_trace_on_non_rebalance_day(self):
        weights = pd.DataFrame({SYMBOL: [0.5, 0.]}, index=SESSIONS[[0, 3]])
        panel = make_panel([10.] * 5)
        extra = panel.assign(symbol="000001.SZ")
        panel = pd.concat([panel[panel.trade_date.ne(SESSIONS[2])], extra])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ExecutionTimingViolation, "missing_execution_bar"):
                simulate_ashare_target_weights(weights, panel,
                    AShareExecutionSimulationConfig(audit_log_dir=folder))
