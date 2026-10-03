"""R10-F01: vendor zero-volume flat bars were classified TRADED / not suspended.

TickFlow prints a flat bar at the last close (volume = amount = 0) on sessions a
stock did not trade. The r29 execution panel called 25,313 of them TRADED, the
dataset kept 20,851 rows whose t+1 entry bar had zero volume, and on an
ex-rights date inside such a stretch the stale pre-ex close was multiplied by
the new factor (000836.SZ 10转20: hfq x3.0) while bonus shares were credited
at the stale price -- NAV inflated until trading resumed.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)
from quantagent.data.ashare import contracts, gold_bridge
from quantagent.data.ashare import execution_panel as ep

SESSIONS = pd.bdate_range("2016-02-22", periods=8)
SYMBOL = "000836.SZ"


def _raw(closes, volumes):
    return pd.DataFrame({
        "symbol": SYMBOL, "trade_date": SESSIONS[: len(closes)],
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": volumes, "amount": [c * v for c, v in zip(closes, volumes)],
    })


def _master():
    return pd.DataFrame([{"symbol": SYMBOL, "board": "SZ_Main",
                          "listing_date": pd.Timestamp("2000-01-04"),
                          "delisting_date": pd.NaT}])


def _factors():
    # 10转20 effective on SESSIONS[3]: factor x3.
    return pd.DataFrame([
        {"symbol": SYMBOL, "effective_date": pd.Timestamp("2000-01-04"), "hfq_factor": 1.0},
        {"symbol": SYMBOL, "effective_date": SESSIONS[3], "hfq_factor": 3.0},
    ])


def _ca():
    return pd.DataFrame([{"symbol": SYMBOL, "ex_date": SESSIONS[3],
                          "cash_dividend_per_share": 0.0, "stock_dividend_ratio": 0.0,
                          "rights_ratio": 2.0}])


# Trades at 30 for three sessions, then a halt printed as flat zero-volume bars at
# the stale 30.00 across the ex-date, then resumes at the ex-rights level 10.10.
CLOSES = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 10.1, 10.2]
VOLUMES = [1e6, 1e6, 1e6, 0.0, 0.0, 0.0, 1e6, 1e6]


def _masked():
    panel = _raw(CLOSES, VOLUMES)
    panel["sessions_since_listing"] = 1000 + np.arange(len(panel))
    adjusted = gold_bridge.apply_adjustment(panel, _factors(), method=contracts.ADJUST_HFQ)
    return gold_bridge.build_masks(
        adjusted, master=_master(), st=pd.DataFrame(
            columns=["symbol", "effective_start", "effective_end"]), st_available={"SZ"})


class TestGoldBridge:
    def test_zero_volume_bars_are_marked_no_trade(self):
        masked = _masked().sort_values("trade_date")
        assert masked["mask_no_trade"].tolist() == [
            "FALSE", "FALSE", "FALSE", "TRUE", "TRUE", "TRUE", "FALSE", "FALSE"]

    def test_stale_close_is_not_inflated_by_the_new_factor(self):
        masked = _masked().sort_values("trade_date")
        adjusted_close = masked["close"].to_numpy()
        # 30 x 1.0 before; the halt carries 30 on the hfq scale instead of 30 x 3.
        assert adjusted_close[3:6] == pytest.approx([30.0, 30.0, 30.0])
        assert adjusted_close[6] == pytest.approx(10.1 * 3.0)

    def test_entry_on_a_zero_volume_bar_is_dropped(self):
        labelled, dropped = gold_bridge.build_labels(_masked(), horizons=[1])
        assert dropped["entry_zero_volume"] == 3      # t = SESSIONS[2..4]
        assert dropped["no_trade_at_t"] == 3          # t = SESSIONS[3..5]
        kept = set(labelled["trade_date"])
        assert SESSIONS[2] not in kept and SESSIONS[3] not in kept


class TestExecutionPanel:
    def _panel(self):
        bars = _raw(CLOSES, VOLUMES)
        bars["available_at"] = bars["trade_date"] + pd.Timedelta(hours=15)
        bars["serving_provider"] = "tickflow"
        for column in ("mask_is_suspended", "mask_is_st", "mask_limit_up", "mask_limit_down"):
            bars[column] = "FALSE"
        return ep.build_execution_panel(bars, session_gaps=None, factors=_factors(),
                                        corporate_actions=_ca())

    def test_zero_volume_bar_is_classified_and_untradeable(self):
        panel, stats = self._panel()
        halted = panel[panel["trade_date"].isin(SESSIONS[3:6])]
        assert (halted["gap_classification"] == ep.GAP_NO_TRADE).all()
        assert halted["is_suspended"].all()
        assert stats["rows_by_gap_classification"][ep.GAP_NO_TRADE] == 3

    def test_ex_date_inside_the_halt_is_marked_at_the_ex_rights_reference(self):
        panel, _ = self._panel()
        row = panel[panel["trade_date"] == SESSIONS[3]].iloc[0]
        assert row["close"] == pytest.approx(10.0)       # 30 / 3, not the stale 30
        assert row["ca_share_ratio"] == 2.0

    def test_bonus_credit_does_not_inflate_nav_during_the_halt(self):
        panel, _ = self._panel()
        weights = pd.DataFrame({SYMBOL: [0.5, 0.0]}, index=SESSIONS[[0, 6]])
        with tempfile.TemporaryDirectory() as folder:
            result = simulate_ashare_target_weights(
                weights, panel, AShareExecutionSimulationConfig(
                    slippage_bps=0, audit_log_dir=folder))
        nav = result.nav
        assert nav.loc[SESSIONS[3]] == pytest.approx(nav.loc[SESSIONS[2]], rel=1e-9)
        assert nav.loc[SESSIONS[5]] == pytest.approx(nav.loc[SESSIONS[2]], rel=1e-9)

    def test_verification_refuses_a_traded_zero_volume_row(self):
        panel, _ = self._panel()
        panel.loc[panel["trade_date"] == SESSIONS[4], "gap_classification"] = ep.GAP_TRADED
        with pytest.raises(ep.ExecutionPanelError, match="zero volume"):
            ep.verify_execution_panel(panel)
