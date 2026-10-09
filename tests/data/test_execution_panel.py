"""Certified raw execution panel (R1-F01/F02/F06).

The strict simulator treats a missing bar for a held name as fatal, the legacy
evaluator panel was qfq-levelled and vendor-stitched, and the certified gold had
no rows for suspended sessions. These tests pin the execution-panel contract:
explicit, classified gap rows that value but never trade; raw prices with
corporate-action credits; fail-closed tradability flags; and an entry check
that refuses anything else.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    ExecutionTimingViolation,
    simulate_ashare_target_weights,
)
from quantagent.data.ashare import execution_panel as ep
from quantagent.data.ashare.contracts import SourceBoundary

SESSIONS = pd.bdate_range("2024-03-04", periods=6)


def _bars(symbol, closes, *, dates=SESSIONS, provider="tickflow", **masks):
    rows = []
    for date, close in zip(dates, closes):
        if close is None:
            continue
        rows.append({
            "symbol": symbol, "trade_date": date, "open": close, "high": close,
            "low": close, "close": close, "volume": 1e7, "amount": close * 1e7,
            "available_at": date + pd.Timedelta(hours=15), "serving_provider": provider,
            "mask_is_suspended": "FALSE", "mask_is_st": "FALSE",
            "mask_limit_up": masks.get("limit_up", "FALSE"),
            "mask_limit_down": "FALSE",
        })
    return pd.DataFrame(rows)


def _gaps(symbol, dates, classification="SUSPENDED"):
    return pd.DataFrame({"symbol": symbol, "trade_date": pd.to_datetime(dates),
                         "classification": classification})


def _factors(symbol, steps):
    return pd.DataFrame([{"symbol": symbol, "effective_date": pd.Timestamp(d), "hfq_factor": f}
                         for d, f in steps])


class TestGapRows:
    def test_in_life_gap_becomes_a_valued_untradeable_row(self):
        bars = _bars("002742.SZ", [10.0, 10.2, None, None, 10.5, 10.6])
        panel, stats = ep.build_execution_panel(
            bars, session_gaps=_gaps("002742.SZ", SESSIONS[2:4]), factors=None)
        gap = panel[panel["gap_classification"] == "SUSPENDED"]
        assert len(gap) == 2
        assert (gap["close"] == 10.2).all()
        assert (gap["volume"] == 0).all() and (gap["amount"] == 0).all()
        assert gap["is_suspended"].all()
        assert not gap["is_limit_up"].any() and not gap["is_limit_down"].any()
        assert stats["rows_by_gap_classification"] == {"TRADED": 4, "SUSPENDED": 2}
        assert (panel["adjustment_method"] == "none").all()

    def test_unexplained_gaps_keep_their_classification(self):
        bars = _bars("002742.SZ", [10.0, None, 10.1, 10.1, 10.1, 10.1])
        panel, _ = ep.build_execution_panel(
            bars, session_gaps=_gaps("002742.SZ", SESSIONS[1:2], "MISSING_UNEXPLAINED"),
            factors=None)
        row = panel[panel["trade_date"] == SESSIONS[1]].iloc[0]
        assert row["gap_classification"] == "MISSING_UNEXPLAINED"
        assert bool(row["is_suspended"]) and row["suspension_status"] == "UNKNOWN"

    def test_gap_before_the_first_bar_has_no_price_and_is_excluded(self):
        bars = pd.concat([_bars("688001.SH", [None, None, 20.0, 20.1, 20.2, 20.3]),
                          _bars("600000.SH", [10.0] * 6)])
        panel, stats = ep.build_execution_panel(
            bars, session_gaps=_gaps("688001.SH", SESSIONS[:2], "PROVIDER_HISTORY_TRUNCATED"),
            factors=None)
        assert (panel["symbol"] == "688001.SH").sum() == 4
        assert stats["gap_rows_without_prior_close_excluded"] == {"PROVIDER_HISTORY_TRUNCATED": 2}

    def test_carried_close_is_rebased_across_an_ex_date_inside_the_gap(self):
        bars = _bars("600000.SH", [10.0, None, None, 9.0, 9.0, 9.0])
        factors = _factors("600000.SH", [("2010-01-01", 1.0), (SESSIONS[2], 1.25)])
        panel, _ = ep.build_execution_panel(
            bars, session_gaps=_gaps("600000.SH", SESSIONS[1:3]), factors=factors)
        by_date = panel.set_index("trade_date")
        assert by_date.loc[SESSIONS[1], "close"] == 10.0
        assert by_date.loc[SESSIONS[2], "close"] == 8.0       # 10 / 1.25
        assert by_date.loc[SESSIONS[2], "ca_cash_per_share"] == pytest.approx(2.0)

    def test_collision_between_gap_table_and_bars_is_refused(self):
        bars = _bars("600000.SH", [10.0] * 6)
        with pytest.raises(ep.ExecutionPanelError, match="collide"):
            ep.build_execution_panel(bars, session_gaps=_gaps("600000.SH", SESSIONS[1:2]),
                                     factors=None)


class TestCorporateActions:
    def test_dividend_record_that_explains_the_factor_step_is_used(self):
        # prev close 10.00, cash 0.50 -> reference 9.50, factor ratio 10/9.5.
        bars = _bars("600000.SH", [10.0, 9.6, 9.6, 9.6, 9.6, 9.6])
        factors = _factors("600000.SH", [("2010-01-01", 1.0), (SESSIONS[1], 10.0 / 9.5)])
        ca = pd.DataFrame([{"symbol": "600000.SH", "ex_date": SESSIONS[1],
                            "cash_dividend_per_share": 0.5, "stock_dividend_ratio": 0.0,
                            "rights_ratio": 0.0}])
        panel, stats = ep.build_execution_panel(bars, session_gaps=None, factors=factors,
                                                corporate_actions=ca)
        row = panel[panel["trade_date"] == SESSIONS[1]].iloc[0]
        assert row["ca_basis"] == ep.CA_BASIS_CORPORATE_ACTION
        assert row["ca_cash_per_share"] == pytest.approx(0.5)
        assert row["ca_share_ratio"] == 0.0
        assert stats["factor_steps_explained_by_dividend_record"] == 1

    def test_transfer_shares_count_as_bonus_shares(self):
        # U0's `rights_ratio` column is the 转增 transfer: 10转10 halves the price.
        bars = _bars("300059.SZ", [20.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        factors = _factors("300059.SZ", [("2010-01-01", 1.0), (SESSIONS[1], 2.0)])
        ca = pd.DataFrame([{"symbol": "300059.SZ", "ex_date": SESSIONS[1],
                            "cash_dividend_per_share": 0.0, "stock_dividend_ratio": 0.0,
                            "rights_ratio": 1.0}])
        panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=factors,
                                            corporate_actions=ca)
        row = panel[panel["trade_date"] == SESSIONS[1]].iloc[0]
        assert row["ca_share_ratio"] == 1.0 and row["ca_cash_per_share"] == 0.0

    def test_unexplained_factor_step_is_credited_at_value(self):
        bars = _bars("600000.SH", [10.0, 8.0, 8.0, 8.0, 8.0, 8.0])
        factors = _factors("600000.SH", [("2010-01-01", 1.0), (SESSIONS[1], 1.25)])
        panel, stats = ep.build_execution_panel(bars, session_gaps=None, factors=factors)
        row = panel[panel["trade_date"] == SESSIONS[1]].iloc[0]
        assert row["ca_basis"] == ep.CA_BASIS_FACTOR_VALUE
        assert row["ca_cash_per_share"] == pytest.approx(2.0)
        assert stats["factor_steps_credited_at_value"] == 1


class TestFlags:
    def test_unknown_limit_state_blocks_the_trade(self):
        bars = _bars("600000.SH", [10.0] * 6, limit_up="UNKNOWN")
        panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=None)
        assert panel["is_limit_up"].all()
        assert (panel["limit_up_status"] == "UNKNOWN").all()

    def test_st_unknown_is_not_reported_as_st_but_stays_visible(self):
        bars = _bars("600000.SH", [10.0] * 6)
        bars["mask_is_st"] = "UNKNOWN"
        panel, stats = ep.build_execution_panel(bars, session_gaps=None, factors=None)
        assert not panel["is_st"].any()
        assert stats["st_status"] == {"UNKNOWN": 6}


class TestVerification:
    def _panel(self, provider_b="tickflow"):
        a = _bars("600000.SH", [10.0] * 6)
        b = _bars("000001.SZ", [10.0] * 3, dates=SESSIONS[:3])
        b2 = _bars("000001.SZ", [10.0] * 3, dates=SESSIONS[3:], provider=provider_b)
        panel, _ = ep.build_execution_panel(pd.concat([a, b, b2]), session_gaps=None,
                                            factors=None)
        return panel

    def test_certified_shape_passes(self):
        report = ep.verify_execution_panel(self._panel())
        assert report["adjustment_method"] == "none"
        assert report["single_source_symbols"] == 2

    def test_adjusted_prices_are_refused(self):
        panel = self._panel()
        panel["adjustment_method"] = "qfq"
        with pytest.raises(ep.ExecutionPanelError, match="raw traded prices"):
            ep.verify_execution_panel(panel)

    def test_undeclared_adjustment_is_refused(self):
        with pytest.raises(ep.ExecutionPanelError, match="does not declare"):
            ep.verify_execution_panel(self._panel().drop(columns=["adjustment_method"]))

    def test_stitched_vendors_need_a_source_boundary(self):
        panel = self._panel(provider_b="sina_truncated")
        with pytest.raises(ep.ExecutionPanelError, match="SourceBoundary"):
            ep.verify_execution_panel(panel)
        boundary = SourceBoundary("000001.SZ", str(SESSIONS[3].date()), "tickflow",
                                  "sina_truncated", "provider switch")
        report = ep.verify_execution_panel(panel, source_boundaries=[boundary])
        assert report["boundary_declared_symbols"] == 1

    def test_raw_panel_without_corporate_actions_is_refused(self):
        panel = self._panel().drop(columns=["ca_cash_per_share"])
        with pytest.raises(ep.ExecutionPanelError, match="corporate-action"):
            ep.verify_execution_panel(panel)


def _sim(weights, panel):
    with tempfile.TemporaryDirectory() as folder:
        return simulate_ashare_target_weights(
            weights, panel, AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=folder))


class TestStrictSimulatorOnTheExecutionPanel:
    """R1-F01: a held name with no bar on a session was fatal."""

    def _bars(self):
        held = _bars("002742.SZ", [10.0, 10.0, None, 10.0, 10.0, 10.0])
        other = _bars("600000.SH", [20.0] * 6)
        return pd.concat([held, other], ignore_index=True)

    def test_raw_bars_alone_crash_on_the_gap(self):
        bars = self._bars()
        sim_panel = bars.assign(is_suspended=False, is_st=False, is_limit_up=False,
                                is_limit_down=False)
        weights = pd.DataFrame({"002742.SZ": [0.5, 0.0]}, index=SESSIONS[[0, 4]])
        with pytest.raises(ExecutionTimingViolation, match="missing_execution_bar"):
            _sim(weights, sim_panel)

    def test_gap_row_values_the_name_and_places_no_order(self):
        panel, _ = ep.build_execution_panel(
            self._bars(), session_gaps=_gaps("002742.SZ", SESSIONS[2:3]), factors=None)
        weights = pd.DataFrame({"002742.SZ": [0.5, 0.0]}, index=SESSIONS[[0, 4]])
        result = _sim(weights, panel)
        assert SESSIONS[2] in result.nav.index
        orders = result.order_audit
        assert not (pd.to_datetime(orders["trade_date"]) == SESSIONS[2]).any()
        # Flat price: NAV on the gap day equals the NAV the day before.
        assert result.nav.loc[SESSIONS[2]] == pytest.approx(result.nav.loc[SESSIONS[1]])

    def test_selling_into_a_gap_day_is_rejected_as_suspended(self):
        panel, _ = ep.build_execution_panel(
            self._bars(), session_gaps=_gaps("002742.SZ", SESSIONS[2:3]), factors=None)
        weights = pd.DataFrame({"002742.SZ": [0.5, 0.0]}, index=SESSIONS[[0, 1]])
        result = _sim(weights, panel)
        failed = result.failed_order_audit
        assert (failed["last_message"].astype(str) == "suspended").any()

    def test_bonus_issue_is_not_booked_as_a_loss(self):
        bars = _bars("300059.SZ", [20.0, 20.0, 10.0, 10.0, 10.0, 10.0])
        factors = _factors("300059.SZ", [("2010-01-01", 1.0), (SESSIONS[2], 2.0)])
        ca = pd.DataFrame([{"symbol": "300059.SZ", "ex_date": SESSIONS[2],
                            "cash_dividend_per_share": 0.0, "stock_dividend_ratio": 0.0,
                            "rights_ratio": 1.0}])
        panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=factors,
                                            corporate_actions=ca)
        weights = pd.DataFrame({"300059.SZ": [0.5, 0.0]}, index=SESSIONS[[0, 4]])
        credited = _sim(weights, panel)
        assert credited.nav.loc[SESSIONS[2]] == pytest.approx(credited.nav.loc[SESSIONS[1]])
        audit = credited.corporate_action_audit
        assert len(audit) == 1 and int(audit["bonus_shares"].iloc[0]) > 0
        # Same bars without the credit columns: the ex-date reads as a -50% loss on
        # the position, which is why the entry check requires them.
        uncredited = _sim(weights, panel.drop(columns=["ca_cash_per_share", "ca_share_ratio"]))
        assert uncredited.nav.loc[SESSIONS[2]] < credited.nav.loc[SESSIONS[2]] * 0.8

    def test_cash_dividend_is_credited(self):
        bars = _bars("600000.SH", [10.0, 10.0, 9.5, 9.5, 9.5, 9.5])
        factors = _factors("600000.SH", [("2010-01-01", 1.0), (SESSIONS[2], 10.0 / 9.5)])
        ca = pd.DataFrame([{"symbol": "600000.SH", "ex_date": SESSIONS[2],
                            "cash_dividend_per_share": 0.5, "stock_dividend_ratio": 0.0,
                            "rights_ratio": 0.0}])
        panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=factors,
                                            corporate_actions=ca)
        weights = pd.DataFrame({"600000.SH": [0.5, 0.0]}, index=SESSIONS[[0, 4]])
        result = _sim(weights, panel)
        assert result.nav.loc[SESSIONS[2]] == pytest.approx(result.nav.loc[SESSIONS[1]])
        assert math.isfinite(float(result.nav.iloc[-1]))


class TestBaselineProtocolPanelOption:
    """`--panel`: the canonical evaluator verifies the execution panel at entry."""

    @staticmethod
    def _write(tmp_path, *, adjustment="none"):
        sessions = pd.bdate_range("2024-03-04", periods=12)
        bars = []
        for i, symbol in enumerate(("600000.SH", "000001.SZ", "300059.SZ")):
            closes = [10.0 + i + 0.05 * k for k in range(12)]
            if symbol == "000001.SZ":
                closes[5] = None  # an in-life gap on a session the book holds it
            bars.append(_bars(symbol, closes, dates=sessions))
        gaps = _gaps("000001.SZ", sessions[5:6], "MISSING_UNEXPLAINED")
        panel, _ = ep.build_execution_panel(pd.concat(bars), session_gaps=gaps, factors=None)
        panel["adjustment_method"] = adjustment
        path = tmp_path / "execution_panel.parquet"
        panel.to_parquet(path, index=False)
        preds = panel[panel["gap_classification"] == "TRADED"][["symbol", "trade_date"]].copy()
        preds["alpha_score"] = preds["symbol"].map({"600000.SH": 0.1, "000001.SZ": 0.9,
                                                    "300059.SZ": 0.5})
        preds = preds[preds["trade_date"].isin(sessions[[0, 6]])]
        pred_path = tmp_path / "preds.parquet"
        preds.to_parquet(pred_path, index=False)
        return str(path), str(pred_path), sessions

    def _bp(self):
        import importlib
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        return importlib.import_module("baseline_protocol")

    def test_verified_panel_runs_through_a_held_gap(self, tmp_path):
        bp = self._bp()
        panel_path, pred_path, sessions = self._write(tmp_path)
        out = bp.evaluate(pred_path, top_k=1, start=str(sessions[0].date()),
                          end=str(sessions[-1].date()), slippage_bps=0.0,
                          variants=["C_flags_eligible_delay1"], panel_path=panel_path)
        assert out["panel"]["verified"] is True
        assert out["benchmark_basis"] == "valuation_sessions_total_return_hfq"
        assert "C_flags_eligible_delay1" in out["variants"]

    def test_adjusted_panel_is_refused_at_entry(self, tmp_path):
        bp = self._bp()
        panel_path, pred_path, sessions = self._write(tmp_path, adjustment="qfq")
        with pytest.raises(ep.ExecutionPanelError, match="raw traded prices"):
            bp.evaluate(pred_path, top_k=1, start=str(sessions[0].date()),
                        end=str(sessions[-1].date()), slippage_bps=0.0,
                        variants=["C_flags_eligible_delay1"], panel_path=panel_path)


class TestDelistingWriteOff:
    def _panel(self):
        held = _bars("600485.SH", [5.0, 5.0, 5.0, None, None, None])
        other = _bars("600000.SH", [20.0] * 6)
        return ep.build_execution_panel(
            pd.concat([held, other], ignore_index=True),
            session_gaps=_gaps("600485.SH", SESSIONS[3:4]), factors=None,
            delisting_dates={"600485.SH": SESSIONS[3], "600000.SH": None})

    def test_one_writeoff_row_follows_the_delisting_session(self):
        panel, stats = self._panel()
        rows = panel[panel["symbol"] == "600485.SH"]
        assert rows["trade_date"].max() == SESSIONS[4]
        last = rows.iloc[-1]
        assert last["gap_classification"] == ep.GAP_DELISTED
        assert bool(last["delisting_writeoff"]) and bool(last["is_suspended"])
        assert stats["delisting_writeoff_rows"] == 1
        assert not panel.loc[panel["symbol"] == "600000.SH", "delisting_writeoff"].any()

    def test_a_name_held_into_delisting_is_written_off_not_fatal(self):
        panel, _ = self._panel()
        weights = pd.DataFrame({"600485.SH": [0.5, 0.0], "600000.SH": [0.0, 0.5]},
                               index=SESSIONS[[0, 4]])
        result = _sim(weights, panel)
        audit = result.corporate_action_audit
        writeoff = audit[audit["basis"] == "delisting_writeoff"]
        assert len(writeoff) == 1 and writeoff["written_off_value"].iloc[0] > 0
        drop = result.nav.loc[SESSIONS[3]] - result.nav.loc[SESSIONS[4]]
        assert drop == pytest.approx(writeoff["written_off_value"].iloc[0], rel=1e-6)
