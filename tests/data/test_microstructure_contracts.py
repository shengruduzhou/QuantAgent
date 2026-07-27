"""Canonical microstructure contracts: semantics, sessions, fidelity.

These tests lock the properties that make the layer trustworthy rather than
merely functional. Each one corresponds to a way the layer could quietly start
lying: a manufactured identifier accepted, a generated tick journalled, a
queue-position claim made on snapshot data, a session boundary misplaced so a
day fails to reconcile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure import fidelity, integrity
from quantagent.data.microstructure.store import (
    ImmutableStoreError,
    RawEventStore,
    assign_ingest_sequence,
)


def _trade_frame(rows: int = 4, **overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "symbol": ["600000.SH"] * rows,
        "exchange": ["SH"] * rows,
        "trade_date": ["2026-07-24"] * rows,
        "exchange_time": pd.to_datetime(
            [f"2026-07-24 10:00:0{i}" for i in range(rows)]
        ),
        "event_time_ns": [1_700_000_000_000_000_000 + i for i in range(rows)],
        "receive_time_ns": [1_700_000_000_100_000_000 + i for i in range(rows)],
        "ingest_sequence": list(range(rows)),
        "sequence": pd.Series([pd.NA] * rows, dtype="Int64"),
        "source_provider": ["tencent"] * rows,
        "source_channel": ["test"] * rows,
        "data_class": [mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE] * rows,
        "raw_partition": [None] * rows,
        "available_at": pd.to_datetime([f"2026-07-24 10:00:0{i}" for i in range(rows)]),
        "trade_id": pd.Series([pd.NA] * rows, dtype="Int64"),
        "price": [9.0 + 0.01 * i for i in range(rows)],
        "volume_shares": [100.0 * (i + 1) for i in range(rows)],
        "amount_cny": [900.0 * (i + 1) for i in range(rows)],
        "side": ["BUY"] * rows,
        "side_method": [mc.SIDE_QUOTE_RULE] * rows,
        "buy_order_id": pd.Series([pd.NA] * rows, dtype="Int64"),
        "sell_order_id": pd.Series([pd.NA] * rows, dtype="Int64"),
        "trade_kind": [None] * rows,
    })
    for key, value in overrides.items():
        base[key] = value
    return base


# --- session phases ---------------------------------------------------------
class TestSessionPhases:
    @pytest.mark.parametrize(
        "clock,expected",
        [
            ("09:15", mc.PHASE_OPENING_AUCTION),
            ("09:21", mc.PHASE_PRE_OPEN_QUIET),
            ("09:30", mc.PHASE_CONTINUOUS_AM),
            ("11:29", mc.PHASE_CONTINUOUS_AM),
            ("12:00", mc.PHASE_LUNCH_BREAK),
            ("13:00", mc.PHASE_CONTINUOUS_PM),
            ("14:58", mc.PHASE_CLOSING_AUCTION),
            ("08:00", mc.PHASE_CLOSED),
            ("16:00", mc.PHASE_CLOSED),
        ],
    )
    def test_phase_classification(self, clock, expected):
        assert mc.session_phase(clock) == expected

    def test_morning_close_print_is_not_lunch_break(self):
        """Measured: the 11:30:00 close aggregate must stay in the AM session.

        A half-open window ending at 11:30 pushed every symbol's morning-close
        print into the lunch break and made the day fail its own integrity check.
        """
        assert mc.session_phase("11:30") == mc.PHASE_CONTINUOUS_AM

    def test_closing_auction_result_print_lands_in_the_auction(self):
        """The 15:00:0x auction result carries the day's closing volume."""
        assert mc.session_phase("15:00") == mc.PHASE_CLOSING_AUCTION

    def test_post_close_prints_are_quarantined_not_accepted(self):
        assert mc.session_phase("15:10") == mc.PHASE_POST_CLOSE
        assert mc.session_phase("15:40") == mc.PHASE_CLOSED

    def test_star_after_hours_only_applies_to_star(self):
        assert mc.session_phase("15:10", board="STAR") == mc.PHASE_AFTER_HOURS
        assert mc.session_phase("15:10", board="SH_Main") == mc.PHASE_POST_CLOSE

    @pytest.mark.parametrize(
        "symbol,board",
        [
            ("600000.SH", "SH_Main"), ("000001.SZ", "SZ_Main"),
            ("300750.SZ", "ChiNext"), ("688981.SH", "STAR"),
            ("920002.BJ", "BSE"),
        ],
    )
    def test_board_inference(self, symbol, board):
        assert mc.board_of(symbol) == board


# --- immutable store --------------------------------------------------------
class TestRawEventStore:
    def test_round_trip(self, tmp_path):
        store = RawEventStore(tmp_path)
        receipts = store.append(
            _trade_frame(), provider="tencent", family=mc.FAMILY_TRADE,
            data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
        )
        assert len(receipts) == 1
        assert receipts[0].status == "WRITTEN"
        back = store.read(family=mc.FAMILY_TRADE)
        assert len(back) == 4
        assert list(back["ingest_sequence"]) == [0, 1, 2, 3]

    def test_generated_ticks_are_refused(self, tmp_path):
        """Strategy Tester ticks may never enter the authoritative journal."""
        store = RawEventStore(tmp_path)
        with pytest.raises(ImmutableStoreError, match="non-authoritative"):
            store.append(
                _trade_frame(), provider="mt5", family=mc.FAMILY_TRADE,
                data_class=mc.GENERATED_TESTER_TICK,
            )

    def test_bar_derived_ticks_are_refused(self, tmp_path):
        store = RawEventStore(tmp_path)
        with pytest.raises(ImmutableStoreError, match="non-authoritative"):
            store.append(
                _trade_frame(), provider="synthetic", family=mc.FAMILY_TRADE,
                data_class=mc.BAR_DERIVED_TICK,
            )

    def test_unknown_data_class_is_refused(self, tmp_path):
        store = RawEventStore(tmp_path)
        with pytest.raises(ImmutableStoreError, match="not a declared class"):
            store.append(
                _trade_frame(), provider="x", family=mc.FAMILY_TRADE,
                data_class="TOTALLY_MADE_UP",
            )

    def test_side_without_method_is_refused(self, tmp_path):
        """A direction with no stated rule is the fabrication we guard against."""
        store = RawEventStore(tmp_path)
        frame = _trade_frame(side_method=[None] * 4)
        with pytest.raises(ImmutableStoreError, match="side_method"):
            store.append(
                frame, provider="tencent", family=mc.FAMILY_TRADE,
                data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
            )

    def test_missing_contract_columns_are_refused(self, tmp_path):
        store = RawEventStore(tmp_path)
        frame = _trade_frame().drop(columns=["amount_cny"])
        with pytest.raises(ImmutableStoreError, match="missing"):
            store.append(
                frame, provider="tencent", family=mc.FAMILY_TRADE,
                data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
            )

    def test_identical_rewrite_is_deduplicated(self, tmp_path):
        store = RawEventStore(tmp_path)
        kwargs = dict(provider="tencent", family=mc.FAMILY_TRADE,
                      data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE)
        store.append(_trade_frame(), **kwargs)
        again = store.append(_trade_frame(), **kwargs)
        assert again[0].status == "DEDUPLICATED"
        assert len(store.read(family=mc.FAMILY_TRADE)) == 4

    def test_conflicting_rewrite_raises_without_supersede(self, tmp_path):
        store = RawEventStore(tmp_path)
        kwargs = dict(provider="tencent", family=mc.FAMILY_TRADE,
                      data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE)
        store.append(_trade_frame(), **kwargs)
        changed = _trade_frame()
        changed.loc[0, "price"] = 99.0
        with pytest.raises(ImmutableStoreError, match="supersede"):
            store.append(changed, **kwargs)

    def test_supersede_records_a_tombstone_and_hides_the_old_part(self, tmp_path):
        store = RawEventStore(tmp_path)
        kwargs = dict(provider="tencent", family=mc.FAMILY_TRADE,
                      data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE)
        store.append(_trade_frame(), **kwargs)
        changed = _trade_frame()
        changed.loc[0, "price"] = 99.0
        receipts = store.append(changed, supersede=True, **kwargs)
        assert receipts[0].status == "SUPERSEDED"
        assert receipts[0].superseded_part
        back = store.read(family=mc.FAMILY_TRADE)
        assert len(back) == 4 and back["price"].max() == 99.0

    def test_unsafe_partition_token_is_refused(self, tmp_path):
        store = RawEventStore(tmp_path)
        frame = _trade_frame(symbol=["../../etc/passwd"] * 4)
        with pytest.raises(ImmutableStoreError, match="unsafe"):
            store.append(
                frame, provider="tencent", family=mc.FAMILY_TRADE,
                data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
            )

    def test_ingest_sequence_is_not_an_exchange_sequence(self):
        frame = assign_ingest_sequence(_trade_frame(), start=100)
        assert list(frame["ingest_sequence"]) == [100, 101, 102, 103]
        assert frame["sequence"].isna().all()


# --- integrity --------------------------------------------------------------
class TestIntegrityChecks:
    def test_clean_frame_has_no_failures(self):
        report = integrity.run_integrity_checks(
            _trade_frame(), family=mc.FAMILY_TRADE,
            data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
        )
        assert report.failed == []

    def test_not_run_is_not_a_pass(self):
        """The bug this guards: gates that treat 'unevaluated' as 'fine'."""
        report = integrity.run_integrity_checks(
            _trade_frame(), family=mc.FAMILY_TRADE,
            data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
        )
        assert report.not_run, "no-sequence source should leave checks unevaluated"
        assert report.usable is False

    def test_manufactured_identifier_is_caught(self):
        frame = _trade_frame()
        frame["trade_id"] = pd.Series(range(len(frame)), dtype="Int64")
        result = integrity.check_manufactured_fields(frame, mc.FAMILY_TRADE)
        assert result.verdict == integrity.FAIL

    def test_unknown_semantics_fails(self):
        frame = _trade_frame(data_class=[mc.UNKNOWN_SEMANTICS] * 4)
        result = integrity.check_declared_semantics(frame)
        assert result.verdict == integrity.FAIL

    def test_lunch_break_events_fail(self):
        frame = _trade_frame()
        frame["exchange_time"] = pd.to_datetime(["2026-07-24 12:15:00"] * 4)
        result = integrity.check_session_boundaries(frame)
        assert result.verdict == integrity.FAIL

    def test_post_close_events_warn_rather_than_fail(self):
        frame = _trade_frame()
        frame["exchange_time"] = pd.to_datetime(["2026-07-24 15:10:00"] * 4)
        result = integrity.check_session_boundaries(frame)
        assert result.verdict == integrity.WARN
        assert result.evidence["post_close_rows"] == 4

    def test_inferred_side_warns_and_is_counted_separately(self):
        result = integrity.check_side_provenance(_trade_frame())
        assert result.verdict == integrity.WARN
        assert result.evidence["observed_rows"] == 0
        assert result.evidence["inferred_rows"] == 4

    def test_exchange_published_side_passes(self):
        frame = _trade_frame(side_method=[mc.SIDE_EXCHANGE_PUBLISHED] * 4)
        result = integrity.check_side_provenance(frame)
        assert result.verdict == integrity.PASS

    def test_out_of_order_timestamps_are_detected(self):
        frame = _trade_frame()
        frame.loc[2, "event_time_ns"] = frame.loc[0, "event_time_ns"] - 5_000_000_000
        result = integrity.check_timestamp_monotonicity(frame)
        assert result.verdict == integrity.FAIL

    def test_crossed_book_is_detected(self):
        book = pd.DataFrame({
            "symbol": ["600000.SH"] * 2, "trade_date": ["2026-07-24"] * 2,
            "snapshot_sequence": [1, 1], "level": [1, 2],
            "bid_price": [10.05, 10.00], "ask_price": [10.00, 10.10],
            "ingest_sequence": [0, 1],
        })
        result = integrity.check_book_ordering(book)
        assert result.verdict == integrity.FAIL
        assert result.evidence["crossed_or_locked_snapshots"] == 1

    def test_sequence_gaps_detected_when_published(self):
        frame = _trade_frame()
        frame["sequence"] = pd.Series([1, 2, 9, 10], dtype="Int64")
        result = integrity.check_sequence_gaps(frame)
        assert result.verdict == integrity.FAIL
        assert result.evidence["total_missing"] == 6

    def test_cumulative_reset_detected(self):
        quote = pd.DataFrame({
            "symbol": ["600000.SH"] * 3, "trade_date": ["2026-07-24"] * 3,
            "ingest_sequence": [0, 1, 2],
            "cum_volume_shares": [100.0, 200.0, 50.0],
        })
        result = integrity.check_cumulative_monotonicity(quote)
        assert result.verdict == integrity.FAIL


# --- fidelity ---------------------------------------------------------------
class TestFidelity:
    def test_snapshot_data_cannot_claim_queue_position(self):
        decision = fidelity.decide_fidelity(
            data_classes=[mc.LEVEL2_SNAPSHOT],
            integrity_reports=[],
        )
        assert decision.level == fidelity.LEVEL_B
        for claim in fidelity.QUEUE_CLAIMS:
            assert not decision.permits(claim)
        with pytest.raises(fidelity.FidelityViolation):
            fidelity.assert_claims_permitted(decision, ["queue_position"])

    def test_order_events_license_level_a(self):
        clean = integrity.IntegrityReport(
            family=mc.FAMILY_ORDER, data_class=mc.EXCHANGE_ORDER_EVENT,
            rows=10, symbols=1,
            checks=[integrity.CheckResult("x", integrity.PASS, "ok")],
        )
        decision = fidelity.decide_fidelity(
            data_classes=[mc.EXCHANGE_ORDER_EVENT], integrity_reports=[clean]
        )
        assert decision.level == fidelity.LEVEL_A
        assert decision.permits("queue_position")

    def test_unevaluated_checks_downgrade_level_a(self):
        skipped = integrity.IntegrityReport(
            family=mc.FAMILY_ORDER, data_class=mc.EXCHANGE_ORDER_EVENT,
            rows=10, symbols=1,
            checks=[integrity.CheckResult("sequence_gaps", integrity.NOT_RUN, "no seq")],
        )
        decision = fidelity.decide_fidelity(
            data_classes=[mc.EXCHANGE_ORDER_EVENT], integrity_reports=[skipped]
        )
        assert decision.level == fidelity.LEVEL_B
        assert any("Level A requires proven" in d for d in decision.downgrades)

    def test_failed_integrity_blocks_all_simulation(self):
        broken = integrity.IntegrityReport(
            family=mc.FAMILY_TRADE, data_class=mc.TRADE_TICK, rows=10, symbols=1,
            checks=[integrity.CheckResult("session_boundaries", integrity.FAIL, "bad")],
        )
        decision = fidelity.decide_fidelity(
            data_classes=[mc.TRADE_TICK], integrity_reports=[broken]
        )
        assert decision.level == fidelity.NOT_SIMULATABLE

    def test_unknown_semantics_blocks_all_simulation(self):
        decision = fidelity.decide_fidelity(data_classes=[mc.UNKNOWN_SEMANTICS])
        assert decision.level == fidelity.NOT_SIMULATABLE

    def test_broker_synthetic_depth_is_withdrawn(self):
        decision = fidelity.decide_fidelity(
            data_classes=[mc.LEVEL2_SNAPSHOT, mc.BROKER_SYNTHETIC_QUOTE]
        )
        assert decision.level == fidelity.LEVEL_C
        assert any("not an exchange order book" in d for d in decision.downgrades)

    def test_aggregated_ticks_declare_their_blind_spot(self):
        decision = fidelity.decide_fidelity(
            data_classes=[mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE]
        )
        assert decision.level == fidelity.LEVEL_C
        assert any("3-second aggregates" in d for d in decision.downgrades)
