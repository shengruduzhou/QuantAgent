"""U0 raw panel -> full-universe gold dataset bridge.

The tests are organised around the specific ways this bridge could reintroduce
the failure it was built to prevent: a mixed adjustment scale declared as
something else, a missing source silently becoming a negative mask, a row with
no tick data reading as zero order flow, a same-day label sneaking back in, or
a training gate that passes because the build succeeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.ashare import contracts, gold_bridge


def _panel(symbols=("AAA.SZ",), days=40, start="2025-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for symbol in symbols:
        for i, day in enumerate(dates):
            price = 10.0 + i * 0.1
            rows.append({
                "symbol": symbol, "trade_date": day,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 100_000.0, "amount": price * 100_000.0,
            })
    return pd.DataFrame(rows)


def _master(symbols=("AAA.SZ",)) -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": s, "board": "SZ_Main", "listing_date": pd.Timestamp("2020-01-02"),
         "delisting_date": pd.NaT}
        for s in symbols
    ])


def _factors(symbol="AAA.SZ") -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": symbol, "effective_date": pd.Timestamp("2020-01-02"), "hfq_factor": 1.0},
        {"symbol": symbol, "effective_date": pd.Timestamp("2025-01-20"), "hfq_factor": 2.0},
    ])


class TestAdjustment:
    def test_unknown_method_is_refused(self):
        with pytest.raises(gold_bridge.GoldBridgeError, match="unknown adjustment"):
            gold_bridge.apply_adjustment(_panel(), _factors(), method="qfq_ish")

    def test_declared_adjustment_without_factors_is_refused(self):
        """The original bug: prices declared one scale and carried another."""
        with pytest.raises(gold_bridge.GoldBridgeError, match="declared scale"):
            gold_bridge.apply_adjustment(
                _panel(), pd.DataFrame(), method=contracts.ADJUST_HFQ
            )

    def test_hfq_steps_at_the_effective_date(self):
        result = gold_bridge.apply_adjustment(
            _panel(), _factors(), method=contracts.ADJUST_HFQ
        )
        before = result[result["trade_date"] < "2025-01-20"]["adjust_factor"]
        after = result[result["trade_date"] >= "2025-01-20"]["adjust_factor"]
        assert (before == 1.0).all()
        assert (after == 2.0).all()

    def test_volume_is_never_scaled_by_the_price_adjustment(self):
        """Mixing an adjusted close with a raw volume is the recorded bug."""
        raw = _panel()
        result = gold_bridge.apply_adjustment(
            raw, _factors(), method=contracts.ADJUST_HFQ
        )
        assert (result["volume"] == 100_000.0).all()
        assert (result["amount"] == raw["amount"].values).all()

    def test_none_leaves_prices_untouched(self):
        raw = _panel()
        result = gold_bridge.apply_adjustment(
            raw, _factors(), method=contracts.ADJUST_NONE
        )
        assert (result["close"].values == raw["close"].values).all()
        assert (result["adjust_factor"] == 1.0).all()

    def test_qfq_rebases_so_the_latest_price_is_the_traded_price(self):
        raw = _panel()
        result = gold_bridge.apply_adjustment(
            raw, _factors(), method=contracts.ADJUST_QFQ
        )
        latest = result.sort_values("trade_date").iloc[-1]
        raw_latest = raw.sort_values("trade_date").iloc[-1]
        assert latest["close"] == pytest.approx(raw_latest["close"])

    def test_every_price_column_gets_the_same_scale(self):
        result = gold_bridge.apply_adjustment(
            _panel(), _factors(), method=contracts.ADJUST_HFQ
        )
        after = result[result["trade_date"] >= "2025-01-20"]
        for column in gold_bridge.PRICE_COLUMNS:
            ratio = after[column] / _panel().set_index(
                ["symbol", "trade_date"]
            ).loc[list(zip(after["symbol"], after["trade_date"])), column].values
            assert np.allclose(ratio, 2.0)


class TestMasks:
    def test_absent_source_produces_unknown_not_false(self):
        """'No ST register for this exchange' must differ from 'not ST'."""
        masked = gold_bridge.build_masks(
            _panel(), master=_master(), suspension=None, st=None, st_available=False
        )
        assert (masked["mask_is_st"] == gold_bridge.MASK_UNKNOWN).all()

    def test_available_source_produces_a_real_negative(self):
        masked = gold_bridge.build_masks(
            _panel(), master=_master(), st=pd.DataFrame(
                columns=["symbol", "effective_start", "effective_end"]
            ), st_available=True,
        )
        assert (masked["mask_is_st"] == gold_bridge.MASK_FALSE).all()

    def test_suspension_interval_marks_the_right_days(self):
        suspension = pd.DataFrame([{
            "symbol": "AAA.SZ", "effective_start": pd.Timestamp("2025-01-08"),
            "effective_end": pd.Timestamp("2025-01-10"),
        }])
        masked = gold_bridge.build_masks(
            _panel(), master=_master(), suspension=suspension
        )
        flagged = masked[masked["mask_is_suspended"] == gold_bridge.MASK_TRUE]
        assert set(flagged["trade_date"].dt.strftime("%Y-%m-%d")) == {
            "2025-01-08", "2025-01-09", "2025-01-10"
        }

    def test_seasoning_counts_sessions_not_calendar_days(self):
        masked = gold_bridge.build_masks(
            _panel(days=40), master=_master(), seasoning_days=20
        )
        assert int((masked["mask_seasoning"] == gold_bridge.MASK_TRUE).sum()) == 20

    def test_pre_listing_rows_are_flagged(self):
        master = _master()
        master.loc[0, "listing_date"] = pd.Timestamp("2025-01-15")
        masked = gold_bridge.build_masks(_panel(), master=master)
        assert (masked["mask_pre_listing"] == gold_bridge.MASK_TRUE).any()

    def test_eligibility_excludes_unknown_free_rows_only(self):
        masked = gold_bridge.build_masks(
            _panel(days=30), master=_master(), seasoning_days=5
        )
        # ST is UNKNOWN, which must not by itself exclude a row.
        assert masked["eligible_for_training"].sum() == 25


class TestAvailability:
    def test_absent_family_is_false_meaning_not_observed(self):
        result = gold_bridge.attach_availability(_panel(), {})
        for family in gold_bridge.OPTIONAL_FAMILIES:
            assert (result[f"has_{family}"] == False).all()  # noqa: E712

    def test_observed_days_are_marked(self):
        panel = _panel(days=5)
        observed = {"tick_events": pd.DataFrame({
            "symbol": ["AAA.SZ"], "trade_date": [panel["trade_date"].iloc[2]],
        })}
        result = gold_bridge.attach_availability(panel, observed)
        assert result["has_tick_events"].sum() == 1
        assert bool(result["has_tick_events"].iloc[2])

    def test_indicator_distinguishes_missing_from_zero(self):
        """A row without tick data must not read as zero order flow."""
        panel = _panel(days=3)
        result = gold_bridge.attach_availability(panel, {})
        assert "has_tick_events" in result.columns
        assert not result["has_tick_events"].any()


class TestLabels:
    def test_labels_are_delay_one_not_same_day(self):
        panel = _panel(days=10)
        labelled, _ = gold_bridge.build_labels(panel, horizons=[1])
        row = labelled.iloc[0]
        closes = panel.sort_values("trade_date")["close"].tolist()
        expected = closes[2] / closes[1] - 1.0
        assert row["forward_return_1d"] == pytest.approx(expected)

    def test_entry_price_is_the_next_close_not_todays(self):
        panel = _panel(days=5)
        labelled, _ = gold_bridge.build_labels(panel, horizons=[1])
        closes = panel.sort_values("trade_date")["close"].tolist()
        assert labelled.iloc[0]["entry_close_t1"] == pytest.approx(closes[1])

    def test_suspended_entry_rows_are_dropped(self):
        panel = gold_bridge.build_masks(
            _panel(days=10), master=_master(),
            suspension=pd.DataFrame([{
                "symbol": "AAA.SZ", "effective_start": pd.Timestamp("2025-01-08"),
                "effective_end": pd.Timestamp("2025-01-08"),
            }]),
        )
        labelled, dropped = gold_bridge.build_labels(panel, horizons=[1])
        assert dropped["suspended_at_t"] == 1
        assert not (
            labelled["trade_date"] == pd.Timestamp("2025-01-08")
        ).any()

    def test_sealed_limit_up_entry_is_dropped(self):
        panel = _panel(days=10)
        panel["mask_limit_up"] = gold_bridge.MASK_FALSE
        panel.loc[3, "mask_limit_up"] = gold_bridge.MASK_TRUE
        labelled, dropped = gold_bridge.build_labels(panel, horizons=[1])
        assert dropped["limit_up_at_t1"] == 1

    def test_tail_rows_without_an_entry_price_are_dropped(self):
        panel = _panel(days=5)
        labelled, dropped = gold_bridge.build_labels(panel, horizons=[1])
        assert dropped["entry_price_missing"] == 1


class TestBuildAndCertify:
    def _build(self, **kwargs):
        return gold_bridge.build_gold_dataset(
            _panel(days=40), master=_master(), factors=_factors(),
            adjustment_method=contracts.ADJUST_HFQ, horizons=[1, 5], **kwargs
        )

    def test_manifest_records_the_factor_version_and_content_hash(self):
        _, manifest = self._build()
        assert manifest.adjustment_factor_version != "none"
        assert manifest.content_hash
        assert manifest.adjustment_method == contracts.ADJUST_HFQ

    def test_manifest_warns_when_st_is_unavailable(self):
        _, manifest = self._build(st_available=False)
        assert any("NOT point-in-time complete" in w for w in manifest.warnings)

    def test_empty_panel_is_refused(self):
        with pytest.raises(gold_bridge.GoldBridgeError, match="empty panel"):
            gold_bridge.build_gold_dataset(
                pd.DataFrame(), master=_master(), factors=_factors()
            )

    def test_training_is_blocked_when_u0_pit_withholds_permission(self):
        _, manifest = self._build()
        certificate = gold_bridge.certify_training_slice(
            manifest,
            u0_pit_certificate={
                "decision": "FULL_UNIVERSE_DATA_NOT_READY_PIT",
                "training_permitted": False,
                "blocked_pit_fields": ["st_intervals"],
            },
        )
        assert certificate.training_permitted is False
        assert certificate.decision == "TRAINING_BLOCKED"
        assert any("withholds training permission" in b for b in certificate.blockers)

    def test_missing_certificate_is_not_permission(self):
        """Absence of evidence must not read as evidence of readiness."""
        _, manifest = self._build()
        certificate = gold_bridge.certify_training_slice(manifest, u0_pit_certificate=None)
        assert certificate.training_permitted is False
        assert any("absence of evidence" in b for b in certificate.blockers)

    def test_a_successful_build_alone_does_not_permit_training(self):
        _, manifest = self._build()
        certificate = gold_bridge.certify_training_slice(
            manifest,
            u0_pit_certificate={"decision": "FULL_UNIVERSE_DATA_READY",
                                "training_permitted": True, "blocked_pit_fields": []},
        )
        # ST still unavailable in the build itself, so the gate holds.
        assert certificate.training_permitted is False
        assert any("incomplete PIT mask" in b for b in certificate.blockers)

    def test_fully_evidenced_build_is_permitted(self):
        dataset, manifest = gold_bridge.build_gold_dataset(
            _panel(days=40), master=_master(), factors=_factors(),
            st=pd.DataFrame(columns=["symbol", "effective_start", "effective_end"]),
            st_available=True, adjustment_method=contracts.ADJUST_HFQ, horizons=[1],
        )
        certificate = gold_bridge.certify_training_slice(
            manifest,
            u0_pit_certificate={"decision": "FULL_UNIVERSE_DATA_READY",
                                "training_permitted": True, "blocked_pit_fields": []},
        )
        assert certificate.training_permitted is True
        assert len(dataset) > 0
