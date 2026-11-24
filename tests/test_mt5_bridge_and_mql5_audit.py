"""MT5 custom-symbol bridge and the MQL5 static audit.

The audit tests matter most: a checker that reports "clean" but cannot catch
known-bad code is worse than no checker, so every rule is proven against a
deliberately broken sample before the repository's own sources are trusted.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantagent.data.microstructure import contracts as mc
from quantagent.mt5 import custom_symbol_bridge as bridge


# --- bridge -----------------------------------------------------------------
def _events(n=5, symbol="600000.SH", with_quotes=False):
    frame = pd.DataFrame({
        "symbol": [symbol] * n,
        "exchange_time": pd.to_datetime([f"2026-07-24 10:00:0{i}" for i in range(n)]),
        "price": [9.0 + 0.01 * i for i in range(n)],
        "volume_shares": [1000.0 * (i + 1) for i in range(n)],
        "side": ["BUY", "SELL"] * (n // 2) + ["BUY"] * (n % 2),
    })
    if with_quotes:
        frame["bid_price"] = frame["price"] - 0.01
        frame["ask_price"] = frame["price"] + 0.01
    return frame


class TestCustomSymbolNaming:
    def test_name_is_unmistakably_ours(self):
        assert bridge.custom_symbol_name("600000.SH") == "QA_600000_SH"
        assert bridge.custom_symbol_name("920002.BJ") == "QA_920002_BJ"

    def test_round_trip(self):
        for canonical in ("600000.SH", "000001.SZ", "688981.SH", "920002.BJ"):
            name = bridge.custom_symbol_name(canonical)
            assert bridge.canonical_from_custom(name) == canonical

    def test_broker_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="not a QuantAgent custom symbol"):
            bridge.canonical_from_custom("EURUSD")


class TestSymbolSpec:
    def test_contract_size_is_one_share_not_an_fx_default(self):
        spec = bridge.build_symbol_spec("600000.SH")
        assert spec.contract_size == 1.0
        assert spec.currency == "CNY"
        assert spec.digits == 2

    def test_star_gets_its_own_lot_rules_and_after_hours_session(self):
        spec = bridge.build_symbol_spec("688981.SH")
        assert spec.board == "STAR"
        assert spec.volume_min == 200.0
        assert spec.volume_step == 1.0
        assert ("15:05", "15:30") in spec.trade_sessions

    def test_main_board_has_no_after_hours_session(self):
        spec = bridge.build_symbol_spec("600000.SH")
        assert ("15:05", "15:30") not in spec.trade_sessions


class TestTickExport:
    def test_shares_land_in_volume_real(self):
        ticks = bridge.events_to_mt5_ticks(_events())
        assert list(ticks["volume_real"]) == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

    def test_bid_ask_are_not_fabricated_from_last_price(self):
        """A synthesised spread would make terminal spread indicators lie."""
        ticks = bridge.events_to_mt5_ticks(_events(with_quotes=False))
        assert (ticks["bid"] == 0.0).all()
        assert (ticks["ask"] == 0.0).all()

    def test_quotes_are_used_when_the_source_has_them(self):
        ticks = bridge.events_to_mt5_ticks(_events(with_quotes=True))
        assert (ticks["bid"] > 0).all()
        assert bool(ticks["flags"].iloc[0] & bridge.TICK_FLAG_BID)

    def test_side_flags_are_set_from_the_declared_side(self):
        ticks = bridge.events_to_mt5_ticks(_events())
        assert bool(ticks["flags"].iloc[0] & bridge.TICK_FLAG_BUY)
        assert bool(ticks["flags"].iloc[1] & bridge.TICK_FLAG_SELL)


class TestImportPlan:
    def test_imported_class_becomes_custom_symbol_replay(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(),
            origin_data_class=mc.EXCHANGE_TRADE_EVENT, output_dir=tmp_path,
        )
        assert manifest.imported_data_class == mc.CUSTOM_SYMBOL_REPLAY
        assert manifest.origin_data_class == mc.EXCHANGE_TRADE_EVENT

    def test_aggregate_origin_is_warned_about(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(),
            origin_data_class=mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE,
            output_dir=tmp_path,
        )
        assert any("3-second snapshot aggregates" in w for w in manifest.warnings)

    def test_undeclared_origin_is_warned_about(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(), output_dir=tmp_path,
        )
        assert any("not declared" in w for w in manifest.warnings)

    def test_manifest_hashes_both_source_and_export(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(),
            origin_data_class=mc.TRADE_TICK, output_dir=tmp_path,
        )
        assert manifest.source_content_hash
        assert manifest.export_content_hash
        assert manifest.source_content_hash != manifest.export_content_hash

    def test_manifest_is_written_to_disk(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(),
            origin_data_class=mc.TRADE_TICK, output_dir=tmp_path,
        )
        payload = json.loads(
            (tmp_path / "QA_600000_SH" / "import_manifest.json").read_text("utf-8")
        )
        assert payload["tick_rows"] == 5


class TestImportVerification:
    def test_matching_counts_pass(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(), origin_data_class=mc.TRADE_TICK,
            output_dir=tmp_path,
        )
        result = bridge.verify_import(manifest, observed_ticks=5, observed_bars=0)
        assert result.verdict == "COUNTS_MATCH"

    def test_dropped_rows_are_detected(self, tmp_path):
        manifest = bridge.build_import_plan(
            "600000.SH", events=_events(), origin_data_class=mc.TRADE_TICK,
            output_dir=tmp_path,
        )
        result = bridge.verify_import(manifest, observed_ticks=3, observed_bars=0)
        assert result.verdict == "COUNT_MISMATCH"
        assert "tick count mismatch" in result.problems[0]


class TestTesterTickClassification:
    def test_generated_ticks_are_not_reportable_as_real(self):
        for mode in (bridge.TESTER_EVERY_TICK, bridge.TESTER_ONE_MINUTE_OHLC,
                     bridge.TESTER_OPEN_PRICES):
            result = bridge.classify_tester_ticks(mode)
            assert result["data_class"] == mc.GENERATED_TESTER_TICK
            assert result["reportable_as_real_ticks"] is False

    def test_real_tick_mode_is_recognised(self):
        result = bridge.classify_tester_ticks(bridge.TESTER_EVERY_TICK_REAL)
        assert result["uses_real_ticks"] is True
        assert result["data_class"] == mc.CUSTOM_SYMBOL_REPLAY


# --- MQL5 static audit ------------------------------------------------------
BAD_EA = """
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>
CTrade trade;

int OnInit()
  {
   return INIT_SUCCEEDED;
  }

void OnTick()
  {
   double fast = iMA(_Symbol, PERIOD_D1, 10, 0, MODE_EMA, PRICE_CLOSE);
   double slow = iMA(_Symbol, PERIOD_D1, 30, 0, MODE_EMA, PRICE_CLOSE);
   if(fast > slow)
     {
      trade.buy(0.01);
      lots *= 2;
     }
   double buffer[];
   CopyBuffer(handle, 0, 0, 10, buffer);
  }
"""


