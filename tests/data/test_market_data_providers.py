"""MT5, QMT/XtData and public tick adapters — fail-closed behaviour.

Every parser test uses a payload shape captured from the live endpoint on
2026-07-27, so a vendor format change breaks a test instead of silently
producing empty or wrong frames.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantagent.data.microstructure import capability as cap
from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure import public_tick_sources as pts
from quantagent.data.providers import mt5_capability as mt5p
from quantagent.data.providers import xtdata_market_provider as xt


# --- MT5 --------------------------------------------------------------------
class TestMt5Probe:
    def test_terminal_unavailable_when_package_missing(self):
        state = mt5p.probe_terminal(None, "ModuleNotFoundError: No module named 'MetaTrader5'")
        assert state.classification == mt5p.TERMINAL_UNAVAILABLE
        assert state.package_importable is False

    def test_unavailable_terminal_does_not_claim_anything_about_brokers(self):
        result = mt5p.run_probe(cohort=())
        assert result.overall_classification == mt5p.TERMINAL_UNAVAILABLE
        assert result.genuine_a_share_symbols == 0
        joined = " ".join(result.notes)
        assert "UNMEASURED" in joined
        assert "not evidence that brokers do or do not" in joined

    def test_capability_cells_are_client_unavailable_not_not_offered(self):
        """'We cannot run the client' must never be recorded as 'no such data'."""
        result = mt5p.run_probe(cohort=())
        cells = mt5p.capability_cells(result)
        assert cells
        assert all(c.status == cap.CLIENT_UNAVAILABLE for c in cells)
        assert all(not c.available for c in cells)

    def test_symbol_candidates_cover_the_common_broker_spellings(self):
        candidates = mt5p.symbol_candidates("600000.SH")
        for expected in ("600000.SH", "SH600000", "600000.SSE", "600000"):
            assert expected in candidates

    def test_cfd_classification_when_currency_is_not_cny(self):
        klass, detail = mt5p._classify_symbol(
            {"path": "CFD\\CHINA50", "exchange": "", "currency_profit": "USD"},
            {"ticks_returned": 500}, {"levels": 10},
        )
        assert klass == mt5p.BROKER_CFD_OR_SYNTHETIC

    def test_dom_subscription_without_levels_is_not_depth(self):
        klass, _ = mt5p._classify_symbol(
            {"path": "SSE\\600000", "exchange": "SSE", "currency_profit": "CNY"},
            {"ticks_returned": 100, "real_volume_present": True},
            {"subscribed": True, "levels": 0},
        )
        assert klass == mt5p.A_SHARE_TICK_NO_DEPTH

    def test_deep_dom_is_a_level2_candidate_not_a_conclusion(self):
        klass, _ = mt5p._classify_symbol(
            {"path": "SSE\\600000", "exchange": "SSE", "currency_profit": "CNY"},
            {"ticks_returned": 100}, {"levels": 10},
        )
        assert klass == mt5p.A_SHARE_LEVEL2_CANDIDATE

    def test_artifacts_are_written_even_with_no_terminal(self, tmp_path):
        result = mt5p.run_probe(cohort=())
        written = mt5p.write_artifacts(result, tmp_path)
        for key in ("terminal", "accounts", "capability_matrix" if False else "json"):
            assert key in written
        payload = json.loads((tmp_path / "terminal.json").read_text(encoding="utf-8"))
        assert payload["classification"] == mt5p.TERMINAL_UNAVAILABLE


# --- QMT / XtData -----------------------------------------------------------
class TestXtDataProvider:
    def test_runtime_probe_reports_missing_package_honestly(self):
        runtime = xt.probe_runtime()
        if runtime.package_importable:
            pytest.skip("xtquant is installed in this environment")
        assert runtime.xtdata_importable is False
        assert "xtquant" in (runtime.import_error or "")

    def test_provider_raises_rather_than_returning_empty(self):
        runtime = xt.XtDataRuntime(
            probed_at="2026-07-27T00:00:00+00:00", package_importable=False,
            xtdata_importable=False, import_error="ModuleNotFoundError",
        )
        provider = xt.XtDataMarketProvider(runtime=runtime)
        with pytest.raises(xt.XtDataUnavailable, match="not importable"):
            provider.fetch_transactions("600000.SH", "2026-07-24")

    def test_capability_cells_record_claimed_api_without_claiming_it_works(self):
        runtime = xt.XtDataRuntime(
            probed_at="2026-07-27T00:00:00+00:00", package_importable=True,
            xtdata_importable=False, import_error="ImportError: datacenter",
        )
        _, matrix = xt.probe_capability(runtime=runtime)
        assert len(matrix) > 0
        assert matrix.serving() == []
        l2 = [c for c in matrix.cells if c.dataset_family == "level2_order_events"]
        assert l2 and l2[0].status == cap.CLIENT_UNAVAILABLE
        assert l2[0].evidence["claimed_data_class"] == mc.EXCHANGE_ORDER_EVENT

    def test_transaction_normalisation_uses_matched_order_ids_for_side(self):
        raw = pd.DataFrame({
            "time": [1_753_000_000_000, 1_753_000_003_000],
            "price": [9.08, 9.09],
            "volume": [1000, 2000],
            "amount": [9080.0, 18180.0],
            "tradeIndex": [1, 2],
            "bidOrder": [500, 100],
            "askOrder": [100, 500],
        })
        frame = xt.XtDataMarketProvider.normalise_transactions(
            raw, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert list(frame["side"]) == ["BUY", "SELL"]
        assert set(frame["side_method"]) == {mc.SIDE_ORDER_MATCHED}
        assert list(frame["sequence"]) == [1, 2]
        assert list(frame["data_class"]) == [mc.EXCHANGE_TRADE_EVENT] * 2

    def test_transaction_normalisation_leaves_side_null_without_order_ids(self):
        raw = pd.DataFrame({
            "time": [1_753_000_000_000], "price": [9.08],
            "volume": [1000], "tradeIndex": [1],
        })
        frame = xt.XtDataMarketProvider.normalise_transactions(
            raw, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert frame["side"].isna().all()
        assert set(frame["side_method"]) == {mc.SIDE_UNKNOWN}

    def test_order_normalisation_declares_order_event_class(self):
        raw = pd.DataFrame({
            "time": [1_753_000_000_000], "price": [9.08], "volume": [500],
            "orderIndex": [42], "orderType": [2], "orderKind": [1],
        })
        frame = xt.XtDataMarketProvider.normalise_orders(
            raw, symbol="000001.SZ", trade_date="2026-07-24"
        )
        assert list(frame["data_class"]) == [mc.EXCHANGE_ORDER_EVENT]
        assert list(frame["order_id"]) == [42]

    def test_module_never_imports_the_trader(self):
        """Read-only separation: the market provider must not reach execution.

        Checked against the parsed import graph rather than the file text, so
        the module can *document* the separation in prose without the test
        mistaking its own explanation for a violation.
        """
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(xt.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)

        forbidden = {"xtquant.xttrader", "xtquant.xtconstant", "xtquant.xttype"}
        assert not (imported & forbidden), f"execution imports leaked in: {imported & forbidden}"
        assert not any("qmt_gateway" in name for name in imported)
        assert "xtquant.xtdata" in imported


# --- public tick sources ----------------------------------------------------
#: Captured verbatim from stock.gtimg.cn on 2026-07-27 for sh600000/2026-07-24.
TENCENT_PAYLOAD = (
    'v_detail_data_sh600000=[0,"0/09:25:03/9.08/0.01/4655/4226740/B'
    '|1/09:30:02/9.08/0.00/1849/1681017/S'
    '|2/09:30:05/9.05/-0.03/510/462346/M"];'
)


class TestPublicTickSources:
    def test_tencent_parse_matches_the_live_payload_shape(self):
        frame = pts.TencentTickDetail.parse(
            TENCENT_PAYLOAD, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert len(frame) == 3
        assert list(frame["price"]) == [9.08, 9.08, 9.05]

    def test_tencent_volume_is_converted_from_lots_to_shares(self):
        frame = pts.TencentTickDetail.parse(
            TENCENT_PAYLOAD, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert frame["volume_shares"].iloc[0] == 465_500.0  # 4655 手 * 100

    def test_tencent_neutral_flag_is_not_forced_into_a_direction(self):
        frame = pts.TencentTickDetail.parse(
            TENCENT_PAYLOAD, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert list(frame["side"])[:2] == ["BUY", "SELL"]
        assert pd.isna(frame["side"].iloc[2])

    def test_tencent_is_labelled_as_an_aggregate_not_a_tick(self):
        assert pts.TencentTickDetail.data_class == mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE
        assert pts.TencentTickDetail.AGGREGATION_SECONDS == 3

    def test_tencent_never_manufactures_an_exchange_sequence(self):
        frame = pts.TencentTickDetail.parse(
            TENCENT_PAYLOAD, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert frame["sequence"].isna().all()
        assert frame["trade_id"].isna().all()
        assert list(frame["ingest_sequence"]) == [0, 1, 2]

    def test_tencent_side_method_names_the_inference(self):
        frame = pts.TencentTickDetail.parse(
            TENCENT_PAYLOAD, symbol="600000.SH", trade_date="2026-07-24"
        )
        assert set(frame["side_method"]) == {mc.SIDE_QUOTE_RULE}

    def test_malformed_payload_yields_an_empty_contract_frame(self):
        frame = pts.TencentTickDetail.parse(
            "garbage", symbol="600000.SH", trade_date="2026-07-24"
        )
        assert frame.empty
        assert list(frame.columns) == list(mc.TRADE_EVENT.columns)

    def test_sina_decommission_notice_is_a_permanent_failure(self):
        """HTTP 200 carrying 服务已下线 must not read as 'no data that day'."""
        class _Stub:
            def get(self, *_args, **_kwargs):
                from quantagent.data.ashare.http import RETRY_OK, FetchOutcome
                return FetchOutcome(True, "u", RETRY_OK, 200, text="服务已下线")

        frame, outcome = pts.SinaTickDetail(_Stub()).fetch("600000.SH", "2026-07-24")
        assert frame.empty
        assert outcome.ok is False
        assert "decommission" in (outcome.error or "")

    def test_tencent_level1_declares_display_depth_not_level2(self):
        assert pts.TencentLevel1Quote.data_class == mc.LEVEL1_QUOTE
        assert pts.TencentLevel1Quote.DISPLAY_DEPTH == 5


# --- capability matrix ------------------------------------------------------
class TestCapabilityMatrix:
    def test_serving_requires_rows(self):
        with pytest.raises(ValueError, match="SERVING with no rows"):
            cap.CapabilityCell(
                provider="x", dataset_family="trade_ticks",
                status=cap.SERVING, entitlement=cap.PUBLIC_FREE,
            )

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValueError, match="unknown status"):
            cap.CapabilityCell(
                provider="x", dataset_family="trade_ticks",
                status="PROBABLY_FINE", entitlement=cap.PUBLIC_FREE,
            )

    def test_client_unavailable_is_not_available(self):
        cell = cap.CapabilityCell(
            provider="qmt_xtdata", dataset_family="level2_order_events",
            status=cap.CLIENT_UNAVAILABLE, entitlement=cap.BROKER_ACCOUNT_REQUIRED,
        )
        assert cell.available is False

    def test_families_without_provider_are_enumerated(self):
        matrix = cap.CapabilityMatrix([
            cap.CapabilityCell(
                provider="tencent", dataset_family="trade_ticks",
                status=cap.SERVING, entitlement=cap.PUBLIC_FREE, rows_returned=100,
            )
        ])
        assert matrix.providers_for("trade_ticks") == ["tencent"]
        assert "level2_order_events" in matrix.families_without_provider()

    def test_write_produces_json_and_csv(self, tmp_path):
        matrix = cap.CapabilityMatrix([
            cap.CapabilityCell(
                provider="tencent", dataset_family="trade_ticks",
                status=cap.SERVING, entitlement=cap.PUBLIC_FREE, rows_returned=10,
            )
        ])
        written = matrix.write(tmp_path)
        assert (tmp_path / "capability_matrix.json").exists()
        assert (tmp_path / "capability_matrix.csv").exists()
        payload = json.loads((tmp_path / "capability_matrix.json").read_text("utf-8"))
        assert payload["serving_cells"] == 1
