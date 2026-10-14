"""T+1 pair summary describes the whole source, not the page (round-29 R6)."""
from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from services.quant_api.adapters.do_t import DoTAdapter


def test_summary_is_independent_of_page_size_and_exit_reason_is_not_outcome(empty_quant_ui_settings) -> None:
    source = empty_quant_ui_settings.runtime_root / "reports" / "intraday_dot_factor_combo_r30"
    source.mkdir(parents=True)
    (source / "factor_combo_report.json").write_text(json.dumps({"verdict": "DO_NOT_ENABLE"}), encoding="utf-8")
    pl.DataFrame({
        "symbol": ["000001.SZ"] * 5,
        "trade_date": [datetime(2026, 1, day) for day in (5, 6, 7, 8, 9)],
        "mode": ["dip_buy"] * 5,
        # a take-profit EXIT that still lost money after costs, as on the real source
        "state": ["closed_profit", "closed_profit", "closed_stop", "closed_eod", "closed_eod"],
        "net_ret": [-0.004, 0.006, -0.01, 0.002, None],
        "entry_px": [10.0] * 5,
        "exit_px": [10.1] * 5,
    }).write_parquet(source / "factor_combo_scored.parquet")

    adapter = DoTAdapter(empty_quant_ui_settings)
    source_id = adapter.list_sources()[0]["id"]
    small = adapter.analyze(source_id, limit=2)
    large = adapter.analyze(source_id, limit=500)

    for analysis in (small, large):
        summary = analysis["summary"]
        assert summary["pairCount"] == 5
        assert summary["unmeasuredPairs"] == 1
        assert summary["successRate"] == 0.5  # 2 wins of 4 measured; the unmeasured pair is not a failure
        assert summary["failureRate"] == 0.5
        assert summary["lowBuyFailureRate"] == 0.2
    assert small["summary"]["pairsShown"] == 2
    assert large["summary"]["pairsShown"] == 5

    losing_take_profit = next(pair for pair in large["pairs"] if pair["netPnl"] == -0.004)
    assert losing_take_profit["exitReason"] == "take_profit_trigger"
    assert losing_take_profit["success"] is False
