from __future__ import annotations

import pandas as pd

from quantagent.research.intraday_dot_walkforward import (
    evaluate_walk_forward_results,
    make_walk_forward_splits,
    render_markdown_report,
)


def _trades(n=5):
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-06-01", periods=n, freq="D"),
            "action": ["SELL_HIGH"] * n,
            "completed_round_trip": [1] * n,
            "eod_restore": [0] * n,
            "net_pnl_bps": [12.0] * n,
            "gross_pnl_bps": [30.0] * n,
            "daily_uplift_bps": [2.0] * n,
            "sell_high_fail_new_high": [0] * n,
            "buy_low_fail_breakdown": [0] * n,
            "confidence": [0.7] * n,
            "regime": ["range"] * n,
            "capacity_usage": [0.03] * n,
            "turnover": [0.1] * n,
        }
    )


def test_walk_forward_sample_below_300_is_paper_only():
    report = evaluate_walk_forward_results(_trades(5), min_round_trips=300)
    assert report.verdict == "PAPER_ONLY"
    assert "证据不足" in report.reason
    assert report.metrics["completed_round_trips"] == 5


def test_walk_forward_no_trades_do_not_enable():
    report = evaluate_walk_forward_results(pd.DataFrame(), min_round_trips=300)
    assert report.verdict == "DO_NOT_ENABLE"


def test_walk_forward_report_renders_required_sections(tmp_path):
    report = evaluate_walk_forward_results(_trades(3), min_round_trips=300)
    path = render_markdown_report(report, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "# A股日内做T模型重构报告" in text
    assert "## 14. 是否可部署" in text
    assert "PAPER_ONLY" in text


def test_make_walk_forward_splits_are_ordered():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    splits = make_walk_forward_splits(dates, train_days=10, validation_days=5, test_days=3)
    assert splits
    first = splits[0]
    assert first.train_end < first.validation_start < first.test_start
