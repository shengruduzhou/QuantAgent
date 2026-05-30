"""Tests for the Stage 5.1 broker reports data layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.broker import (
    BROKER_REPORT_REQUIRED_COLUMNS,
    BROKER_TIER_TABLE,
    BrokerReportBuilder,
    BrokerReportConfig,
    apply_broker_report_features,
    broker_reports_for_features,
    build_broker_reports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _good_batch(n: int = 12) -> pd.DataFrame:
    brokers = ["中信证券", "华泰证券", "中金公司", "招商证券"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "broker": brokers[i % len(brokers)],
                "symbol": f"60000{i:02d}.SH",
                "announced_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "rating": "买入" if i % 2 == 0 else "增持",
                "rating_change": "upgrade" if i % 3 == 0 else "maintain",
                "target_price": 50.0 + i,
                "prev_target_price": 45.0 + i,
                "summary": f"研报{i}",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------

def test_known_broker_assigned_tier_1():
    raw = _make_raw(
        [{"broker": "中信证券", "symbol": "A.SH", "announced_at": "2024-01-15"}]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    row = result.frame.iloc[0]
    assert row["broker_tier"] == "tier_1"
    assert row["broker_credibility"] >= 0.80


def test_unknown_broker_falls_back_to_tier_3():
    raw = _make_raw(
        [{"broker": "山寨证券", "symbol": "A.SH", "announced_at": "2024-01-15"}]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    row = result.frame.iloc[0]
    assert row["broker_tier"] == "tier_3"
    assert row["broker_credibility"] == 0.50


def test_partial_broker_name_match():
    raw = _make_raw(
        [{"broker": "中信证券股份有限公司", "symbol": "A.SH", "announced_at": "2024-01-15"}]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    assert result.frame.iloc[0]["broker_tier"] == "tier_1"


def test_credibility_override_supersedes_tier():
    raw = _make_raw(
        [
            {
                "broker": "山寨证券",
                "symbol": "A.SH",
                "announced_at": "2024-01-15",
                "broker_credibility_override": 0.92,
            }
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    assert result.frame.iloc[0]["broker_credibility"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Rating normalisation
# ---------------------------------------------------------------------------

def test_chinese_rating_normalised_to_english():
    raw = _make_raw(
        [
            {"broker": "中信", "symbol": "A.SH", "announced_at": "2024-01-15", "rating": "买入"},
            {"broker": "华泰", "symbol": "B.SH", "announced_at": "2024-01-15", "rating": "增持"},
            {"broker": "中金", "symbol": "C.SH", "announced_at": "2024-01-15", "rating": "持有"},
            {"broker": "招商", "symbol": "D.SH", "announced_at": "2024-01-15", "rating": "减持"},
            {"broker": "国君", "symbol": "E.SH", "announced_at": "2024-01-15", "rating": "卖出"},
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    ratings = list(result.frame.sort_values("symbol")["rating"])
    assert ratings == ["buy", "overweight", "hold", "underweight", "sell"]


def test_invalid_rating_becomes_n_a():
    raw = _make_raw(
        [{"broker": "中信", "symbol": "A.SH", "announced_at": "2024-01-15", "rating": "garbage"}]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    assert result.frame.iloc[0]["rating"] == "n/a"


# ---------------------------------------------------------------------------
# Target price + pct change
# ---------------------------------------------------------------------------

def test_target_price_pct_change_derived():
    raw = _make_raw(
        [
            {
                "broker": "中信",
                "symbol": "A.SH",
                "announced_at": "2024-01-15",
                "target_price": 60.0,
                "prev_target_price": 50.0,
            }
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    assert result.frame.iloc[0]["target_price_pct_change"] == pytest.approx(0.20)


def test_missing_prev_target_gives_nan_pct():
    raw = _make_raw(
        [
            {"broker": "中信", "symbol": "A.SH", "announced_at": "2024-01-15", "target_price": 60.0}
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=1, min_unique_brokers=1))
    assert pd.isna(result.frame.iloc[0]["target_price_pct_change"])


# ---------------------------------------------------------------------------
# Schema + PIT
# ---------------------------------------------------------------------------

def test_required_columns_present():
    result = build_broker_reports(_good_batch(n=12))
    assert set(result.frame.columns) == set(BROKER_REPORT_REQUIRED_COLUMNS)


def test_available_at_uses_business_day_lag():
    raw = _make_raw(
        [{"broker": "中信", "symbol": "A.SH", "announced_at": "2024-01-15"}]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(
        min_events=1, min_unique_brokers=1, available_at_lag_days=1,
    ))
    row = result.frame.iloc[0]
    assert row["available_at"] > row["announced_at"]


def test_dedup_by_event_id():
    raw = _good_batch(n=12)
    dup = pd.concat([raw, raw.head(4)], ignore_index=True)
    result = build_broker_reports(dup)
    assert result.coverage["duplicates_removed"] == 4


def test_missing_required_columns_raises():
    raw = pd.DataFrame([{"broker": "中信", "announced_at": "2024-01-15"}])  # no symbol
    with pytest.raises(ValueError, match="missing required columns"):
        build_broker_reports(raw)


def test_empty_input_closed_gate():
    result = build_broker_reports(pd.DataFrame())
    assert result.frame.empty
    assert result.coverage["gate"]["broker_reports_usable_for_features"] is False
    assert result.coverage["gate"]["reason"] == "no_events"


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def test_gate_opens_with_good_batch():
    result = build_broker_reports(_good_batch(n=12))
    gate = result.coverage["gate"]
    assert gate["broker_reports_usable_for_features"] is True
    assert gate["reason"] == "passed"


def test_gate_blocks_with_too_few_events():
    raw = _good_batch(n=3)
    result = build_broker_reports(raw, config=BrokerReportConfig(min_events=10))
    gate = result.coverage["gate"]
    assert gate["broker_reports_usable_for_features"] is False
    assert "too_few_events" in gate["reason"]


def test_gate_blocks_with_too_few_unique_brokers():
    raw = _make_raw(
        [
            {"broker": "中信证券", "symbol": f"X{i:03d}.SH", "announced_at": f"2024-01-{i + 1:02d}"}
            for i in range(15)
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_unique_brokers=3))
    gate = result.coverage["gate"]
    assert gate["broker_reports_usable_for_features"] is False
    assert "too_few_brokers" in gate["reason"]


def test_gate_blocks_with_low_mean_credibility():
    raw = _make_raw(
        [
            {"broker": f"山寨{i}证券", "symbol": f"X{i}.SH", "announced_at": f"2024-01-{i + 1:02d}"}
            for i in range(15)
        ]
    )
    result = build_broker_reports(raw, config=BrokerReportConfig(min_mean_credibility=0.70))
    gate = result.coverage["gate"]
    assert gate["broker_reports_usable_for_features"] is False
    assert "mean_credibility" in gate["reason"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_emits_parquet_and_manifest(tmp_path):
    builder = BrokerReportBuilder(BrokerReportConfig(output_root=tmp_path))
    result = builder.write(builder.build(_good_batch(n=12)))
    assert (tmp_path / "silver" / "broker_reports" / "broker_reports.parquet").exists()
    assert (tmp_path / "manifests" / "broker_reports.json").exists()
    m = json.loads((tmp_path / "manifests" / "broker_reports.json").read_text())
    assert "broker_reports_usable_for_features" in m["extra"]["coverage_report"]["gate"]


# ---------------------------------------------------------------------------
# Feature attach
# ---------------------------------------------------------------------------

def test_apply_features_attaches_consensus_and_premium():
    reports = build_broker_reports(_good_batch(n=12)).frame
    panel = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-02-15", periods=4),
            "symbol": ["600000.SH", "600001.SH", "600002.SH", "600003.SH"],
        }
    )
    out = apply_broker_report_features(panel, reports)
    assert "broker_consensus_score" in out.columns
    assert "broker_target_premium" in out.columns


def test_apply_features_pit_safe_no_future_leak():
    reports = build_broker_reports(_good_batch(n=12)).frame
    # Force one report's available_at to be after the panel date
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2023-01-01")],  # pre-reports
            "symbol": ["600000.SH"],
        }
    )
    out = apply_broker_report_features(panel, reports)
    # The panel row predates all reports → consensus must be 0
    assert float(out.iloc[0]["broker_consensus_score"]) == 0.0


def test_apply_features_higher_credibility_brokers_dominate_consensus():
    reports = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "symbol": "X.SH",
                "broker": "中信",
                "broker_tier": "tier_1",
                "broker_credibility": 0.85,
                "announced_at": pd.Timestamp("2024-01-10"),
                "available_at": pd.Timestamp("2024-01-11"),
                "rating": "buy",
                "rating_change": "upgrade",
                "target_price": 50.0,
                "prev_target_price": np.nan,
                "target_price_pct_change": np.nan,
                "summary": "",
                "source": "test",
                "source_version": "v1",
                "fetched_at": pd.Timestamp("2024-01-11"),
            },
            {
                "event_id": "e2",
                "symbol": "X.SH",
                "broker": "山寨",
                "broker_tier": "tier_3",
                "broker_credibility": 0.50,
                "announced_at": pd.Timestamp("2024-01-10"),
                "available_at": pd.Timestamp("2024-01-11"),
                "rating": "sell",
                "rating_change": "downgrade",
                "target_price": 30.0,
                "prev_target_price": np.nan,
                "target_price_pct_change": np.nan,
                "summary": "",
                "source": "test",
                "source_version": "v1",
                "fetched_at": pd.Timestamp("2024-01-11"),
            },
        ]
    )
    panel = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2024-01-15")], "symbol": ["X.SH"]}
    )
    out = apply_broker_report_features(panel, reports)
    # consensus should lean positive (tier_1 buy outweighs tier_3 sell)
    assert float(out.iloc[0]["broker_consensus_score"]) > 0.0


def test_apply_features_empty_reports_returns_zero_columns():
    panel = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2024-02-15")], "symbol": ["A.SH"]}
    )
    out = apply_broker_report_features(panel, pd.DataFrame())
    assert "broker_consensus_score" in out.columns
    assert float(out.iloc[0]["broker_consensus_score"]) == 0.0


# ---------------------------------------------------------------------------
# Manifest gate
# ---------------------------------------------------------------------------

def test_overlay_helper_returns_none_when_gate_closed(tmp_path):
    closed = tmp_path / "closed.json"
    closed.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"broker_reports_usable_for_features": False}}}}
        ),
        encoding="utf-8",
    )
    reports = pd.DataFrame([{"event_id": "x"}])
    assert broker_reports_for_features(reports, closed) is None


def test_overlay_helper_returns_frame_when_gate_open(tmp_path):
    open_path = tmp_path / "open.json"
    open_path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"broker_reports_usable_for_features": True}}}}
        ),
        encoding="utf-8",
    )
    reports = pd.DataFrame([{"event_id": "x"}])
    assert broker_reports_for_features(reports, open_path) is not None


def test_overlay_helper_missing_inputs(tmp_path):
    assert broker_reports_for_features(None, tmp_path / "x.json") is None
    assert broker_reports_for_features(pd.DataFrame(), tmp_path / "x.json") is None
    assert broker_reports_for_features(pd.DataFrame([{"x": 1}]), None) is None
