"""Tests for the Stage 4.4 state-team inference data layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.state_team import (
    EVIDENCE_TYPES,
    INFERENCE_REQUIRED_COLUMNS,
    StateTeamInferenceBuilder,
    StateTeamInferenceConfig,
    apply_state_team_features,
    build_state_team_inference,
    infer_etf_concentrated_inflow,
    infer_post_crash_index_buying,
    infer_top10_holder_appearance,
    state_team_inference_for_features,
)


# ---------------------------------------------------------------------------
# Compliance — evidence_label is hard-coded "inferred"
# ---------------------------------------------------------------------------

def test_every_row_is_labelled_inferred():
    """The compliance posture is that we never claim "confirmed". Even if
    a caller passes evidence_label="confirmed" via extra_events, the
    builder rewrites it to "inferred".
    """
    rogue = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-02-05"),
                "evidence_type": "block_trade_match",
                "scope": "symbol",
                "scope_value": "A.SZ",
                "evidence_strength": 0.7,
                "evidence_label": "confirmed",   # NOT honoured
                "description": "should be relabelled",
            },
            {
                "trade_date": pd.Timestamp("2024-02-06"),
                "evidence_type": "block_trade_match",
                "scope": "symbol",
                "scope_value": "B.SZ",
                "evidence_strength": 0.7,
                "evidence_label": "official",   # also NOT honoured
                "description": "should be relabelled",
            },
            {
                "trade_date": pd.Timestamp("2024-02-07"),
                "evidence_type": "block_trade_match",
                "scope": "symbol",
                "scope_value": "C.SZ",
                "evidence_strength": 0.7,
                "evidence_label": "inferred",
                "description": "ok",
            },
        ]
    )
    result = build_state_team_inference(extra_events=rogue)
    assert (result.frame["evidence_label"] == "inferred").all()


# ---------------------------------------------------------------------------
# ETF concentrated inflow detector
# ---------------------------------------------------------------------------

def test_etf_inflow_above_threshold_triggers():
    flows = pd.DataFrame(
        {
            "510300.SH": [2.0, 8.0, 12.0, 3.0, 0.5],
        },
        index=pd.bdate_range("2024-02-05", periods=5),
    )
    events = infer_etf_concentrated_inflow(flows, config=StateTeamInferenceConfig(
        etf_concentrated_inflow_threshold_cny_bn=5.0
    ))
    # Two days above threshold (8.0 and 12.0)
    assert len(events) == 2
    assert all(e["evidence_type"] == "etf_concentrated_inflow" for e in events)
    assert all(e["evidence_label"] == "inferred" for e in events)


def test_etf_inflow_below_threshold_no_event():
    flows = pd.DataFrame(
        {"510300.SH": [1.0, 2.0, 3.0]},
        index=pd.bdate_range("2024-02-05", periods=3),
    )
    events = infer_etf_concentrated_inflow(flows, config=StateTeamInferenceConfig(
        etf_concentrated_inflow_threshold_cny_bn=5.0
    ))
    assert events == []


def test_etf_inflow_strength_scales_with_magnitude():
    flows = pd.DataFrame(
        {"510300.SH": [5.0, 50.0]},  # at threshold vs 10x threshold
        index=pd.bdate_range("2024-02-05", periods=2),
    )
    events = infer_etf_concentrated_inflow(flows, config=StateTeamInferenceConfig(
        etf_concentrated_inflow_threshold_cny_bn=5.0
    ))
    assert events[1]["evidence_strength"] > events[0]["evidence_strength"]


# ---------------------------------------------------------------------------
# Post-crash buying detector
# ---------------------------------------------------------------------------

def test_post_crash_buying_fires_after_5d_crash_with_etf_inflow():
    # 30 day benchmark with a -10% drop spanning days 10-15
    dates = pd.bdate_range("2024-02-05", periods=30)
    rets = np.zeros(30)
    rets[10:15] = -0.025  # cumulative ~-12% over 5 days
    bench = pd.Series(rets, index=dates)
    flows = pd.DataFrame(
        {"510300.SH": [0.0] * 30},
        index=dates,
    )
    # Day 16 (right after crash): heavy ETF inflow
    flows.iloc[16] = 20.0
    events = infer_post_crash_index_buying(
        bench, flows,
        config=StateTeamInferenceConfig(
            post_crash_5d_threshold=-0.08,
            post_crash_etf_inflow_threshold_cny_bn=10.0,
        ),
    )
    assert any(e["evidence_type"] == "post_crash_index_buying" for e in events)
    assert all(e["evidence_strength"] >= 0.60 for e in events)


def test_post_crash_no_event_when_no_crash():
    dates = pd.bdate_range("2024-02-05", periods=20)
    bench = pd.Series([0.001] * 20, index=dates)
    flows = pd.DataFrame({"510300.SH": [15.0] * 20}, index=dates)  # heavy inflow but no crash
    events = infer_post_crash_index_buying(bench, flows)
    assert events == []


# ---------------------------------------------------------------------------
# Top-10 holder detector
# ---------------------------------------------------------------------------

def test_top10_holder_matches_state_team_keywords():
    holders = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "601398.SH", "holder_name": "中央汇金资产管理有限责任公司", "share_pct": 35.42},
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "600519.SH", "holder_name": "贵州茅台集团", "share_pct": 62.0},  # not state team
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "601628.SH", "holder_name": "全国社保基金理事会", "share_pct": 5.5},
        ]
    )
    events = infer_top10_holder_appearance(holders)
    assert len(events) == 2
    assert any("601398.SH" in e["scope_value"] for e in events)
    assert any("601628.SH" in e["scope_value"] for e in events)


def test_top10_holder_strength_scales_with_stake():
    holders = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "A.SH", "holder_name": "中央汇金", "share_pct": 1.0},
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "B.SH", "holder_name": "中央汇金", "share_pct": 10.0},
        ]
    )
    events = sorted(infer_top10_holder_appearance(holders), key=lambda e: e["scope_value"])
    assert events[1]["evidence_strength"] > events[0]["evidence_strength"]


def test_top10_holder_available_at_uses_45bd_lag():
    """Quarterly filings publish ~45 business days after quarter-end.
    The inference must respect that disclosure lag.
    """
    holders = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "A.SH", "holder_name": "汇金", "share_pct": 5.0},
        ]
    )
    events = infer_top10_holder_appearance(holders)
    available = events[0]["available_at"]
    assert available > pd.Timestamp("2024-04-30")  # at least 30 days later
    assert available < pd.Timestamp("2024-07-01")  # not absurdly late


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------

def test_builder_with_all_three_sources_produces_required_schema():
    dates = pd.bdate_range("2024-02-05", periods=20)
    etf = pd.DataFrame({"510300.SH": np.r_[np.full(10, 1.0), [12.0] * 5, np.full(5, 1.0)]}, index=dates)
    bench = pd.Series(np.r_[np.zeros(8), [-0.03] * 5, np.zeros(7)], index=dates)
    holders = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-03-31"), "symbol": "601398.SH",
             "holder_name": "中央汇金", "share_pct": 35.0}
        ]
    )
    result = build_state_team_inference(
        etf_flows=etf, benchmark_returns=bench, top10_holders=holders,
        config=StateTeamInferenceConfig(min_events=1, min_mean_strength=0.3),
    )
    assert set(result.frame.columns) == set(INFERENCE_REQUIRED_COLUMNS)
    assert (result.frame["evidence_label"] == "inferred").all()
    types = set(result.frame["evidence_type"])
    assert "etf_concentrated_inflow" in types
    assert "top10_holder_appearance" in types


def test_builder_no_inputs_yields_closed_gate():
    result = build_state_team_inference()
    assert result.frame.empty
    gate = result.coverage["gate"]
    assert gate["state_team_inference_usable_for_features"] is False
    assert gate["reason"] == "no_events"


def test_builder_dedupes_by_event_id():
    flows = pd.DataFrame(
        {"510300.SH": [12.0, 12.0]},  # same value, but each row has its own date
        index=pd.bdate_range("2024-02-05", periods=2),
    )
    # The same day repeated → de-duped
    flows_dup = pd.concat([flows, flows]).drop_duplicates()  # idempotent
    result = build_state_team_inference(
        etf_flows=flows_dup,
        config=StateTeamInferenceConfig(min_events=1, min_mean_strength=0.3),
    )
    n_rows = len(result.frame)
    n_unique = result.frame["event_id"].nunique()
    assert n_rows == n_unique


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def test_gate_opens_with_strong_signals():
    flows = pd.DataFrame(
        {"510300.SH": [15.0, 18.0, 12.0, 20.0]},
        index=pd.bdate_range("2024-02-05", periods=4),
    )
    result = build_state_team_inference(
        etf_flows=flows,
        config=StateTeamInferenceConfig(min_events=3, min_mean_strength=0.30),
    )
    gate = result.coverage["gate"]
    assert gate["state_team_inference_usable_for_features"] is True
    assert gate["reason"] == "passed"


def test_gate_blocks_with_too_few_events():
    flows = pd.DataFrame({"510300.SH": [12.0]}, index=[pd.Timestamp("2024-02-05")])
    result = build_state_team_inference(
        etf_flows=flows,
        config=StateTeamInferenceConfig(min_events=3),
    )
    gate = result.coverage["gate"]
    assert gate["state_team_inference_usable_for_features"] is False
    assert "too_few_events" in gate["reason"]


def test_gate_blocks_with_low_mean_strength():
    flows = pd.DataFrame(
        {"510300.SH": [5.01, 5.02, 5.03, 5.04]},  # just above threshold → low strength
        index=pd.bdate_range("2024-02-05", periods=4),
    )
    result = build_state_team_inference(
        etf_flows=flows,
        config=StateTeamInferenceConfig(min_events=3, min_mean_strength=0.80),
    )
    gate = result.coverage["gate"]
    assert gate["state_team_inference_usable_for_features"] is False
    assert "mean_strength" in gate["reason"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_emits_parquet_and_compliant_manifest(tmp_path):
    flows = pd.DataFrame(
        {"510300.SH": [15.0, 18.0, 12.0, 20.0]},
        index=pd.bdate_range("2024-02-05", periods=4),
    )
    builder = StateTeamInferenceBuilder(StateTeamInferenceConfig(
        output_root=tmp_path, min_events=2, min_mean_strength=0.30,
    ))
    builder.write(builder.build(etf_flows=flows))
    parquet = tmp_path / "silver" / "state_team_inference" / "state_team_inference.parquet"
    manifest = tmp_path / "manifests" / "state_team_inference.json"
    assert parquet.exists() and manifest.exists()
    m = json.loads(manifest.read_text())
    assert "compliance_note" in m  # must surface the "inferred" posture
    assert "inferred" in m["compliance_note"].lower()


# ---------------------------------------------------------------------------
# Feature attach
# ---------------------------------------------------------------------------

def test_apply_features_adds_state_team_signal_column():
    panel = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-02-20", periods=5),
            "symbol": ["A.SH"] * 5,
        }
    )
    events = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-02-05"),
                "available_at": pd.Timestamp("2024-02-06"),
                "evidence_type": "etf_concentrated_inflow",
                "evidence_label": "inferred",
                "evidence_strength": 0.70,
                "scope": "index_wide",
                "scope_value": "510300.SH",
            }
        ]
    )
    out = apply_state_team_features(panel, events)
    assert "state_team_signal" in out.columns
    assert "state_team_evidence_label" in out.columns
    assert (out["state_team_evidence_label"] == "inferred").all()
    assert (out["state_team_signal"] > 0).all()


def test_apply_features_pit_safe_no_future_leak():
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-20")],
            "symbol": ["A.SH", "A.SH"],
        }
    )
    events = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-02-10"),
                "available_at": pd.Timestamp("2024-02-11"),
                "evidence_type": "etf_concentrated_inflow",
                "evidence_label": "inferred",
                "evidence_strength": 0.7,
                "scope": "index_wide",
                "scope_value": "510300.SH",
            }
        ]
    )
    out = apply_state_team_features(panel, events).sort_values("trade_date")
    # Feb 1 row should be 0 (before event); Feb 20 row should be > 0
    assert float(out.iloc[0]["state_team_signal"]) == 0.0
    assert float(out.iloc[1]["state_team_signal"]) > 0.0


def test_apply_features_no_events_returns_zero_column():
    panel = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2024-02-20")], "symbol": ["A.SH"]}
    )
    out = apply_state_team_features(panel, pd.DataFrame())
    assert "state_team_signal" in out.columns
    assert (out["state_team_signal"] == 0.0).all()


def test_apply_features_symbol_scoped_event_only_hits_matching_symbol():
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01")],
            "symbol": ["A.SH", "B.SH"],
        }
    )
    events = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-03-31"),
                "available_at": pd.Timestamp("2024-05-15"),  # T+45 lag
                "evidence_type": "top10_holder_appearance",
                "evidence_label": "inferred",
                "evidence_strength": 0.8,
                "scope": "symbol",
                "scope_value": "A.SH",
            }
        ]
    )
    out = apply_state_team_features(panel, events)
    by_sym = out.set_index("symbol")["state_team_signal"]
    assert by_sym.loc["A.SH"] == 0.0  # available_at 2024-05-15 > trade_date 2024-04-01
    # B.SH has trade_date 2024-06-01 but event scope is A.SH, not B → should remain 0
    assert by_sym.loc["B.SH"] == 0.0


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def test_overlay_helper_returns_none_when_gate_closed(tmp_path):
    closed = tmp_path / "closed.json"
    closed.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"state_team_inference_usable_for_features": False}}}}
        ),
        encoding="utf-8",
    )
    events = pd.DataFrame([{"event_id": "x"}])
    assert state_team_inference_for_features(events, closed) is None


def test_overlay_helper_returns_events_when_gate_open(tmp_path):
    open_path = tmp_path / "open.json"
    open_path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"state_team_inference_usable_for_features": True}}}}
        ),
        encoding="utf-8",
    )
    events = pd.DataFrame([{"event_id": "x"}])
    assert state_team_inference_for_features(events, open_path) is not None


def test_overlay_helper_missing_inputs(tmp_path):
    assert state_team_inference_for_features(None, tmp_path / "x.json") is None
    assert state_team_inference_for_features(pd.DataFrame(), tmp_path / "x.json") is None
    assert state_team_inference_for_features(pd.DataFrame([{"x": 1}]), None) is None
