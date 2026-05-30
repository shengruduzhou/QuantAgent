"""Canonical evidence schema + PIT lint tests.

Covers the four v8.2 silver builders (policy / bond / broker /
state_team) → canonical EvidenceRecord adapter pipeline. Each test
builds a realistic source frame, runs it through the adapter, and
asserts:

* required columns are present
* scoring fields land in their declared ranges
* PIT contract holds: ``available_at`` ≥ ``publish_time``, never null
* ``audit_trace`` carries the adapter name + source version

The final test concats every adapter output and runs
:func:`validate_pit_safety` — this is the cross-source PIT lint that
:doc:`docs/v8_gap_report.md` flagged as missing in P0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.evidence import (
    CANONICAL_EVIDENCE_COLUMNS,
    CANONICAL_SOURCE_TYPES,
    EvidenceRecord,
    bond_flows_to_evidence,
    broker_reports_to_evidence,
    policy_events_to_evidence,
    state_team_events_to_evidence,
    to_canonical_evidence_frame,
    validate_pit_safety,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal silver-shape DataFrames matching each builder schema
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_events_silver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "p_event_a",
                "source": "csrc",
                "url": "https://csrc.gov.cn/p1",
                "announced_at": pd.Timestamp("2024-03-01 09:30"),
                "effective_at": pd.Timestamp("2024-03-05"),
                "fetched_at": pd.Timestamp("2024-03-01 10:00"),
                "available_at": pd.Timestamp("2024-03-01 10:00"),
                "title": "关于支持新能源汽车产业的指导意见",
                "body_summary": "扩大补贴范围",
                "themes": ["tech_innovation", "industrial"],
                "sectors_hint": ["NEV"],
                "policy_strength": 0.7,
                "source_version": "csrc_v1",
            },
            {
                "event_id": "p_event_b",
                "source": "pboc",
                "url": "https://pbc.gov.cn/p2",
                "announced_at": pd.Timestamp("2024-04-15"),
                "effective_at": pd.Timestamp("2024-04-15"),
                "fetched_at": pd.Timestamp("2024-04-15 16:00"),
                "available_at": pd.Timestamp("2024-04-15 16:00"),
                "title": "下调存款准备金率",
                "body_summary": "",
                "themes": ["monetary"],
                "sectors_hint": [],
                "policy_strength": 0.4,
                "source_version": "pboc_v1",
            },
        ]
    )


@pytest.fixture
def bond_flows_silver() -> pd.DataFrame:
    base = pd.Timestamp("2024-05-01")
    rows = []
    for i in range(3):
        d = base + pd.tseries.offsets.BDay(i)
        rows.append(
            {
                "trade_date": d,
                "available_at": d + pd.tseries.offsets.BDay(1),
                "yield_1y": 1.80 + 0.01 * i,
                "yield_5y": 2.30 + 0.005 * i,
                "yield_10y": 2.55 + 0.005 * i,
                "spread_10y_1y": 0.75,
                "spread_10y_3m": 0.85,
                "credit_spread_aa": 1.10,
                "credit_spread_aaa_aa": 0.40,
                "dr007": 1.95,
                "bond_fund_flow": 30.0 - 10.0 * i,  # 30, 20, 10 亿
                "source": "wind",
                "source_version": "wind_v1",
                "fetched_at": d + pd.tseries.offsets.BDay(1),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def broker_reports_silver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "br_a",
                "symbol": "600519.SH",
                "broker": "中信证券",
                "broker_tier": "tier_1",
                "announced_at": pd.Timestamp("2024-06-01"),
                "available_at": pd.Timestamp("2024-06-02"),
                "rating": "buy",
                "rating_change": "upgrade",
                "target_price": 1900.0,
                "prev_target_price": 1600.0,
                "target_price_pct_change": 0.1875,
                "summary": "公司护城河强",
                "broker_credibility": 0.85,
                "source": "wind",
                "source_version": "wind_v1",
                "fetched_at": pd.Timestamp("2024-06-02 08:00"),
            },
            {
                "event_id": "br_b",
                "symbol": "000001.SZ",
                "broker": "未知小券商",
                "broker_tier": "tier_3",
                "announced_at": pd.Timestamp("2024-06-10"),
                "available_at": pd.Timestamp("2024-06-11"),
                "rating": "sell",
                "rating_change": "downgrade",
                "target_price": 8.0,
                "prev_target_price": 10.0,
                "target_price_pct_change": -0.2,
                "summary": "下调评级",
                "broker_credibility": 0.50,
                "source": "manual",
                "source_version": "manual_v1",
                "fetched_at": pd.Timestamp("2024-06-11 09:00"),
            },
        ]
    )


@pytest.fixture
def state_team_silver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "st_a",
                "trade_date": pd.Timestamp("2024-07-15"),
                "available_at": pd.Timestamp("2024-07-16"),
                "evidence_type": "etf_concentrated_inflow",
                "evidence_label": "inferred",
                "evidence_strength": 0.65,
                "scope": "index_wide",
                "scope_value": "510300.SH",
                "description": "510300.SH net inflow 25.00亿",
                "source": "wind",
                "source_version": "wind_v1",
                "fetched_at": pd.Timestamp("2024-07-16 09:00"),
            },
            {
                "event_id": "st_b",
                "trade_date": pd.Timestamp("2024-07-20"),
                "available_at": pd.Timestamp("2024-09-15"),  # +45 BDay
                "evidence_type": "top10_holder_appearance",
                "evidence_label": "inferred",
                "evidence_strength": 0.80,
                "scope": "symbol",
                "scope_value": "600519.SH",
                "description": "中央汇金 2.50%",
                "source": "wind",
                "source_version": "wind_v1",
                "fetched_at": pd.Timestamp("2024-09-15 10:00"),
            },
        ]
    )


# ---------------------------------------------------------------------------
# EvidenceRecord dataclass
# ---------------------------------------------------------------------------

def test_evidence_record_defaults_and_to_dict():
    r = EvidenceRecord(
        evidence_id="x_1",
        source_name="csrc",
        source_type="policy",
        publish_time=pd.Timestamp("2024-01-01"),
        available_at=pd.Timestamp("2024-01-02"),
        entity_type="policy_event",
    )
    d = r.to_dict()
    assert d["evidence_id"] == "x_1"
    assert d["lag_window_candidates"] == [1, 5, 20, 60, 120]
    assert d["entities"] == []
    assert d["sentiment_score"] == 0.0
    assert d["audit_trace"] == {}


# ---------------------------------------------------------------------------
# Adapter: policy
# ---------------------------------------------------------------------------

def test_policy_adapter_produces_canonical_columns(policy_events_silver):
    out = policy_events_to_evidence(policy_events_silver)
    assert list(out.columns) == list(CANONICAL_EVIDENCE_COLUMNS)
    assert len(out) == 2
    assert (out["source_type"] == "policy").all()
    assert "policy" in out.iloc[0]["evidence_id"]


def test_policy_adapter_maps_strength_to_direction_and_confidence(policy_events_silver):
    out = policy_events_to_evidence(policy_events_silver)
    row_csrc = out[out["audit_trace"].map(lambda d: d["source_event_id"]) == "p_event_a"].iloc[0]
    assert row_csrc["policy_direction_score"] == pytest.approx(0.7)
    assert row_csrc["confidence"] == pytest.approx(0.7)
    assert row_csrc["sentiment_score"] == 0.0
    assert row_csrc["capital_flow_direction_score"] == 0.0


def test_policy_adapter_preserves_entities(policy_events_silver):
    out = policy_events_to_evidence(policy_events_silver)
    nev_row = out[out["audit_trace"].map(lambda d: d["source_event_id"]) == "p_event_a"].iloc[0]
    assert "NEV" in nev_row["entities"]
    assert "tech_innovation" in nev_row["entities"]


def test_policy_adapter_audit_trace_records_adapter_name(policy_events_silver):
    out = policy_events_to_evidence(policy_events_silver)
    trace = out.iloc[0]["audit_trace"]
    assert trace["adapter"] == "policy_events_to_evidence"
    assert trace["source_version"] != "unknown"


def test_policy_adapter_empty_input_returns_empty_canonical_frame():
    out = policy_events_to_evidence(pd.DataFrame())
    assert list(out.columns) == list(CANONICAL_EVIDENCE_COLUMNS)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Adapter: bond
# ---------------------------------------------------------------------------

def test_bond_adapter_emits_one_row_per_trade_date(bond_flows_silver):
    out = bond_flows_to_evidence(bond_flows_silver)
    assert len(out) == len(bond_flows_silver)
    assert (out["source_type"] == "bond").all()


def test_bond_adapter_capital_flow_direction_tracks_fund_flow_sign(bond_flows_silver):
    out = bond_flows_to_evidence(bond_flows_silver)
    # Three rows: flow=30 (positive), 20, 10 — all positive → all positive scores
    assert (out["capital_flow_direction_score"] > 0).all()
    # Highest flow → highest score
    order = out.sort_values("publish_time")["capital_flow_direction_score"].tolist()
    assert order[0] > order[-1]


def test_bond_adapter_policy_score_in_signed_range(bond_flows_silver):
    out = bond_flows_to_evidence(bond_flows_silver)
    assert (out["policy_direction_score"] >= -1.0).all()
    assert (out["policy_direction_score"] <= 1.0).all()


def test_bond_adapter_skips_rows_without_publish_time():
    bad = pd.DataFrame(
        [
            {"trade_date": None, "available_at": None, "yield_1y": 1.0, "source": "wind"},
            {
                "trade_date": pd.Timestamp("2024-01-01"),
                "available_at": pd.Timestamp("2024-01-02"),
                "yield_1y": 1.0,
                "source": "wind",
            },
        ]
    )
    out = bond_flows_to_evidence(bad)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Adapter: broker
# ---------------------------------------------------------------------------

def test_broker_adapter_buy_upgrade_yields_positive_sentiment(broker_reports_silver):
    out = broker_reports_to_evidence(broker_reports_silver)
    buy_row = out[out["audit_trace"].map(lambda d: d["broker"]) == "中信证券"].iloc[0]
    assert buy_row["sentiment_score"] == pytest.approx(1.0)
    # tier_1 credibility 0.85 + upgrade bump 0.10 = 0.95
    assert buy_row["confidence"] == pytest.approx(0.95)


def test_broker_adapter_sell_downgrade_yields_negative_sentiment(broker_reports_silver):
    out = broker_reports_to_evidence(broker_reports_silver)
    sell_row = out[out["audit_trace"].map(lambda d: d["broker"]) == "未知小券商"].iloc[0]
    assert sell_row["sentiment_score"] == pytest.approx(-1.0)
    # tier_3 credibility 0.50 + downgrade penalty -0.10 = 0.40
    assert sell_row["confidence"] == pytest.approx(0.40)


def test_broker_adapter_symbol_lands_in_entities(broker_reports_silver):
    out = broker_reports_to_evidence(broker_reports_silver)
    assert any("600519.SH" in row["entities"] for _, row in out.iterrows())


# ---------------------------------------------------------------------------
# Adapter: state_team
# ---------------------------------------------------------------------------

def test_state_team_adapter_records_inferred_label_in_audit(state_team_silver):
    out = state_team_events_to_evidence(state_team_silver)
    assert (out["source_type"] == "state_team_inference").all()
    for trace in out["audit_trace"]:
        assert trace["evidence_label"] == "inferred"


def test_state_team_adapter_strength_becomes_capital_flow_score(state_team_silver):
    out = state_team_events_to_evidence(state_team_silver)
    assert out["capital_flow_direction_score"].tolist() == pytest.approx([0.65, 0.80])
    assert out["confidence"].tolist() == pytest.approx([0.65, 0.80])


def test_state_team_adapter_scope_propagates_to_entities(state_team_silver):
    out = state_team_events_to_evidence(state_team_silver)
    index_row = out[out["entity_type"] == "etf_concentrated_inflow"].iloc[0]
    symbol_row = out[out["entity_type"] == "top10_holder_appearance"].iloc[0]
    assert "index:510300.SH" in index_row["entities"]
    assert "600519.SH" in symbol_row["entities"]


# ---------------------------------------------------------------------------
# Aggregator + PIT lint
# ---------------------------------------------------------------------------

def test_aggregator_concats_and_dedups(
    policy_events_silver, bond_flows_silver, broker_reports_silver, state_team_silver
):
    out = to_canonical_evidence_frame(
        policy_events=policy_events_silver,
        bond_flows=bond_flows_silver,
        broker_reports=broker_reports_silver,
        state_team_events=state_team_silver,
    )
    expected = (
        len(policy_events_silver) + len(bond_flows_silver)
        + len(broker_reports_silver) + len(state_team_silver)
    )
    assert len(out) == expected
    assert out["available_at"].is_monotonic_increasing
    # Every source_type should fall in the canonical vocabulary
    assert set(out["source_type"]).issubset(set(CANONICAL_SOURCE_TYPES))


def test_aggregator_with_no_inputs_returns_empty_frame():
    out = to_canonical_evidence_frame()
    assert list(out.columns) == list(CANONICAL_EVIDENCE_COLUMNS)
    assert len(out) == 0


def test_pit_lint_passes_on_clean_v8_silver_outputs(
    policy_events_silver, bond_flows_silver, broker_reports_silver, state_team_silver
):
    out = to_canonical_evidence_frame(
        policy_events=policy_events_silver,
        bond_flows=bond_flows_silver,
        broker_reports=broker_reports_silver,
        state_team_events=state_team_silver,
    )
    report = validate_pit_safety(out)
    assert report.passed, report.to_dict()
    assert report.n_missing_available_at == 0
    assert report.n_available_before_publish == 0


def test_pit_lint_catches_inverted_available_at():
    bad_policy = pd.DataFrame(
        [
            {
                "event_id": "bad",
                "source": "csrc",
                "url": "u",
                "announced_at": pd.Timestamp("2024-05-01"),
                "effective_at": pd.Timestamp("2024-05-01"),
                "fetched_at": pd.Timestamp("2024-04-30"),  # before announce
                "available_at": pd.Timestamp("2024-04-29"),  # leak!
                "title": "x",
                "body_summary": "",
                "themes": [],
                "sectors_hint": [],
                "policy_strength": 0.5,
                "source_version": "v1",
            }
        ]
    )
    out = policy_events_to_evidence(bad_policy)
    report = validate_pit_safety(out)
    assert not report.passed
    assert report.n_available_before_publish == 1
    assert report.sample_violations[0]["reason"] == "available_at_before_publish"


def test_pit_lint_flags_future_publish_when_as_of_supplied(state_team_silver):
    out = state_team_events_to_evidence(state_team_silver)
    # state_team_silver has publish_times in July 2024; as_of in Jan → all future
    report = validate_pit_safety(out, as_of=pd.Timestamp("2024-01-01"))
    assert report.n_future_publish == len(out)
    # but the hard contract still holds
    assert report.passed


def test_pit_lint_counts_by_source_type(
    policy_events_silver, bond_flows_silver, broker_reports_silver, state_team_silver
):
    out = to_canonical_evidence_frame(
        policy_events=policy_events_silver,
        bond_flows=bond_flows_silver,
        broker_reports=broker_reports_silver,
        state_team_events=state_team_silver,
    )
    report = validate_pit_safety(out)
    assert report.by_source_type["policy"] == 2
    assert report.by_source_type["bond"] == 3
    assert report.by_source_type["broker_view"] == 2
    assert report.by_source_type["state_team_inference"] == 2
