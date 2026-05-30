"""Tests for the Stage 5.2 news credibility weighting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.credibility import (
    SOURCE_TIER_TABLE,
    apply_credibility_column,
    apply_credibility_weight_to_strength,
    lookup_source_credibility,
    lookup_source_tier,
)


# ---------------------------------------------------------------------------
# Tier lookup
# ---------------------------------------------------------------------------

def test_official_source_tier():
    assert lookup_source_tier("中国证监会") == "official"
    assert lookup_source_tier("中国人民银行") == "official"
    assert lookup_source_tier("https://csrc.gov.cn/p1") == "official"


def test_state_media_tier():
    assert lookup_source_tier("新华社") == "state_media"
    assert lookup_source_tier("人民日报海外版") == "state_media"


def test_top_financial_tier():
    assert lookup_source_tier("财新周刊") == "top_financial"
    assert lookup_source_tier("21世纪经济报道") == "top_financial"


def test_mainstream_financial_tier():
    assert lookup_source_tier("证券时报") == "mainstream_financial"
    assert lookup_source_tier("上海证券报") == "mainstream_financial"


def test_online_financial_tier():
    assert lookup_source_tier("财联社") == "online_financial"
    assert lookup_source_tier("华尔街见闻") == "online_financial"


def test_industry_data_tier():
    assert lookup_source_tier("同花顺") == "industry_data"
    assert lookup_source_tier("东方财富网") == "industry_data"
    assert lookup_source_tier("xueqiu.com") == "industry_data"


def test_social_media_tier():
    assert lookup_source_tier("某公众号") == "social_media"
    assert lookup_source_tier("微博") == "social_media"


def test_unknown_source_returns_unknown_tier():
    assert lookup_source_tier("xyz_unknown_domain") == "unknown"
    assert lookup_source_tier(None) == "unknown"
    assert lookup_source_tier("") == "unknown"


# ---------------------------------------------------------------------------
# Credibility band ordering
# ---------------------------------------------------------------------------

def test_credibility_band_ordering():
    """Official > state_media > top_financial > mainstream > online > industry > social > unknown."""
    pairs = [
        ("中国证监会", "新华社"),
        ("新华社", "财新"),
        ("财新", "证券时报"),
        ("证券时报", "财联社"),
        ("财联社", "同花顺"),
        ("同花顺", "公众号"),
    ]
    for higher, lower in pairs:
        assert lookup_source_credibility(higher) > lookup_source_credibility(lower)


def test_unknown_falls_between_industry_and_social():
    """Unknown defaults to 0.50 — better than social, worse than industry."""
    unknown_score = lookup_source_credibility("nonexistent_outlet")
    assert unknown_score == 0.50


# ---------------------------------------------------------------------------
# Apply column
# ---------------------------------------------------------------------------

def test_apply_credibility_column_adds_score_and_tier():
    events = pd.DataFrame(
        [
            {"source": "证监会公告", "evidence": "x"},
            {"source": "财联社", "evidence": "y"},
            {"source": "未知来源", "evidence": "z"},
        ]
    )
    out = apply_credibility_column(events)
    assert "source_credibility" in out.columns
    assert "source_tier" in out.columns
    by_src = out.set_index("source")
    assert by_src.loc["证监会公告", "source_tier"] == "official"
    assert by_src.loc["财联社", "source_tier"] == "online_financial"
    assert by_src.loc["未知来源", "source_tier"] == "unknown"


def test_apply_credibility_column_does_not_mutate_input():
    events = pd.DataFrame([{"source": "证监会", "x": 1}])
    out = apply_credibility_column(events)
    assert "source_credibility" not in events.columns
    assert "source_credibility" in out.columns


def test_apply_credibility_column_missing_source_col_uses_unknown():
    events = pd.DataFrame([{"some_other_col": 1}])
    out = apply_credibility_column(events)
    assert out.iloc[0]["source_credibility"] == 0.50
    assert out.iloc[0]["source_tier"] == "unknown"


def test_apply_credibility_column_empty_frame():
    out = apply_credibility_column(pd.DataFrame())
    assert out.empty


def test_apply_credibility_column_custom_output_names():
    events = pd.DataFrame([{"src": "证监会"}])
    out = apply_credibility_column(events, source_col="src", out_col="cred", tier_col="tier")
    assert "cred" in out.columns
    assert "tier" in out.columns


# ---------------------------------------------------------------------------
# Credibility-weighted strength
# ---------------------------------------------------------------------------

def test_credibility_weighting_multiplies_strength():
    events = pd.DataFrame(
        [
            {"source": "证监会", "evidence_strength": 0.80},  # cred=1.00 → 0.80
            {"source": "财联社", "evidence_strength": 0.80},  # cred=0.70 → 0.56
            {"source": "某公众号", "evidence_strength": 0.80},  # cred=0.40 → 0.32
        ]
    )
    out = apply_credibility_weight_to_strength(events)
    weights = out.set_index("source")["credibility_weighted_strength"]
    assert weights.loc["证监会"] == pytest.approx(0.80)
    assert weights.loc["财联社"] == pytest.approx(0.56)
    assert weights.loc["某公众号"] == pytest.approx(0.32)


def test_credibility_weighting_clips_to_unit_interval():
    events = pd.DataFrame(
        [
            {"source": "证监会", "evidence_strength": 1.5},  # would be 1.5 → clipped to 1.0
            {"source": "未知", "evidence_strength": -0.3},   # negative → clipped to 0
        ]
    )
    out = apply_credibility_weight_to_strength(events)
    weights = out.set_index("source")["credibility_weighted_strength"]
    assert weights.loc["证监会"] == 1.0
    assert weights.loc["未知"] == 0.0


def test_credibility_weighting_missing_strength_raises():
    events = pd.DataFrame([{"source": "证监会"}])
    with pytest.raises(ValueError, match="missing strength column"):
        apply_credibility_weight_to_strength(events)


def test_credibility_weighting_empty_frame():
    out = apply_credibility_weight_to_strength(pd.DataFrame())
    assert out.empty


# ---------------------------------------------------------------------------
# Integration with broker_reports (5.1) — verifies the weighting playstable
# ---------------------------------------------------------------------------

def test_can_weight_broker_reports_by_source_credibility():
    from quantagent.data.broker import build_broker_reports

    raw = pd.DataFrame(
        [
            {"broker": "中信证券", "symbol": "A.SH", "announced_at": "2024-01-15", "source": "证监会披露"},
            {"broker": "中信证券", "symbol": "B.SH", "announced_at": "2024-01-15", "source": "微博营业部"},
        ]
    )
    reports = build_broker_reports(raw, config=__import__("quantagent.data.broker", fromlist=["BrokerReportConfig"]).BrokerReportConfig(
        min_events=1, min_unique_brokers=1,
    )).frame
    # broker_credibility from 5.1 represents the BROKER tier (tier_1 here).
    # Now overlay source credibility — should differ between the two rows.
    out = apply_credibility_column(reports)
    by_sym = out.set_index("symbol")
    # The 证监会披露 row should have higher source_credibility than the
    # 微博 row, regardless of broker tier.
    assert by_sym.loc["A.SH", "source_credibility"] > by_sym.loc["B.SH", "source_credibility"]
