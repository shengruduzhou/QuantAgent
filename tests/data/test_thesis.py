"""Capital-flow thesis builder + validation loop tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.evidence import to_canonical_evidence_frame
from quantagent.data.thesis import (
    CAPITAL_FLOW_THESIS_COLUMNS,
    CapitalFlowThesis,
    CapitalFlowThesisBuilder,
    CapitalFlowThesisConfig,
    THESIS_VALIDATION_STATES,
    ThesisValidationConfig,
    build_capital_flow_theses,
    theses_to_frame,
    validate_theses,
    validate_thesis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy_silver(sector: str, n: int, strength: float = 0.7) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01")
    return pd.DataFrame(
        [
            {
                "event_id": f"p_{sector}_{i}",
                "source": "csrc",
                "url": f"https://csrc.gov.cn/{sector}/{i}",
                "announced_at": base + pd.Timedelta(days=i),
                "effective_at": base + pd.Timedelta(days=i),
                "fetched_at": base + pd.Timedelta(days=i, hours=1),
                "available_at": base + pd.Timedelta(days=i, hours=1),
                "title": f"政策{i}",
                "body_summary": "",
                "themes": ["tech_innovation"],
                "sectors_hint": [sector],
                "policy_strength": strength,
                "source_version": "v1",
            }
            for i in range(n)
        ]
    )


def _broker_silver(symbol: str, n: int, rating: str = "buy") -> pd.DataFrame:
    base = pd.Timestamp("2024-03-05")
    return pd.DataFrame(
        [
            {
                "event_id": f"br_{symbol}_{i}",
                "symbol": symbol,
                "broker": "中信证券",
                "broker_tier": "tier_1",
                "announced_at": base + pd.Timedelta(days=i),
                "available_at": base + pd.Timedelta(days=i, hours=1),
                "rating": rating,
                "rating_change": "upgrade" if rating == "buy" else "downgrade",
                "target_price": 100.0,
                "prev_target_price": 80.0,
                "target_price_pct_change": 0.25,
                "summary": "",
                "broker_credibility": 0.85,
                "source": "wind",
                "source_version": "v1",
                "fetched_at": base + pd.Timedelta(days=i, hours=2),
            }
            for i in range(n)
        ]
    )


def _build_evidence(*, sectors: list[tuple[str, int]] | None = None,
                     brokers: list[tuple[str, int, str]] | None = None) -> pd.DataFrame:
    pol = pd.concat(
        [_policy_silver(s, n) for s, n in (sectors or [])],
        ignore_index=True,
    ) if sectors else None
    br = pd.concat(
        [_broker_silver(sym, n, rating) for sym, n, rating in (brokers or [])],
        ignore_index=True,
    ) if brokers else None
    return to_canonical_evidence_frame(policy_events=pol, broker_reports=br)


# ---------------------------------------------------------------------------
# Builder: aggregation
# ---------------------------------------------------------------------------

def test_thesis_builder_emits_one_thesis_per_unique_direction():
    ev = _build_evidence(sectors=[("Semi", 3), ("RealEstate", 3)])
    theses = build_capital_flow_theses(ev)
    directions = {(t.direction_kind, t.direction_value) for t in theses}
    # tech_innovation (theme), Semi/RealEstate (sectors_hint→theme)
    assert ("theme", "Semi") in directions or ("theme", "tech_innovation") in directions
    assert len(theses) >= 2


def test_thesis_builder_respects_min_supporting():
    ev = _build_evidence(sectors=[("Semi", 1)])  # only 1 evidence record
    theses = build_capital_flow_theses(ev)
    # Default min_supporting = 2 → no thesis
    assert theses == []


def test_thesis_builder_signs_match_direction_score():
    ev = _build_evidence(brokers=[("600519.SH", 3, "buy")])
    theses = build_capital_flow_theses(ev)
    assert any(t.thesis_sign > 0 for t in theses)


def test_thesis_builder_rejects_low_confidence_thesis():
    # Two very weak policy events → low aggregate confidence
    weak = _policy_silver("Weak", 2, strength=0.10)
    ev = to_canonical_evidence_frame(policy_events=weak)
    theses = build_capital_flow_theses(
        ev, config=CapitalFlowThesisConfig(min_aggregate_confidence=0.50)
    )
    # If any thesis is emitted, it must be marked rejected
    for t in theses:
        assert t.validation_status == "rejected"


def test_thesis_builder_records_supporting_and_contradiction_ids():
    # mix buy + sell on same symbol
    buy = _broker_silver("000001.SZ", 2, "buy")
    sell = _broker_silver("000001.SZ", 2, "sell")
    # shift sell dates so event_ids differ
    sell["announced_at"] = sell["announced_at"] + pd.Timedelta(days=10)
    sell["available_at"] = sell["available_at"] + pd.Timedelta(days=10)
    sell["event_id"] = sell["event_id"] + "_sell"
    ev = to_canonical_evidence_frame(broker_reports=pd.concat([buy, sell], ignore_index=True))
    theses = build_capital_flow_theses(ev)
    sym_theses = [t for t in theses if t.direction_kind == "symbol" and t.direction_value == "000001.SZ"]
    assert sym_theses
    t = sym_theses[0]
    assert t.supporting_evidence_ids
    assert t.contradiction_evidence_ids
    assert 0.0 < t.contradiction_score <= 1.0


def test_theses_to_frame_has_canonical_schema():
    ev = _build_evidence(sectors=[("Semi", 3)])
    theses = build_capital_flow_theses(ev)
    frame = theses_to_frame(theses)
    assert list(frame.columns) == list(CAPITAL_FLOW_THESIS_COLUMNS)


def test_capital_flow_thesis_builder_class_smoke():
    ev = _build_evidence(sectors=[("Semi", 3)])
    builder = CapitalFlowThesisBuilder()
    frame = builder.build_frame(ev)
    assert not frame.empty
    assert "thesis_id" in frame.columns


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _panel_with_excess(
    direction_kind: str,
    direction_value: str,
    *,
    start: pd.Timestamp,
    direction_excess: float,
    horizons: int = 120,
) -> pd.DataFrame:
    """Synthetic panel where the target direction outperforms benchmark."""
    dates = pd.bdate_range(start + pd.Timedelta(days=1), periods=horizons)
    rows: list[dict] = []
    for d in dates:
        rows.append({
            "trade_date": d,
            "sector_level_1": direction_value if direction_kind == "sector" else "OTHER",
            "theme": direction_value if direction_kind == "theme" else "OTHER",
            "symbol": direction_value if direction_kind == "symbol" else "999999.SH",
            "sector_return": direction_excess + 0.001,
            "forward_return": direction_excess + 0.001,
            "benchmark_return": 0.001,
        })
        # add a non-target row each day so the column has spread
        rows.append({
            "trade_date": d,
            "sector_level_1": "OTHER",
            "theme": "OTHER",
            "symbol": "999999.SH",
            "sector_return": 0.001,
            "forward_return": 0.001,
            "benchmark_return": 0.001,
        })
    return pd.DataFrame(rows)


def test_validation_promotes_thesis_to_verified_with_strong_excess():
    created = pd.Timestamp("2024-03-01")
    thesis = CapitalFlowThesis(
        thesis_id="t_a",
        direction_kind="sector",
        direction_value="Semi",
        thesis_sign=1.0,
        supporting_evidence_ids=["e1", "e2"],
        confidence=0.7,
        contradiction_score=0.0,
        expected_lag_days=5,
        validation_status="unverified",
        created_at=created,
    )
    # daily 0.5% direction excess compounds to >2% at h=5d, 20d, 60d, 120d
    panel = _panel_with_excess("sector", "Semi", start=created, direction_excess=0.005)
    result = validate_thesis(thesis, panel)
    assert result.new_status == "verified"
    assert len(result.horizons_confirmed) >= 2
    assert result.decay_score == 1.0
    assert result.tradability_score > 0.6


def test_validation_rejects_thesis_with_adverse_returns():
    created = pd.Timestamp("2024-03-01")
    thesis = CapitalFlowThesis(
        thesis_id="t_b",
        direction_kind="sector",
        direction_value="RealEstate",
        thesis_sign=1.0,
        supporting_evidence_ids=["e1", "e2"],
        confidence=0.7,
        contradiction_score=0.0,
        expected_lag_days=5,
        validation_status="unverified",
        created_at=created,
    )
    # daily -0.6% adverse compounds well past the -3% reject threshold
    panel = _panel_with_excess("sector", "RealEstate", start=created, direction_excess=-0.006)
    result = validate_thesis(thesis, panel)
    assert result.new_status == "rejected"
    assert result.decay_score == 0.0
    assert result.tradability_score < 0.2


def test_validation_partial_when_only_some_horizons_confirm():
    created = pd.Timestamp("2024-03-01")
    thesis = CapitalFlowThesis(
        thesis_id="t_c",
        direction_kind="sector",
        direction_value="Semi",
        thesis_sign=1.0,
        supporting_evidence_ids=["e1", "e2"],
        confidence=0.6,
        contradiction_score=0.0,
        expected_lag_days=5,
        validation_status="unverified",
        created_at=created,
    )
    # weak per-day excess: 1d cum=0.03%, 5d=0.15%, 20d=0.6%, 60d=1.8%
    # only h=120 (cum ≈3.7%) clears the 2% threshold → 1 confirmed → partial
    panel = _panel_with_excess("sector", "Semi", start=created, direction_excess=0.0003)
    result = validate_thesis(thesis, panel)
    assert result.new_status == "partially_verified"
    assert result.horizons_confirmed and max(result.horizons_confirmed) == 120


def test_validation_handles_short_horizon_only():
    created = pd.Timestamp("2024-03-01")
    thesis = CapitalFlowThesis(
        thesis_id="t_d",
        direction_kind="sector",
        direction_value="Semi",
        thesis_sign=1.0,
        supporting_evidence_ids=["e1", "e2"],
        confidence=0.6,
        expected_lag_days=5,
        validation_status="unverified",
        created_at=created,
    )
    # Only 3 business days of post-event panel — only h=1 will be present.
    # daily 5% → 1d cum 5% confirms, but only 1 horizon → partially_verified
    # (longest horizon not yet elapsed so cannot reject either)
    panel = _panel_with_excess("sector", "Semi", start=created, direction_excess=0.05, horizons=3)
    result = validate_thesis(thesis, panel)
    assert result.new_status in {"partially_verified", "unverified"}


def test_validate_theses_batch_returns_updated_immutable_copies():
    ev = _build_evidence(sectors=[("Semi", 3)])
    theses = build_capital_flow_theses(ev)
    panel = _panel_with_excess(
        "theme", "Semi",
        start=theses[0].created_at,
        direction_excess=0.05,
    )
    updated, results = validate_theses(theses, panel)
    assert len(updated) == len(theses)
    # Originals must not have been mutated (frozen dataclass + new instances)
    for original, new in zip(theses, updated):
        assert original is not new
    for r in results:
        assert r.new_status in THESIS_VALIDATION_STATES


def test_validation_returns_unverified_when_panel_missing_direction():
    created = pd.Timestamp("2024-03-01")
    thesis = CapitalFlowThesis(
        thesis_id="t_e",
        direction_kind="sector",
        direction_value="Nonexistent",
        thesis_sign=1.0,
        supporting_evidence_ids=["e1", "e2"],
        confidence=0.7,
        expected_lag_days=5,
        validation_status="unverified",
        created_at=created,
    )
    panel = _panel_with_excess("sector", "Semi", start=created, direction_excess=0.05)
    result = validate_thesis(thesis, panel)
    # No matching direction rows → no horizons computed → stays unverified
    assert result.new_status == "unverified"
    assert result.horizon_excess_returns == {}
