"""Tests for the Stage 4 policy event data layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.data.policy import (
    POLICY_EVENT_REQUIRED_COLUMNS,
    PolicyEventBuilder,
    PolicyEventConfig,
    build_policy_events,
    policy_events_for_features,
    tag_policy_event,
)


# ---------------------------------------------------------------------------
# Theme tagger
# ---------------------------------------------------------------------------

def test_monetary_keyword_triggers_monetary_theme():
    tags = tag_policy_event("央行宣布降准0.5个百分点", "释放长期资金约1万亿元")
    assert "monetary" in tags["themes"]
    assert tags["policy_strength"] > 0


def test_real_estate_keywords_tag_both_theme_and_sector():
    tags = tag_policy_event("关于优化个人住房贷款政策的通知", "首付比例下调")
    assert "real_estate" in tags["themes"]
    assert "房地产" in tags["sectors_hint"]


def test_multiple_themes_can_fire_together():
    tags = tag_policy_event(
        "关于支持半导体产业发展的指导意见",
        "对集成电路设计企业实施减税",
    )
    assert "tech_innovation" in tags["themes"]
    assert "fiscal" in tags["themes"]
    assert "电子" in tags["sectors_hint"]


def test_strength_band_hard_regulation():
    tags = tag_policy_event("证券公司风险控制管理办法", "规定")
    assert tags["policy_strength"] == 1.0


def test_strength_band_directive():
    tags = tag_policy_event("关于加强资本市场监管的指导意见", "")
    assert tags["policy_strength"] == 0.7


def test_strength_band_informational():
    tags = tag_policy_event("证监会负责人就近期市场情况答记者问", "")
    assert tags["policy_strength"] == 0.4


def test_empty_inputs_return_zero_tags():
    tags = tag_policy_event("", "")
    assert tags["themes"] == []
    assert tags["sectors_hint"] == []
    assert tags["policy_strength"] == 0.2  # informational default


# ---------------------------------------------------------------------------
# Builder normalisation
# ---------------------------------------------------------------------------

def _make_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_builder_normalises_required_columns():
    raw = _make_raw(
        [
            {
                "source": "csrc",
                "announced_at": "2024-01-15 09:30",
                "title": "关于资本市场监管的指导意见",
                "url": "https://csrc.gov.cn/x1",
            }
        ]
    )
    result = build_policy_events(raw)
    assert set(result.frame.columns) == set(POLICY_EVENT_REQUIRED_COLUMNS)
    assert len(result.frame) == 1


def test_builder_rejects_rows_without_announced_at():
    raw = _make_raw(
        [
            {"source": "csrc", "announced_at": "garbage", "title": "x", "url": "u"},
            {"source": "csrc", "announced_at": "2024-01-15", "title": "y", "url": "v"},
        ]
    )
    result = build_policy_events(raw)
    assert len(result.frame) == 1
    assert result.coverage["rejected_no_date"] == 1


def test_builder_deduplicates_by_event_id():
    raw = _make_raw(
        [
            {"source": "pboc", "announced_at": "2024-02-01", "title": "降准", "url": "https://pbc.gov.cn/a"},
            {"source": "pboc", "announced_at": "2024-02-01", "title": "降准", "url": "https://pbc.gov.cn/a"},
        ]
    )
    result = build_policy_events(raw)
    assert len(result.frame) == 1
    assert result.coverage["duplicates_removed"] == 1


def test_builder_unknown_source_falls_back_to_manual():
    raw = _make_raw(
        [{"source": "weird_blog", "announced_at": "2024-01-15", "title": "X", "url": "u"}]
    )
    result = build_policy_events(raw)
    assert (result.frame["source"] == "manual_local_import").all()


def test_builder_effective_at_defaults_to_announced_at():
    raw = _make_raw(
        [{"source": "csrc", "announced_at": "2024-02-15", "title": "Hello", "url": "u"}]
    )
    result = build_policy_events(raw)
    row = result.frame.iloc[0]
    assert row["effective_at"] == row["announced_at"]


def test_builder_available_at_is_max_of_announced_and_fetched():
    raw = _make_raw(
        [
            {
                "source": "csrc",
                "announced_at": "2024-02-15",
                "fetched_at": "2024-03-01",
                "title": "X",
                "url": "u",
            }
        ]
    )
    result = build_policy_events(raw)
    row = result.frame.iloc[0]
    assert row["available_at"] >= row["announced_at"]
    assert row["available_at"] >= row["fetched_at"]


def test_builder_themes_and_sectors_auto_tagged_from_title():
    raw = _make_raw(
        [
            {
                "source": "pboc",
                "announced_at": "2024-03-01",
                "title": "央行宣布下调贷款市场报价利率",
                "url": "u",
            }
        ]
    )
    result = build_policy_events(raw)
    row = result.frame.iloc[0]
    assert "monetary" in row["themes"]


def test_builder_themes_override_supersedes_auto_tagging():
    raw = _make_raw(
        [
            {
                "source": "csrc",
                "announced_at": "2024-03-01",
                "title": "完全没有关键词",
                "url": "u",
                "themes_override": ["industry"],
                "sectors_hint_override": ["Auto"],
            }
        ]
    )
    result = build_policy_events(raw)
    row = result.frame.iloc[0]
    assert row["themes"] == ["industry"]
    assert row["sectors_hint"] == ["Auto"]


def test_builder_missing_required_columns_raises():
    raw = _make_raw([{"announced_at": "2024-01-01", "title": "x"}])  # no source
    with pytest.raises(ValueError, match="missing required columns"):
        build_policy_events(raw)


def test_builder_empty_input_yields_empty_frame_with_closed_gate():
    result = build_policy_events(pd.DataFrame())
    assert result.frame.empty
    assert result.coverage["gate"]["policy_events_usable_for_features"] is False
    assert result.coverage["gate"]["reason"] == "no_events"


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------

def _good_batch(n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "source": "csrc",
                "announced_at": f"2024-01-{i + 1:02d}",
                "title": f"关于资本市场监管的指导意见{i}",
                "url": f"https://csrc.gov.cn/p{i}",
            }
        )
    return pd.DataFrame(rows)


def test_gate_opens_with_good_batch():
    result = build_policy_events(_good_batch(n=10))
    gate = result.coverage["gate"]
    assert gate["policy_events_usable_for_features"] is True
    assert gate["reason"] == "passed"


def test_gate_blocks_when_too_few_events():
    result = build_policy_events(_good_batch(n=2), config=PolicyEventConfig(min_events=5))
    gate = result.coverage["gate"]
    assert gate["policy_events_usable_for_features"] is False
    assert "too_few_events" in gate["reason"]


def test_gate_blocks_when_theme_coverage_too_low():
    # All titles contain only generic words → no themes tag
    raw = pd.DataFrame(
        [
            {"source": "csrc", "announced_at": f"2024-0{i // 30 + 1}-{i % 30 + 1:02d}", "title": f"通知{i}", "url": f"u{i}"}
            for i in range(10)
        ]
    )
    result = build_policy_events(raw, config=PolicyEventConfig(min_theme_coverage=0.50))
    gate = result.coverage["gate"]
    assert gate["policy_events_usable_for_features"] is False
    assert "theme_coverage" in gate["reason"]


def test_gate_blocks_when_strength_median_too_low():
    raw = pd.DataFrame(
        [
            {"source": "csrc", "announced_at": "2024-01-01", "title": "答记者问降准", "url": "u1"},
            {"source": "csrc", "announced_at": "2024-01-02", "title": "答记者问降准", "url": "u2"},
            {"source": "csrc", "announced_at": "2024-01-03", "title": "答记者问降准", "url": "u3"},
            {"source": "csrc", "announced_at": "2024-01-04", "title": "答记者问降准", "url": "u4"},
            {"source": "csrc", "announced_at": "2024-01-05", "title": "答记者问降准", "url": "u5"},
        ]
    )
    result = build_policy_events(raw, config=PolicyEventConfig(min_strength_median=0.60))
    gate = result.coverage["gate"]
    assert gate["policy_events_usable_for_features"] is False
    assert "median_strength" in gate["reason"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_emits_parquet_and_manifest(tmp_path):
    builder = PolicyEventBuilder(PolicyEventConfig(output_root=tmp_path))
    result = builder.write(builder.build(_good_batch(n=10)))
    parquet = tmp_path / "silver" / "policy_events" / "policy_events.parquet"
    manifest = tmp_path / "manifests" / "policy_events.json"
    assert parquet.exists()
    assert manifest.exists()
    assert (tmp_path / "silver" / "policy_events" / "coverage_report.json").exists()
    assert result.output_paths["policy_events"].endswith("policy_events.parquet")


def test_writer_manifest_has_gate_key(tmp_path):
    builder = PolicyEventBuilder(PolicyEventConfig(output_root=tmp_path))
    builder.write(builder.build(_good_batch(n=10)))
    manifest = json.loads(
        (tmp_path / "manifests" / "policy_events.json").read_text()
    )
    gate = manifest["extra"]["coverage_report"]["gate"]
    assert "policy_events_usable_for_features" in gate


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def test_overlay_helper_returns_none_when_gate_closed(tmp_path):
    closed = tmp_path / "closed.json"
    closed.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"policy_events_usable_for_features": False}}}}
        ),
        encoding="utf-8",
    )
    events = pd.DataFrame([{"event_id": "x"}])
    assert policy_events_for_features(events, closed) is None


def test_overlay_helper_returns_events_when_gate_open(tmp_path):
    open_path = tmp_path / "open.json"
    open_path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"policy_events_usable_for_features": True}}}}
        ),
        encoding="utf-8",
    )
    events = pd.DataFrame([{"event_id": "x"}])
    out = policy_events_for_features(events, open_path)
    assert out is not None
    assert len(out) == 1


def test_overlay_helper_handles_missing_inputs(tmp_path):
    assert policy_events_for_features(None, tmp_path / "x.json") is None
    assert policy_events_for_features(pd.DataFrame(), tmp_path / "x.json") is None
    assert policy_events_for_features(pd.DataFrame([{"x": 1}]), None) is None
    assert policy_events_for_features(pd.DataFrame([{"x": 1}]), tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_import_policy_events(tmp_path):
    from typer.testing import CliRunner
    from quantagent.cli import app

    raw = _good_batch(n=10)
    in_path = tmp_path / "policy.parquet"
    raw.to_parquet(in_path, index=False)
    out_root = tmp_path / "lake"
    result = CliRunner().invoke(
        app,
        [
            "import-policy-events-v7",
            "--input", str(in_path),
            "--output-root", str(out_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_root / "silver" / "policy_events" / "policy_events.parquet").exists()
    assert (out_root / "manifests" / "policy_events.json").exists()
