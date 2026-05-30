"""Tests for the Stage 4.3 bond-market flow data layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.bond import (
    BOND_FLOW_REQUIRED_COLUMNS,
    BondFlowBuilder,
    BondFlowConfig,
    apply_bond_flow_features,
    bond_flows_for_features,
    build_bond_flows,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _good_batch(n_days: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "yield_1y": rng.normal(2.20, 0.05, n_days),
            "yield_5y": rng.normal(2.55, 0.05, n_days),
            "yield_10y": rng.normal(2.80, 0.05, n_days),
            "yield_3m": rng.normal(1.95, 0.05, n_days),
            "yield_aa": rng.normal(4.10, 0.07, n_days),
            "yield_aaa": rng.normal(3.50, 0.06, n_days),
            "dr007": rng.normal(1.85, 0.06, n_days),
            "bond_fund_flow": rng.normal(0.0, 30.0, n_days),
        }
    )


# ---------------------------------------------------------------------------
# Builder normalisation
# ---------------------------------------------------------------------------

def test_builder_produces_required_columns():
    result = build_bond_flows(_good_batch(n_days=30))
    assert set(result.frame.columns) == set(BOND_FLOW_REQUIRED_COLUMNS)


def test_builder_derives_spreads_from_yields():
    raw = _good_batch(n_days=30)
    result = build_bond_flows(raw)
    # spread_10y_1y, spread_10y_3m, credit_spread_aa, credit_spread_aaa_aa
    # should be derived from the yield columns
    assert result.frame["spread_10y_1y"].notna().any()
    assert result.frame["spread_10y_3m"].notna().any()
    assert result.frame["credit_spread_aa"].notna().any()
    assert result.frame["credit_spread_aaa_aa"].notna().any()
    # Sanity: 10y - 1y > 0 most days (normal yield curve)
    pos_term = (result.frame["spread_10y_1y"] > 0).sum()
    assert pos_term >= 25  # >80% of 30 days


def test_builder_rejects_rows_without_trade_date():
    raw = _good_batch(n_days=30)
    bad = raw.copy()
    bad.loc[5:8, "trade_date"] = "garbage"
    result = build_bond_flows(bad)
    assert result.coverage["rejected_no_date"] == 4
    assert len(result.frame) == 26


def test_builder_deduplicates_by_trade_date():
    raw = _good_batch(n_days=30)
    dup = pd.concat([raw, raw.head(5)], ignore_index=True)
    result = build_bond_flows(dup)
    assert result.coverage["duplicates_removed"] == 5
    assert len(result.frame) == 30


def test_builder_available_at_defaults_to_next_business_day():
    raw = _good_batch(n_days=10)
    result = build_bond_flows(raw)
    for _, row in result.frame.iterrows():
        # available_at is at least the next BD after trade_date
        expected = row["trade_date"] + pd.tseries.offsets.BDay(1)
        assert row["available_at"] >= expected


def test_builder_empty_input_yields_closed_gate():
    result = build_bond_flows(pd.DataFrame())
    assert result.frame.empty
    gate = result.coverage["gate"]
    assert gate["bond_flows_usable_for_features"] is False
    assert gate["reason"] == "no_rows"


def test_builder_missing_trade_date_raises():
    with pytest.raises(ValueError, match="trade_date"):
        build_bond_flows(pd.DataFrame([{"yield_10y": 2.8}]))


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def test_gate_opens_with_good_batch():
    result = build_bond_flows(_good_batch(n_days=40))
    gate = result.coverage["gate"]
    assert gate["bond_flows_usable_for_features"] is True
    assert gate["reason"] == "passed"


def test_gate_blocks_with_too_few_days():
    result = build_bond_flows(
        _good_batch(n_days=10), config=BondFlowConfig(min_days=30)
    )
    gate = result.coverage["gate"]
    assert gate["bond_flows_usable_for_features"] is False
    assert "too_few_days" in gate["reason"]


def test_gate_blocks_with_low_field_coverage():
    raw = _good_batch(n_days=40)
    # Strip out 80% of yield fields
    for col in ("yield_1y", "yield_5y", "yield_10y", "yield_aa", "yield_aaa", "dr007", "bond_fund_flow"):
        raw[col] = np.nan
    result = build_bond_flows(raw)
    gate = result.coverage["gate"]
    assert gate["bond_flows_usable_for_features"] is False
    assert "field_coverage" in gate["reason"]


def test_gate_blocks_with_low_date_continuity():
    raw = _good_batch(n_days=40)
    # Drop every other row → continuity ~50%
    raw = raw.iloc[::2].reset_index(drop=True)
    result = build_bond_flows(raw, config=BondFlowConfig(min_days=10, min_date_continuity=0.95))
    gate = result.coverage["gate"]
    assert gate["bond_flows_usable_for_features"] is False
    assert "date_continuity" in gate["reason"]


# ---------------------------------------------------------------------------
# Writer + manifest
# ---------------------------------------------------------------------------

def test_writer_emits_parquet_and_manifest(tmp_path):
    builder = BondFlowBuilder(BondFlowConfig(output_root=tmp_path, min_days=10))
    result = builder.write(builder.build(_good_batch(n_days=15)))
    assert (tmp_path / "silver" / "bond_flows" / "bond_flows.parquet").exists()
    assert (tmp_path / "manifests" / "bond_flows.json").exists()
    manifest = json.loads((tmp_path / "manifests" / "bond_flows.json").read_text())
    assert "bond_flows_usable_for_features" in manifest["extra"]["coverage_report"]["gate"]
    assert result.output_paths["bond_flows"].endswith("bond_flows.parquet")


# ---------------------------------------------------------------------------
# Feature join
# ---------------------------------------------------------------------------

def test_apply_features_adds_prefixed_columns():
    panel = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-15", periods=5),
            "symbol": ["A.SZ"] * 5,
        }
    )
    flows = build_bond_flows(_good_batch(n_days=20)).frame
    out = apply_bond_flow_features(panel, flows)
    assert "bond_yield_10y" in out.columns
    assert "bond_spread_10y_1y" in out.columns
    assert "bond_dr007" in out.columns


def test_apply_features_pit_safe_no_future_leak():
    """A panel row on 2024-01-15 must not see bond data from 2024-01-20."""
    flows = build_bond_flows(_good_batch(n_days=20)).frame
    # Build a panel with a row before all bond data and one within
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-15")],
            "symbol": ["A.SZ", "A.SZ"],
        }
    )
    out = apply_bond_flow_features(panel, flows)
    before_first_bond = out.iloc[0]
    after_first_bond = out.iloc[1]
    # The "before all bond data" row should have all NaN bond features
    assert pd.isna(before_first_bond["bond_yield_10y"])
    # The "within bond panel" row should have a real value
    assert pd.notna(after_first_bond["bond_yield_10y"])


def test_apply_features_returns_panel_unchanged_when_no_flows():
    panel = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-15")], "symbol": ["A.SZ"]})
    out = apply_bond_flow_features(panel, pd.DataFrame())
    assert list(out.columns) == ["trade_date", "symbol"]


def test_apply_features_handles_empty_panel():
    flows = build_bond_flows(_good_batch(n_days=20)).frame
    out = apply_bond_flow_features(pd.DataFrame(), flows)
    assert out.empty


def test_apply_features_subset_columns():
    flows = build_bond_flows(_good_batch(n_days=20)).frame
    panel = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2024-01-15")], "symbol": ["A.SZ"]}
    )
    out = apply_bond_flow_features(
        panel, flows, feature_columns=("yield_10y", "dr007"), prefix="m_"
    )
    assert "m_yield_10y" in out.columns
    assert "m_dr007" in out.columns
    # Other bond fields not attached
    assert "m_credit_spread_aa" not in out.columns


# ---------------------------------------------------------------------------
# Overlay helper / manifest gate
# ---------------------------------------------------------------------------

def test_overlay_helper_returns_none_when_gate_closed(tmp_path):
    closed = tmp_path / "closed.json"
    closed.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"bond_flows_usable_for_features": False}}}}
        ),
        encoding="utf-8",
    )
    flows = pd.DataFrame([{"trade_date": pd.Timestamp("2024-01-15")}])
    assert bond_flows_for_features(flows, closed) is None


def test_overlay_helper_returns_frame_when_gate_open(tmp_path):
    open_path = tmp_path / "open.json"
    open_path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"bond_flows_usable_for_features": True}}}}
        ),
        encoding="utf-8",
    )
    flows = pd.DataFrame([{"trade_date": pd.Timestamp("2024-01-15")}])
    assert bond_flows_for_features(flows, open_path) is not None


def test_overlay_helper_missing_inputs_return_none(tmp_path):
    assert bond_flows_for_features(None, tmp_path / "x.json") is None
    assert bond_flows_for_features(pd.DataFrame(), tmp_path / "x.json") is None
    assert bond_flows_for_features(pd.DataFrame([{"x": 1}]), None) is None
    assert bond_flows_for_features(pd.DataFrame([{"x": 1}]), tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_import_bond_flows(tmp_path):
    from typer.testing import CliRunner
    from quantagent.cli import app

    raw = _good_batch(n_days=40)
    in_path = tmp_path / "bond.parquet"
    raw.to_parquet(in_path, index=False)
    out_root = tmp_path / "lake"
    result = CliRunner().invoke(
        app,
        [
            "import-bond-flows-v7",
            "--input", str(in_path),
            "--output-root", str(out_root),
            "--min-days", "30",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_root / "silver" / "bond_flows" / "bond_flows.parquet").exists()
    assert (out_root / "manifests" / "bond_flows.json").exists()
