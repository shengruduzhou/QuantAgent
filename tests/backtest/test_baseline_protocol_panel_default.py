"""The canonical evaluator must not silently fall back to the legacy panel (round-29 R4-F01).

The legacy v7 silver panel's is_st is the 2026-05-31 ST list broadcast to every
historical date, so a historical result built on it excludes names that only
became ST later - future losers - which is look-ahead.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.baseline_protocol as baseline_protocol


def _predictions(tmp_path):
    path = tmp_path / "preds.parquet"
    pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-02"]), "symbol": ["600000.SH"], "alpha_score": [1.0]}
    ).to_parquet(path)
    return str(path)


def test_without_a_certified_panel_the_legacy_panel_needs_a_stated_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_protocol, "CERTIFIED_PANEL", str(tmp_path / "absent.parquet"))
    with pytest.raises(ValueError, match="look-ahead"):
        baseline_protocol.evaluate(
            _predictions(tmp_path), top_k=1, start="2024-01-02", end="2024-01-31",
            slippage_bps=8.0, variants=["C_flags_eligible_delay1"],
        )


def test_the_legacy_trust_class_names_the_lookahead():
    assert "lookahead" in baseline_protocol.LEGACY_PANEL_TRUST_CLASS
    assert "ST list" in baseline_protocol.LEGACY_PANEL_NOTE


def test_st_unknown_buy_share_counts_only_unmeasured_names():
    trades = pd.DataFrame({
        "symbol": ["600000.SH", "000001.SZ"], "trade_date": ["2024-01-03", "2024-01-03"],
        "side": ["buy", "buy"], "filled_quantity": [100, 300], "avg_price": [10.0, 10.0],
    })
    panel = pd.DataFrame({
        "symbol": ["600000.SH", "000001.SZ"], "trade_date": pd.to_datetime(["2024-01-03"] * 2),
        "st_status": ["UNKNOWN", "FALSE"],
    })
    assert baseline_protocol._st_unknown_buy_share(trades, panel) == 0.25


def test_b_shares_are_excluded_and_counted():
    """Round-29 R10: HKD/USD-quoted B shares were bought with prices read as CNY."""
    panel = pd.DataFrame({"symbol": ["200054.SZ", "900901.SH", "000001.SZ"], "trade_date": pd.to_datetime(["2024-01-03"] * 3)})
    preds = pd.DataFrame({"symbol": ["200054.SZ", "000001.SZ"], "trade_date": pd.to_datetime(["2024-01-02"] * 2)})
    note: dict = {}
    kept_panel, kept_preds = baseline_protocol._exclude_b_shares(panel, preds, note)
    assert list(kept_panel["symbol"]) == ["000001.SZ"]
    assert list(kept_preds["symbol"]) == ["000001.SZ"]
    assert note == {"excluded_b_share_symbols": 2, "excluded_b_share_prediction_rows": 1}
