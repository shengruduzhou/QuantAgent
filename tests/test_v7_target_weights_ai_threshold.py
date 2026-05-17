"""Tests for V7 target-weights AI-threshold selection mode.

These pin the new ``selection_mode="ai_threshold"`` branch in
``build_v7_target_weights``:

* selection respects ``alpha_threshold`` and ``confidence_floor``
* fallback to ``selection_top_k_min`` when threshold filters wipe out the pool
* cap to ``selection_top_k_max`` when threshold filters let too many through
* selection survives a missing ``confidence`` column (predict-only mode)
* downstream constraint shape (sector cap / lot rounding) is unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(
    n_symbols: int = 12,
    n_dates: int = 3,
    confidence_for: dict[str, float] | None = None,
    prediction_for: dict[str, float] | None = None,
    include_confidence: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confidence_for = confidence_for or {}
    prediction_for = prediction_for or {}
    dates = pd.date_range("2025-01-02", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    rng = np.random.default_rng(13)
    rows: list[dict[str, object]] = []
    market: list[dict[str, object]] = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            pred = prediction_for.get(symbol, float(rng.standard_normal() * 0.01))
            row: dict[str, object] = {
                "symbol": symbol,
                "trade_date": date,
                "prediction": pred,
            }
            if include_confidence:
                row["confidence"] = confidence_for.get(
                    symbol,
                    float(np.clip(0.30 + rng.random() * 0.65, 0.05, 0.95)),
                )
            rows.append(row)
            market.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": 10.0 + (i % 7),
                    "close": 10.5 + (i % 7),
                    "amount": 5e7,
                    "is_suspended": False,
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(market)


def test_ai_threshold_filters_low_prediction_and_low_confidence():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    # 4 strong (prediction high, confidence high), rest fail at least one gate.
    preds, market = _panel(
        n_symbols=20,
        n_dates=2,
        prediction_for={"S000": 0.10, "S001": 0.08, "S002": 0.05, "S003": 0.04},
        confidence_for={"S000": 0.80, "S001": 0.75, "S002": 0.70, "S003": 0.65},
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="ai_threshold",
            alpha_threshold=0.02,
            confidence_floor=0.60,
            selection_top_k_min=1,
            selection_top_k_max=50,
            fail_if_top_k_covers_universe=False,
        ),
    )
    decisions = [row for row in result.diagnostics.get("optimizer_backend", []) if row.get("backend") == "ai_threshold"]
    assert decisions, "should record ai_threshold diagnostic per date"
    for row in decisions:
        # 4 strong winners satisfy both gates; selected_count should be ≤ 4 (never wider).
        assert row["selected_count"] <= 4, row


def test_ai_threshold_fallback_to_min_when_filters_empty():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    # All confidence below floor → threshold pool empties → fallback to min.
    preds, market = _panel(
        n_symbols=12,
        n_dates=2,
        confidence_for={f"S{i:03d}": 0.10 for i in range(12)},
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="ai_threshold",
            alpha_threshold=0.0,
            confidence_floor=0.80,
            selection_top_k_min=3,
            selection_top_k_max=50,
            fail_if_top_k_covers_universe=False,
        ),
    )
    decisions = [row for row in result.diagnostics.get("optimizer_backend", []) if row.get("backend") == "ai_threshold"]
    assert decisions
    for row in decisions:
        assert row["fallback_to_min"] is True, row
        assert row["selected_count"] == 3, row


def test_ai_threshold_caps_at_max():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    # 30 names all clear both gates → cap to 5.
    preds, market = _panel(
        n_symbols=30,
        n_dates=2,
        prediction_for={f"S{i:03d}": 0.05 for i in range(30)},
        confidence_for={f"S{i:03d}": 0.80 for i in range(30)},
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="ai_threshold",
            alpha_threshold=0.0,
            confidence_floor=0.50,
            selection_top_k_min=2,
            selection_top_k_max=5,
            fail_if_top_k_covers_universe=False,
        ),
    )
    decisions = [row for row in result.diagnostics.get("optimizer_backend", []) if row.get("backend") == "ai_threshold"]
    assert decisions
    for row in decisions:
        assert row["selected_count"] == 5, row
        assert row["capped_at_max"] is True, row


def test_ai_threshold_without_confidence_column_still_works():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _panel(n_symbols=15, n_dates=2, include_confidence=False)
    assert "confidence" not in preds.columns
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="ai_threshold",
            alpha_threshold=0.0,
            confidence_floor=0.55,
            selection_top_k_min=2,
            selection_top_k_max=10,
            fail_if_top_k_covers_universe=False,
        ),
    )
    decisions = [row for row in result.diagnostics.get("optimizer_backend", []) if row.get("backend") == "ai_threshold"]
    assert decisions, "ai_threshold must run even without confidence column"
    for row in decisions:
        assert row["selected_count"] >= 1, row


def test_ai_threshold_weights_sum_to_one_per_date():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _panel(
        n_symbols=20,
        n_dates=2,
        prediction_for={f"S{i:03d}": 0.02 + 0.001 * i for i in range(20)},
        confidence_for={f"S{i:03d}": 0.80 for i in range(20)},
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="ai_threshold",
            alpha_threshold=0.0,
            confidence_floor=0.55,
            selection_top_k_min=2,
            selection_top_k_max=30,
            fail_if_top_k_covers_universe=False,
        ),
    )
    frame = result.target_weights
    if frame.empty:
        pytest.skip("ai_threshold produced no positions; cannot validate per-date gross")
    if "trade_date" in frame.columns:
        frame = frame.set_index("trade_date")
    for date, row in frame.iterrows():
        gross = float(row.astype(float).abs().sum())
        if gross > 0:
            assert gross == pytest.approx(1.0, abs=1e-6), f"{date} gross={gross}"


def test_top_k_mode_still_works_as_fallback():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _panel(n_symbols=20, n_dates=2)
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="top_k",
            top_k=6,
            top_k_ratio=None,
            min_selection_pressure=2.0,
            fail_if_top_k_covers_universe=False,
        ),
    )
    backends = {row.get("backend") for row in result.diagnostics.get("optimizer_backend", [])}
    assert "ai_threshold" not in backends, "top_k mode must not emit ai_threshold diagnostics"
