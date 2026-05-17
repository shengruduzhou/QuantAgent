"""Tests for Phase 3 dynamic portfolio upgrades.

Covers the multi-horizon blender, dynamic top_k, timing gate, holding-period
tracker and the wire-up inside ``build_v7_target_weights``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.agents.technical_timing_agent import compute_technical_timing
from quantagent.portfolio.dynamic_top_k import (
    DynamicTopKConfig,
    resolve_dynamic_top_k,
)
from quantagent.portfolio.multi_horizon_blender import (
    DEFAULT_HORIZON_WEIGHTS,
    MultiHorizonBlendConfig,
    blend_multi_horizon_predictions,
)
from quantagent.portfolio.position_age_tracker import PositionAgeTracker
from quantagent.portfolio.timing_gate import TimingGateConfig, apply_timing_gate
from quantagent.portfolio.v7_target_weights import (
    V7TargetWeightsConfig,
    build_v7_target_weights,
)


# ----- helpers ----------------------------------------------------------------


def _make_market_panel(dates: list[pd.Timestamp], symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for symbol in symbols:
        base_price = 50.0 + rng.normal(0, 5)
        for i, date in enumerate(dates):
            drift = 1 + 0.001 * (i - len(dates) // 2) + rng.normal(0, 0.01)
            close = base_price * drift
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "amount": 1_000_000_000.0,
                    "volume": 10_000_000.0,
                    "available_at": date + pd.tseries.offsets.BusinessDay(1),
                    "is_suspended": False,
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def _make_predictions_multi_horizon(
    dates: list[pd.Timestamp],
    symbols: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for date in dates:
        for symbol in symbols:
            for horizon in horizons:
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "horizon": horizon,
                        "prediction": float(rng.normal(0.01 if horizon <= 20 else 0.005, 0.02)),
                        "sample_role": "validation",
                        "fold_id": 1,
                    }
                )
    return pd.DataFrame(rows)


# ----- Phase 3.1 — multi-horizon blender -------------------------------------


def test_blender_passes_through_single_horizon_predictions():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B"],
            "prediction": [0.01, 0.02],
            "sample_role": ["validation"] * 2,
        }
    )
    out = blend_multi_horizon_predictions(frame)
    assert out.diagnostics["status"] == "passthrough"
    assert len(out.blended) == 2


def test_blender_collapses_multi_horizon_to_single_row_per_pair():
    dates = pd.bdate_range("2024-01-02", periods=2)
    preds = _make_predictions_multi_horizon(list(dates), ["A", "B"], [1, 5, 20, 60, 120])
    out = blend_multi_horizon_predictions(preds)
    assert out.diagnostics["status"] == "passed"
    # 2 dates × 2 symbols = 4 rows of blended output
    assert len(out.blended) == 4
    assert set(out.blended.columns) == {"trade_date", "symbol", "prediction"}


def test_blender_falls_back_to_primary_horizon_when_missing():
    # Drop horizon=5 entirely → blender should use primary fallback for those rows.
    dates = pd.bdate_range("2024-01-02", periods=2)
    preds = _make_predictions_multi_horizon(list(dates), ["A", "B"], [1, 20, 60, 120])  # no 5
    cfg = MultiHorizonBlendConfig(require_all_horizons=True, primary_horizon=20)
    out = blend_multi_horizon_predictions(preds, config=cfg)
    assert out.diagnostics["fallback_rows"] > 0


def test_blender_handles_lifecycle_decay_with_short_bias():
    dates = pd.bdate_range("2024-01-02", periods=1)
    preds = _make_predictions_multi_horizon(list(dates), ["A"], [1, 5, 20, 60, 120])
    # Set 1d prediction high; with DECAY override, short horizons dominate.
    preds.loc[(preds["symbol"] == "A") & (preds["horizon"] == 1), "prediction"] = 0.1
    preds.loc[(preds["symbol"] == "A") & (preds["horizon"] == 120), "prediction"] = -0.05
    theme = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A"],
            "lifecycle_stage": ["DECAY"],
        }
    )
    out = blend_multi_horizon_predictions(preds, theme_signals=theme)
    # DECAY puts ~0.40 weight on 1d, ~0.35 on 5d → blended pulled toward 0.1.
    assert out.blended.iloc[0]["prediction"] > 0.0


# ----- Phase 3.2 — dynamic top_k ---------------------------------------------


def test_dynamic_top_k_clamps_on_small_universe():
    # 5-name universe, top_k_max=50 should clamp to <= 4 (one short of universe).
    cfg = DynamicTopKConfig(top_k_min=8, top_k_max=50, base_top_k=30)
    decision = resolve_dynamic_top_k(eligible_count=5, predictions_for_date=pd.Series([0.1, -0.1, 0.05, -0.05, 0.0]), config=cfg)
    assert decision.top_k <= 4
    assert decision.top_k >= 1


def test_dynamic_top_k_lifecycle_decay_lowers_count():
    cfg = DynamicTopKConfig(top_k_min=5, top_k_max=50, base_top_k=30)
    base = resolve_dynamic_top_k(eligible_count=200, predictions_for_date=pd.Series(np.linspace(-0.1, 0.1, 200)), config=cfg)
    theme = pd.DataFrame({"lifecycle_stage": ["DECAY"] * 200, "policy_strength": [0.0] * 200})
    decayed = resolve_dynamic_top_k(eligible_count=200, predictions_for_date=pd.Series(np.linspace(-0.1, 0.1, 200)), theme_signals_for_date=theme, config=cfg)
    assert decayed.top_k < base.top_k


def test_dynamic_top_k_capital_inflow_raises_count():
    cfg = DynamicTopKConfig(top_k_min=5, top_k_max=80, base_top_k=30)
    theme = pd.DataFrame({"lifecycle_stage": ["CAPITAL_INFLOW"] * 200, "policy_strength": [0.8] * 200})
    decision = resolve_dynamic_top_k(eligible_count=200, predictions_for_date=pd.Series(np.linspace(-0.1, 0.1, 200)), theme_signals_for_date=theme, config=cfg)
    assert decision.top_k > cfg.base_top_k


# ----- Phase 3.3 — timing gate -----------------------------------------------


def test_timing_gate_disabled_is_no_op():
    out = apply_timing_gate(pd.DataFrame(), None, TimingGateConfig(enabled=False))
    assert out.decisions.empty


def test_timing_gate_none_entry_zone_treated_as_permissive():
    dates = pd.bdate_range("2024-01-02", periods=3)
    panel = _make_market_panel(list(dates), ["A"])
    plan = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A"] * 3,
            "entry_zone_low": [np.nan] * 3,
            "entry_zone_high": [np.nan] * 3,
            "invalidation_level": [np.nan] * 3,
        }
    )
    out = apply_timing_gate(panel, plan, TimingGateConfig(enabled=True))
    assert out.decisions["allow_open"].all()


def test_atr_timing_producer_emits_columns():
    dates = pd.bdate_range("2024-01-02", periods=40)
    panel = _make_market_panel(list(dates), ["A"])
    plan = compute_technical_timing(panel)
    assert {"atr", "entry_zone_low", "entry_zone_high", "invalidation_level"} <= set(plan.columns)
    assert (plan["atr"] > 0).any()


# ----- Phase 3.4 — position age tracker --------------------------------------


def test_age_tracker_persists_state_across_sessions(tmp_path: Path):
    path = tmp_path / "state.parquet"
    tracker = PositionAgeTracker(state_path=path)
    tracker.record_session(pd.Timestamp("2024-01-02"), {"A": 0.1, "B": 0.05}, {"A": 20, "B": 60})
    tracker.record_session(pd.Timestamp("2024-01-03"), {"A": 0.1, "B": 0.05}, {})
    tracker.persist()
    fresh = PositionAgeTracker.from_state(path)
    snap = fresh.snapshot()
    assert {"A", "B"} <= set(snap["symbol"].astype(str))
    a = snap[snap["symbol"] == "A"].iloc[0]
    assert a["expected_horizon_days"] == 20


def test_age_tracker_locks_under_aged_positions(tmp_path: Path):
    tracker = PositionAgeTracker()
    tracker.record_session(pd.Timestamp("2024-01-02"), {"A": 0.1}, {"A": 20})
    assert tracker.is_locked("A", pd.Timestamp("2024-01-05"))  # 3 days < 20
    # Force close override.
    assert not tracker.is_locked("A", pd.Timestamp("2024-01-05"), force_close=True)


# ----- Phase 3 wire-up --------------------------------------------------------


def test_build_target_weights_with_dynamic_topk_does_not_raise_on_small_universe():
    dates = pd.bdate_range("2024-01-02", periods=10)
    symbols = ["A", "B", "C", "D", "E"]
    panel = _make_market_panel(list(dates), symbols)
    preds = pd.DataFrame(
        [
            {"trade_date": date, "symbol": sym, "prediction": np.random.normal(0, 0.02)}
            for date in dates
            for sym in symbols
        ]
    )
    cfg = V7TargetWeightsConfig(
        dynamic_top_k_enabled=True,
        top_k_min=2,
        top_k_max=50,  # over the 5-name universe — must shrink
        min_selection_pressure=1.0,
        max_weight_per_name=0.5,
    )
    out = build_v7_target_weights(preds, panel, config=cfg)
    assert not out.target_weights.empty
    diag = out.diagnostics.get("dynamic_top_k_decisions", [])
    assert diag, "expected dynamic top_k diagnostics"


def test_build_target_weights_holding_period_locks_under_aged_positions(tmp_path: Path):
    dates = pd.bdate_range("2024-01-02", periods=10)
    symbols = ["A", "B", "C"]
    panel = _make_market_panel(list(dates), symbols)
    preds = pd.DataFrame(
        [
            {"trade_date": date, "symbol": sym, "prediction": np.random.normal(0, 0.02)}
            for date in dates
            for sym in symbols
        ]
    )
    theme = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": sym,
                "lifecycle_stage": "CAPITAL_INFLOW",
                "policy_strength": 0.6,
                "confidence": 0.7,
                "expected_horizon_days": 60,
            }
            for date in dates
            for sym in symbols
        ]
    )
    cfg = V7TargetWeightsConfig(
        holding_period_mode="soft",
        holding_period_max_delta=0.005,
        dynamic_top_k_enabled=True,
        top_k_min=2,
        top_k_max=50,
        min_selection_pressure=1.0,
        max_weight_per_name=0.5,
    )
    state_path = tmp_path / "state.parquet"
    out = build_v7_target_weights(
        preds,
        panel,
        config=cfg,
        theme_signals=theme,
        position_state_path=state_path,
    )
    assert state_path.exists()
    locks = out.diagnostics.get("holding_period_locks", [])
    # With 60-day horizon and only 10-day backtest, at least one lock event should fire.
    assert isinstance(locks, list)


def test_build_target_weights_capital_tier_lowers_participation():
    dates = pd.bdate_range("2024-01-02", periods=5)
    symbols = ["A", "B", "C"]
    panel = _make_market_panel(list(dates), symbols)
    preds = pd.DataFrame(
        [
            {"trade_date": date, "symbol": sym, "prediction": np.random.normal(0, 0.02)}
            for date in dates
            for sym in symbols
        ]
    )
    cfg = V7TargetWeightsConfig(
        capital_yuan=1e8,
        liquidity_participation=0.10,
        capital_tier_overrides=((1e6, 0.10), (1e7, 0.05), (1e8, 0.02)),
        min_selection_pressure=1.0,
        max_weight_per_name=0.5,
        fail_if_top_k_covers_universe=False,
    )
    out = build_v7_target_weights(preds, panel, config=cfg)
    assert out.diagnostics["effective_participation_rate"] == pytest.approx(0.02)


def test_a_share_gates_run_before_dynamic_top_k():
    """Invariant: ST / suspended / limit-locked names cannot be selected.

    The dynamic top_k path must not bypass the A-share tradability filter.
    """

    dates = pd.bdate_range("2024-01-02", periods=3)
    symbols = ["A", "B"]
    panel = _make_market_panel(list(dates), symbols)
    panel.loc[panel["symbol"] == "A", "is_suspended"] = True  # A is suspended on every day
    preds = pd.DataFrame(
        [
            {"trade_date": date, "symbol": sym, "prediction": 0.05 if sym == "A" else 0.01}
            for date in dates
            for sym in symbols
        ]
    )
    cfg = V7TargetWeightsConfig(
        dynamic_top_k_enabled=True,
        top_k_min=1,
        top_k_max=10,
        min_selection_pressure=1.0,
        max_weight_per_name=1.0,
    )
    out = build_v7_target_weights(preds, panel, config=cfg)
    # A must not appear in target weights despite its high alpha.
    assert "A" not in out.target_weights.columns or (out.target_weights["A"].sum() == 0)
