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
    resolve_horizon_blend_config,
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
    """DECAY must tilt the blend toward the short sleeves.

    Stated as a *cross-sectional* claim, because that is the only kind the
    blender can answer: it ranks each horizon within each date, so "A scores
    above B" is meaningful and "A scores 0.07" is not. A one-name panel has no
    cross-section to rank and every score collapses to 0.0 by construction —
    which the ``min_cross_section_size`` diagnostic now reports.
    """
    dates = pd.bdate_range("2024-01-02", periods=1)
    preds = _make_predictions_multi_horizon(list(dates), ["A", "B"], [1, 5, 20, 60, 120])
    # A is the short-horizon winner and the long-horizon loser; B is the reverse.
    short, long = preds["horizon"].isin([1, 5]), preds["horizon"].isin([60, 120])
    preds.loc[(preds["symbol"] == "A") & short, "prediction"] = 0.10
    preds.loc[(preds["symbol"] == "B") & short, "prediction"] = -0.10
    preds.loc[(preds["symbol"] == "A") & long, "prediction"] = -0.10
    preds.loc[(preds["symbol"] == "B") & long, "prediction"] = 0.10
    theme = pd.DataFrame(
        {
            "trade_date": list(dates) * 2,
            "symbol": ["A", "B"],
            "lifecycle_stage": ["DECAY", "DECAY"],
        }
    )

    out = blend_multi_horizon_predictions(preds, theme_signals=theme)
    scores = out.blended.set_index("symbol")["prediction"]

    # DECAY puts 0.40/0.35 on 1d/5d against 0.07/0.03 on 60d/120d, so the
    # short-horizon winner must rank first.
    assert scores["A"] > scores["B"]
    assert out.diagnostics["min_cross_section_size"] == 2


def test_blender_weights_are_the_realised_weights():
    """Declared horizon weights must survive contact with the label scales.

    Each horizon predicts a different label, and a 120-day forward return is an
    order of magnitude larger than a 1-day one (cross-sectional sigma 0.237 vs
    0.022 on the gold panel). Summing raw predictions therefore weights horizon
    ``h`` by ``w_h * sigma_h``: the shipped 10% on 1d realised 1.8% and the 15%
    on 120d realised 29.3%. Rank-normalising first is what makes the config mean
    what it says.
    """
    dates = pd.bdate_range("2024-01-02", periods=200)
    symbols = [f"S{index:02d}" for index in range(60)]
    rng = np.random.default_rng(0)
    # Cross-sectional sigmas measured on the gold panel's own labels.
    sigma = {1: 0.0216, 5: 0.0511, 20: 0.1029, 60: 0.1710, 120: 0.2372}
    rows = []
    for date in dates:
        for horizon, scale in sigma.items():
            # Each horizon ranks the cross-section independently, so the blend's
            # correlation with a horizon measures that horizon's influence.
            rows.append(
                pd.DataFrame(
                    {
                        "trade_date": date,
                        "symbol": symbols,
                        "horizon": horizon,
                        "prediction": rng.normal(0.0, scale, len(symbols)),
                    }
                )
            )
    preds = pd.concat(rows, ignore_index=True)

    def influence(scale_normalisation: str) -> dict[int, float]:
        blended = blend_multi_horizon_predictions(
            preds,
            config=MultiHorizonBlendConfig(scale_normalisation=scale_normalisation),
        ).blended
        merged = preds.merge(blended, on=["trade_date", "symbol"], suffixes=("_h", "_b"))
        return {
            int(horizon): float(
                group["prediction_h"].corr(group["prediction_b"], method="spearman")
            )
            for horizon, group in merged.groupby("horizon")
        }

    corrected = influence("cross_sectional_rank")
    legacy = influence("none")

    # DEFAULT_HORIZON_WEIGHTS declares 20d the heaviest sleeve at 30% and 120d
    # the lightest of the long ones at 15%.
    assert corrected[20] > corrected[120], (
        f"declared 30% on 20d must outrank declared 15% on 120d, got {corrected}"
    )
    assert corrected[20] == max(corrected.values())
    # Summing raw predictions inverts that: the 120d sleeve dominates on scale
    # alone, and the 1d sleeve all but disappears.
    assert legacy[120] > legacy[20], f"legacy mode should be 120d-dominated, got {legacy}"
    assert legacy[1] < corrected[1] / 2.0


def test_adaptive_horizon_blend_learns_on_early_oos_and_freezes_before_holdout():
    dates = pd.bdate_range("2024-01-02", periods=40)
    symbols = [f"S{index:02d}" for index in range(8)]
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            cross_section = float(symbol_index - 3.5)
            rows.extend([
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "horizon": 5,
                    "prediction": cross_section + date_index * 0.001,
                    "forward_return_5d": cross_section,
                },
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "horizon": 20,
                    "prediction": -cross_section,
                    "forward_return_20d": cross_section,
                },
            ])
    predictions = pd.DataFrame(rows)

    config, diagnostics = resolve_horizon_blend_config(
        predictions,
        method="adaptive_oos",
        primary_horizon=5,
        holdout_days=10,
    )

    assert dict(config.horizon_weights) == {5: 1.0}
    assert diagnostics["source"] == "early_oos_rank_ic"
    assert diagnostics["frozenBeforeHoldout"] is True
    assert diagnostics["forwardLabelPurgeApplied"] is True
    assert diagnostics["purgeByHorizon"][5]["purgeDays"] == 5
    assert diagnostics["holdoutStart"] == dates[-10].strftime("%Y-%m-%d")

    tampered_holdout = predictions.copy()
    holdout_dates = set(dates[-10:])
    mask = tampered_holdout["trade_date"].isin(holdout_dates)
    tampered_holdout.loc[mask, "prediction"] *= -100
    for column in ("forward_return_5d", "forward_return_20d"):
        tampered_holdout.loc[mask, column] = 999.0
    frozen_config, frozen_diagnostics = resolve_horizon_blend_config(
        tampered_holdout,
        method="adaptive_oos",
        primary_horizon=5,
        holdout_days=10,
    )

    assert frozen_config.horizon_weights == config.horizon_weights
    assert frozen_diagnostics["horizonWeights"] == diagnostics["horizonWeights"]


def test_declared_horizon_preset_filters_to_available_prediction_horizons():
    dates = pd.bdate_range("2024-01-02", periods=2)
    predictions = _make_predictions_multi_horizon(
        list(dates),
        ["A", "B"],
        [1, 5, 20],
    )

    config, diagnostics = resolve_horizon_blend_config(
        predictions,
        method="balanced",
        primary_horizon=5,
        holdout_days=10,
    )
    result = blend_multi_horizon_predictions(predictions, config=config)

    assert set(dict(config.horizon_weights)) == {1, 5, 20}
    assert sum(dict(config.horizon_weights).values()) == pytest.approx(1.0)
    assert diagnostics["source"] == "declared_preset"
    assert result.diagnostics["fallback_rows"] == 0


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
        selection_mode="top_k",
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
