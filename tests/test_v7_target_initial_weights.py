"""First-date portfolio-state regressions for V7 target construction."""

from __future__ import annotations

import pandas as pd

import quantagent.portfolio.v7_target_weights as v7_target_weights
import pytest

from quantagent.portfolio.v7_target_weights import (
    V7TargetWeightsConfig,
    build_v7_target_weights,
)


def _two_session_inputs():
    dates = [pd.Timestamp("2026-08-06"), pd.Timestamp("2026-08-07")]
    market = pd.DataFrame(
        [
            {"trade_date": dates[0], "symbol": "A", "close": 10.0, "low": 9.8, "amount": 100_000_000.0},
            {"trade_date": dates[1], "symbol": "A", "close": 10.2, "low": 10.0, "amount": 100_000_000.0},
            {"trade_date": dates[0], "symbol": "B", "close": 20.0, "low": 19.8, "amount": 100_000_000.0},
            {"trade_date": dates[1], "symbol": "B", "close": 20.2, "low": 20.0, "amount": 100_000_000.0},
        ]
    )
    timing = pd.DataFrame(
        [
            # Previous-session close is deliberately outside both entry zones,
            # so the next session blocks *new* opens in both names.
            {"trade_date": dates[0], "symbol": "A", "entry_zone_low": 5.0, "entry_zone_high": 6.0, "invalidation_level": 1.0},
            {"trade_date": dates[1], "symbol": "A", "entry_zone_low": 5.0, "entry_zone_high": 6.0, "invalidation_level": 1.0},
            {"trade_date": dates[0], "symbol": "B", "entry_zone_low": 15.0, "entry_zone_high": 16.0, "invalidation_level": 1.0},
            {"trade_date": dates[1], "symbol": "B", "entry_zone_low": 15.0, "entry_zone_high": 16.0, "invalidation_level": 1.0},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"trade_date": dates[1], "symbol": "A", "prediction": 0.20, "confidence": 0.90},
            {"trade_date": dates[1], "symbol": "B", "prediction": 0.10, "confidence": 0.90},
        ]
    )
    return predictions, market, timing


def _config(**overrides) -> V7TargetWeightsConfig:
    values = dict(
        optimizer_backend="deterministic",
        selection_mode="ai_threshold",
        selection_top_k_min=1,
        selection_top_k_max=2,
        max_weight_per_name=1.0,
        max_sector_weight=1.0,
        max_turnover=0.0,
        liquidity_participation=1.0,
        capital_yuan=1_000_000.0,
        timing_gate_enabled=True,
        min_selection_pressure=1.0,
    )
    values.update(overrides)
    return V7TargetWeightsConfig(**values)


def test_initial_actual_holding_is_not_reclassified_as_a_new_timing_entry():
    predictions, market, timing = _two_session_inputs()

    zero_state = build_v7_target_weights(
        predictions,
        market,
        config=_config(),
        timing_plan=timing,
    )
    assert zero_state.target_weights.empty

    actual_state = build_v7_target_weights(
        predictions,
        market,
        config=_config(),
        timing_plan=timing,
        initial_weights=pd.Series({"A": 0.20}),
    )

    assert not actual_state.target_weights.empty
    assert "A" in actual_state.target_weights.columns
    assert "B" not in actual_state.target_weights.columns
    assert actual_state.diagnostics["initial_weights"] == {
        "supplied": True,
        "symbol_count": 1,
        "gross_exposure": pytest.approx(0.20),
        "net_exposure": pytest.approx(0.20),
    }


def test_initial_weights_fail_closed_on_invalid_long_only_account_state():
    predictions, market, _ = _two_session_inputs()
    with pytest.raises(ValueError, match="cannot start from negative weights"):
        build_v7_target_weights(
            predictions,
            market,
            config=_config(timing_gate_enabled=False),
            initial_weights=pd.Series({"A": -0.10}),
        )


def test_liquidity_capacity_changes_with_account_nav():
    trade_date = pd.Timestamp("2026-08-07")
    predictions = pd.DataFrame(
        [
            {"trade_date": trade_date, "symbol": "A", "prediction": 0.20, "confidence": 0.90},
            {"trade_date": trade_date, "symbol": "B", "prediction": 0.10, "confidence": 0.90},
        ]
    )
    market = pd.DataFrame(
        [
            {"trade_date": trade_date, "symbol": "A", "close": 10.0, "amount": 1_000_000.0},
            {"trade_date": trade_date, "symbol": "B", "close": 20.0, "amount": 1_000_000.0},
        ]
    )
    common = dict(
        optimizer_backend="deterministic",
        selection_mode="ai_threshold",
        selection_top_k_min=1,
        selection_top_k_max=2,
        max_weight_per_name=1.0,
        max_sector_weight=1.0,
        max_turnover=0.0,
        liquidity_participation=0.05,
        min_selection_pressure=1.0,
    )

    one_million = build_v7_target_weights(
        predictions,
        market,
        config=V7TargetWeightsConfig(**common, capital_yuan=1_000_000.0),
    )
    five_million = build_v7_target_weights(
        predictions,
        market,
        config=V7TargetWeightsConfig(**common, capital_yuan=5_000_000.0),
    )

    one_weight = float(one_million.target_weights.drop(columns=["trade_date"]).sum(axis=1).iloc[0])
    five_weight = float(five_million.target_weights.drop(columns=["trade_date"]).sum(axis=1).iloc[0])
    assert one_weight > five_weight
    assert one_weight == pytest.approx(0.10)
    assert five_weight == pytest.approx(0.02)


def test_explicit_empty_initial_weights_still_reconcile_age_tracker(monkeypatch) -> None:
    predictions, market, _ = _two_session_inputs()
    calls: list[dict[str, float]] = []
    original = v7_target_weights.PositionAgeTracker.begin_session

    def spy(self, initial_weights, expected_horizons=None):
        calls.append(dict(initial_weights or {}))
        return original(self, initial_weights, expected_horizons)

    monkeypatch.setattr(v7_target_weights.PositionAgeTracker, "begin_session", spy)
    build_v7_target_weights(
        predictions,
        market,
        config=_config(
            timing_gate_enabled=False, holding_period_mode="hard", max_turnover=1.0
        ),
        initial_weights=pd.Series(dtype=float),
    )
    assert calls == [{}]


def test_all_dates_rejected_persists_cash_only_tracker_reconciliation(tmp_path) -> None:
    predictions, market, timing = _two_session_inputs()
    state_path = tmp_path / "position_age.parquet"
    tracker = v7_target_weights.PositionAgeTracker(state_path=state_path)
    tracker.begin_session({"STALE": 0.20}, {"STALE": 5})
    tracker.persist()
    assert not v7_target_weights.PositionAgeTracker.from_state(state_path).snapshot().empty

    result = build_v7_target_weights(
        predictions,
        market,
        config=_config(
            holding_period_mode="hard",
            max_turnover=1.0,
        ),
        timing_plan=timing,
        position_state_path=state_path,
        initial_weights=pd.Series(dtype=float),
    )
    assert result.target_weights.empty
    assert result.diagnostics["status"] == "all_dates_rejected"
    assert result.diagnostics["position_state_rows"] == 0
    assert v7_target_weights.PositionAgeTracker.from_state(state_path).snapshot().empty
