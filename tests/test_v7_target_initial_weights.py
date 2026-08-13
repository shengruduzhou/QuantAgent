"""First-date portfolio-state regressions for V7 target construction."""

from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

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


def test_force_closed_zero_weight_cannot_bypass_next_entry_gate(monkeypatch) -> None:
    dates = pd.to_datetime(["2026-08-05", "2026-08-06", "2026-08-07"])
    predictions = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "prediction": 1.0 if symbol == "A" else 0.1,
                "confidence": 0.9,
            }
            for date in dates
            for symbol in ("A", "B")
        ]
    )
    market = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "close": 10.0,
                "low": 9.9,
                "amount": 100_000_000.0,
            }
            for date in dates
            for symbol in ("A", "B")
        ]
    )
    decisions = pd.DataFrame(
        [
            {"trade_date": dates[0], "symbol": "A", "allow_open": True, "force_close": False},
            {"trade_date": dates[1], "symbol": "A", "allow_open": True, "force_close": True},
            {"trade_date": dates[2], "symbol": "A", "allow_open": False, "force_close": False},
        ]
    )
    monkeypatch.setattr(
        v7_target_weights,
        "apply_timing_gate",
        lambda *_args, **_kwargs: SimpleNamespace(decisions=decisions, diagnostics={}),
    )
    theme = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "expected_horizon_days": 5}
            for date in dates
            for symbol in ("A", "B")
        ]
    )

    result = build_v7_target_weights(
        predictions,
        market,
        timing_plan=pd.DataFrame({"enabled": [True]}),
        theme_signals=theme,
        config=_config(
            holding_period_mode="hard",
            max_turnover=0.0,
            max_weight_per_name=1.0,
            selection_top_k_min=1,
            selection_top_k_max=1,
        ),
    )

    weights = result.target_weights.set_index("trade_date")
    assert weights.loc[dates[0], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[1], "A"] == pytest.approx(0.0)
    assert weights.loc[dates[2], "A"] == pytest.approx(0.0)
    assert weights.loc[dates[2], "B"] == pytest.approx(1.0)


def test_locked_holding_reserves_budget_before_new_exposure() -> None:
    date = pd.Timestamp("2026-08-07")
    predictions = pd.DataFrame(
        [
            {"trade_date": date, "symbol": "A", "prediction": 0.1, "confidence": 0.9},
            {"trade_date": date, "symbol": "B", "prediction": 1.0, "confidence": 0.9},
        ]
    )
    market = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "close": 10.0, "low": 9.9, "amount": 100_000_000.0}
            for symbol in ("A", "B")
        ]
    )
    theme = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "expected_horizon_days": 60}
            for symbol in ("A", "B")
        ]
    )
    result = build_v7_target_weights(
        predictions,
        market,
        sector_map=pd.DataFrame(
            [{"symbol": "A", "industry": "X"}, {"symbol": "B", "industry": "X"}]
        ),
        theme_signals=theme,
        initial_weights=pd.Series({"A": 0.8}),
        config=_config(
            holding_period_mode="hard",
            holding_period_max_delta=0.02,
            max_turnover=0.5,
            cash_floor=0.1,
            max_weight_per_name=0.9,
            max_sector_weight=0.9,
            selection_top_k_min=1,
            selection_top_k_max=1,
        ),
    )

    row = result.target_weights.drop(columns=["trade_date"]).iloc[0]
    assert float(row.sum()) <= 0.9 + 1e-9
    assert float(row.max()) <= 0.9 + 1e-9
    assert float((row.reindex(["A", "B"]).fillna(0.0) - pd.Series({"A": 0.8, "B": 0.0})).abs().sum()) <= 0.5 + 1e-9
    assert row["A"] >= 0.78 - 1e-9


def test_single_day_turnover_diagnostic_includes_canonical_starting_weights() -> None:
    date = pd.Timestamp("2026-08-07")
    predictions = pd.DataFrame(
        [
            {"trade_date": date, "symbol": "A", "prediction": 0.1, "confidence": 0.9},
            {"trade_date": date, "symbol": "B", "prediction": 1.0, "confidence": 0.9},
        ]
    )
    market = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "close": 10.0, "low": 9.9, "amount": 100_000_000.0}
            for symbol in ("A", "B")
        ]
    )
    result = build_v7_target_weights(
        predictions,
        market,
        initial_weights=pd.Series({"A": 0.6}),
        config=_config(
            max_turnover=0.4,
            max_weight_per_name=1.0,
            selection_top_k_min=1,
            selection_top_k_max=1,
        ),
    )

    assert result.diagnostics["average_turnover"] == pytest.approx(0.4)
