"""Tests for the Stage 3 regime sub-model label generator + ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.training.regime_sub_models import (
    EnsembleWeights,
    SubModelLabelConfig,
    build_regime_sub_labels,
    ensemble_sub_model_predictions,
    sub_model_setup_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_days: int = 80,
    n_symbols: int = 3,
    *,
    closes_override: dict[str, list[float]] | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for sid in range(n_symbols):
        sym = f"S{sid:03d}.SZ"
        base = 50.0
        if closes_override and sym in closes_override:
            closes = closes_override[sym]
            assert len(closes) == n_days
        else:
            closes = base * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_days))
        vol = rng.uniform(100_000, 500_000, n_days)
        for i, d in enumerate(dates):
            row = {"symbol": sym, "trade_date": d, "close": float(closes[i]), "volume": float(vol[i])}
            for h in horizons:
                row[f"forward_return_{h}d"] = 0.005 if (i + h) < n_days else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LowBuy setup
# ---------------------------------------------------------------------------

def test_lowbuy_setup_fires_on_deep_drawdown_with_stabilisation():
    # Engineer a series that drops 15% over 20 days then stabilises for 5 days
    n = 60
    closes = np.full(n, 100.0)
    closes[0:20] = 100.0
    closes[20:40] = np.linspace(100.0, 85.0, 20)  # -15% over 20 days
    closes[40:60] = np.full(20, 85.0)             # stabilised
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "lowbuy_setup"]
    # Setup should fire only when both: 20d return ≤ -10% AND 5d return ≥ -2%
    assert setups.iloc[40:60].any()
    # Before the drop, no LowBuy
    assert not setups.iloc[0:20].any()


def test_lowbuy_setup_does_not_fire_during_active_bleed():
    n = 60
    closes = np.full(n, 100.0)
    # Continuous bleed — 5d ret stays negative throughout
    closes[20:60] = np.linspace(100.0, 70.0, 40)
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "lowbuy_setup"]
    # The 5d return is < -2% during the bleed, so LowBuy should not fire
    assert not setups.iloc[25:55].any()


# ---------------------------------------------------------------------------
# Breakout setup
# ---------------------------------------------------------------------------

def test_breakout_fires_on_new_high_with_volume_confirmation():
    n = 80
    closes = np.full(n, 100.0)
    closes[0:70] = np.full(70, 100.0)
    closes[70:80] = np.linspace(101.0, 110.0, 10)  # break to new 60d high
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    # Force 5d volume far above 60d volume in the breakout window
    panel.loc[
        (panel["symbol"] == "S000.SZ") & (panel.index >= 70),
        "volume",
    ] = 1_000_000.0
    panel.loc[
        (panel["symbol"] == "S000.SZ") & (panel.index < 70),
        "volume",
    ] = 100_000.0
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "breakout_setup"]
    assert setups.iloc[70:80].any()


def test_breakout_does_not_fire_without_volume_confirmation():
    n = 80
    closes = np.full(n, 100.0)
    closes[70:80] = np.linspace(101.0, 110.0, 10)
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    # Volume identical throughout — no spike
    panel["volume"] = 200_000.0
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "breakout_setup"]
    assert not setups.iloc[70:80].any()


def test_breakout_works_when_volume_column_missing():
    n = 80
    closes = np.full(n, 100.0)
    closes[70:80] = np.linspace(101.0, 110.0, 10)
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    panel = panel.drop(columns=["volume"])
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "breakout_setup"]
    # Volume gate falls back to price-only → setup should still trigger on price breakout
    assert setups.iloc[70:80].any()


# ---------------------------------------------------------------------------
# LimitUpRisk setup
# ---------------------------------------------------------------------------

def test_limitup_risk_fires_on_recent_limit_up_with_high_run():
    n = 30
    closes = np.full(n, 50.0)
    # Build a 30% run, then a limit-up day at index 25
    closes[0:25] = np.linspace(50.0, 65.0, 25)
    closes[25] = 71.5     # +10% from 65 (limit-up)
    closes[26:30] = 71.5
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "limitup_risk_setup"]
    assert setups.iloc[26:29].any()


def test_limitup_risk_does_not_fire_without_high_run():
    n = 30
    closes = np.full(n, 50.0)
    # Limit-up at index 25 but no prior run-up
    closes[25] = 55.0  # +10% (limit-up)
    closes[26:30] = 55.0
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    out = build_regime_sub_labels(panel)
    setups = out.loc[out["symbol"] == "S000.SZ", "limitup_risk_setup"]
    # 20d run from 50 → 55 = +10% < 25% threshold → no setup
    assert not setups.iloc[26:29].any()


# ---------------------------------------------------------------------------
# Label content
# ---------------------------------------------------------------------------

def test_label_columns_are_nan_outside_setup_days():
    panel = _make_panel(n_days=40, n_symbols=2)
    out = build_regime_sub_labels(panel)
    for col in ("lowbuy_label_5d", "breakout_label_5d", "limitup_risk_label_5d"):
        assert col in out.columns
    # On non-setup rows the labels should be NaN, allowing dropna in the trainer
    non_setup = (
        ~out["lowbuy_setup"]
        & ~out["breakout_setup"]
        & ~out["limitup_risk_setup"]
    )
    assert out.loc[non_setup, "lowbuy_label_5d"].isna().all()
    assert out.loc[non_setup, "breakout_label_5d"].isna().all()
    assert out.loc[non_setup, "limitup_risk_label_5d"].isna().all()


def test_limitup_risk_label_is_negated_forward_return():
    """LimitUpRisk targets DOWNSIDE — its label is `-forward_return_Nd` so
    a positive prediction means the model thinks the name will drop.
    """
    n = 30
    closes = np.full(n, 50.0)
    closes[0:25] = np.linspace(50.0, 65.0, 25)
    closes[25] = 71.5
    closes[26:30] = 71.5
    panel = _make_panel(n_days=n, n_symbols=1, closes_override={"S000.SZ": closes.tolist()})
    out = build_regime_sub_labels(panel)
    setup_rows = out[out["limitup_risk_setup"]]
    assert not setup_rows.empty
    # forward_return_5d is the test-fixture constant 0.005 → label is -0.005
    fwd = setup_rows["forward_return_5d"].dropna()
    lab = setup_rows["limitup_risk_label_5d"].dropna()
    assert (lab.iloc[: len(fwd)] == -fwd.iloc[: len(lab)]).all()


# ---------------------------------------------------------------------------
# Missing-input handling
# ---------------------------------------------------------------------------

def test_empty_panel_returns_empty():
    out = build_regime_sub_labels(pd.DataFrame())
    assert out.empty


def test_missing_required_column_raises():
    bad = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")], "close": [10.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_regime_sub_labels(bad)


# ---------------------------------------------------------------------------
# Ensemble fusion
# ---------------------------------------------------------------------------

def _pred(date: pd.Timestamp, symbols: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"trade_date": date, "symbol": symbols, "alpha_score": scores}
    )


def test_ensemble_combines_with_default_weights_when_no_regime():
    d = pd.Timestamp("2024-03-01")
    lb = _pred(d, ["A", "B"], [1.0, 0.0])
    bo = _pred(d, ["A", "B"], [0.0, 1.0])
    lr = _pred(d, ["A", "B"], [0.5, 0.0])
    out = ensemble_sub_model_predictions(lb, bo, lr)
    by_sym = out.set_index("symbol")
    # Default: (0.40, 0.50, -0.10)
    assert by_sym.loc["A", "composite_score"] == pytest.approx(0.40 * 1.0 + 0.50 * 0.0 - 0.10 * 0.5)
    assert by_sym.loc["B", "composite_score"] == pytest.approx(0.40 * 0.0 + 0.50 * 1.0 - 0.10 * 0.0)


def test_ensemble_routes_lowbuy_higher_in_bear():
    d = pd.Timestamp("2024-03-01")
    lb = _pred(d, ["A"], [1.0])
    bo = _pred(d, ["A"], [0.0])
    lr = _pred(d, ["A"], [0.0])
    regimes = pd.DataFrame({"trade_date": [d], "regime_state": ["bear"]})
    out = ensemble_sub_model_predictions(lb, bo, lr, regime_states=regimes)
    # bear weights: (0.65, 0.20, -0.15)
    assert out["composite_score"].iloc[0] == pytest.approx(0.65)
    assert out["lowbuy_weight"].iloc[0] == pytest.approx(0.65)


def test_ensemble_missing_submodel_treated_as_zero_contribution():
    d = pd.Timestamp("2024-03-01")
    lb = _pred(d, ["A"], [1.0])
    bo = pd.DataFrame(columns=["trade_date", "symbol", "alpha_score"])  # missing
    lr = pd.DataFrame(columns=["trade_date", "symbol", "alpha_score"])  # missing
    out = ensemble_sub_model_predictions(lb, bo, lr)
    # Only LowBuy contributes; default lowbuy_weight = 0.40
    assert out["composite_score"].iloc[0] == pytest.approx(0.40)


def test_ensemble_outer_join_keeps_symbols_unique_per_submodel():
    d = pd.Timestamp("2024-03-01")
    lb = _pred(d, ["A"], [1.0])
    bo = _pred(d, ["B"], [1.0])
    lr = _pred(d, ["C"], [1.0])
    out = ensemble_sub_model_predictions(lb, bo, lr)
    assert set(out["symbol"]) == {"A", "B", "C"}
    # Each symbol only has its own sub-model's contribution
    by_sym = out.set_index("symbol")
    assert by_sym.loc["A", "lowbuy_score"] == 1.0
    assert by_sym.loc["A", "breakout_score"] == 0.0
    assert by_sym.loc["A", "limitup_risk_score"] == 0.0


def test_ensemble_unknown_regime_falls_back_to_default():
    d = pd.Timestamp("2024-03-01")
    lb = _pred(d, ["A"], [1.0])
    bo = _pred(d, ["A"], [0.0])
    lr = _pred(d, ["A"], [0.0])
    regimes = pd.DataFrame({"trade_date": [d], "regime_state": ["unicorn_market"]})
    out = ensemble_sub_model_predictions(lb, bo, lr, regime_states=regimes)
    # Unknown regime → default weights (0.40, 0.50, -0.10)
    assert out["composite_score"].iloc[0] == pytest.approx(0.40)


def test_ensemble_weights_dataframe():
    """Check that the per-row weight columns reflect the regime mapping."""
    d1 = pd.Timestamp("2024-03-01")
    d2 = pd.Timestamp("2024-03-02")
    lb = pd.concat([_pred(d1, ["A"], [1.0]), _pred(d2, ["A"], [1.0])])
    bo = pd.concat([_pred(d1, ["A"], [0.0]), _pred(d2, ["A"], [0.0])])
    lr = pd.concat([_pred(d1, ["A"], [0.0]), _pred(d2, ["A"], [0.0])])
    regimes = pd.DataFrame(
        {"trade_date": [d1, d2], "regime_state": ["normal", "bear"]}
    )
    out = ensemble_sub_model_predictions(lb, bo, lr, regime_states=regimes).sort_values("trade_date")
    assert list(out["lowbuy_weight"]) == [0.35, 0.65]  # normal then bear


# ---------------------------------------------------------------------------
# Setup statistics
# ---------------------------------------------------------------------------

def test_sub_model_setup_stats_basic_coverage():
    panel = _make_panel(n_days=80, n_symbols=2)
    out = build_regime_sub_labels(panel)
    stats = sub_model_setup_stats(out)
    assert stats["n_rows"] == int(len(out))
    for col in ("lowbuy_setup_rate", "breakout_setup_rate", "limitup_risk_setup_rate"):
        assert 0.0 <= stats[col] <= 1.0


def test_sub_model_setup_stats_empty():
    stats = sub_model_setup_stats(pd.DataFrame())
    assert stats["n_rows"] == 0
    assert stats["lowbuy_setup_rate"] == 0.0
