"""Validate that the symbolic-GA factor synthesiser actually discovers signal.

We inject a known cross-sectional alpha (5-day momentum) into synthetic OHLCV
and confirm the GA produces at least one factor whose train/validation IC have
the same sign and reach a meaningful magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.factors import expr as E
from quantagent.factors.factor_synthesis import SymbolicGAConfig, _evaluate_fitness, synthesize_factors


def _make_panel_with_momentum_signal(
    *,
    n_symbols: int = 6,
    n_days: int = 220,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate OHLCV where forward_return_5d ~ rank(close_t / close_{t-5})."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    closes: dict[str, list[float]] = {}
    for j in range(n_symbols):
        symbol = f"S{j:02d}"
        # Mean-reverting random walk so closes do not blow up.
        path = [10.0 + j]
        for _ in range(1, n_days):
            shock = rng.normal(0.0, 0.02)
            path.append(max(0.5, path[-1] * (1.0 + shock)))
        closes[symbol] = path

    for i, date in enumerate(dates):
        # Compute cross-sectional 5d momentum at i to drive label at i.
        momentum: dict[str, float] = {}
        for symbol, path in closes.items():
            if i >= 5:
                momentum[symbol] = path[i] / path[i - 5] - 1.0
            else:
                momentum[symbol] = 0.0
        if i >= 5:
            ranked = pd.Series(momentum).rank(method="average")
            # forward_return_5d ~ this rank (with small noise)
            normed = (ranked - ranked.mean()) / max(ranked.std(), 1e-9)
            label = {sym: float(normed[sym]) * 0.01 + rng.normal(0.0, 0.003) for sym in closes}
        else:
            label = {sym: 0.0 for sym in closes}
        for symbol, path in closes.items():
            close = path[i]
            volume = 1_000_000.0 + 50_000.0 * (j + 1) + rng.normal(0, 30_000)
            volume = float(max(volume, 1_000.0))
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": close * 0.997,
                    "high": close * 1.012,
                    "low": close * 0.985,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "forward_return_5d": label[symbol],
                }
            )
    return pd.DataFrame(rows)


def test_synthesize_factors_recovers_injected_momentum_signal():
    panel = _make_panel_with_momentum_signal()
    cfg = SymbolicGAConfig(
        population=60,
        generations=12,
        max_depth=3,
        min_depth=2,
        tournament_size=5,
        elitism=4,
        top_k=10,
        label_column="forward_return_5d",
        complexity_penalty=2e-4,
        min_finite_ratio=0.3,
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=400,
        fitness_sample_symbols=20,
        seed=2024,
    )
    result = synthesize_factors(panel, config=cfg)

    assert not result.leaderboard.empty, "GA produced no surviving factors"
    top = result.leaderboard.iloc[0]
    # The injected signal is monotone in 5-day momentum; even a coarse GA
    # should find an expression with at least mild positive IC on both
    # train and validation, and not flip sign between them.
    assert top["train_rank_ic"] >= 0.10, (
        f"top train_rank_ic too low: {top['train_rank_ic']:.4f}"
    )
    assert top["validation_rank_ic"] >= 0.0, (
        f"top validation_rank_ic flipped negative: {top['validation_rank_ic']:.4f}"
    )

    # GA history must show monotone (best ≥ first generation) progress over
    # the run — i.e., evolution is actually doing something.
    assert not result.history.empty
    last_best = float(result.history.iloc[-1]["best_fitness"])
    first_best = float(result.history.iloc[0]["best_fitness"])
    assert last_best >= first_best - 1e-9


def test_synthesize_factors_rejects_pure_noise():
    """Random returns should yield few/no factors above the validation gate."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-02", periods=160, freq="B")
    rows: list[dict[str, object]] = []
    for j in range(5):
        sym = f"N{j:02d}"
        close = 10.0
        for date in dates:
            close *= 1.0 + rng.normal(0.0, 0.015)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": date,
                    "open": close * 0.998,
                    "high": close * 1.011,
                    "low": close * 0.986,
                    "close": close,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    # No signal — label is pure noise, uncorrelated with anything.
                    "forward_return_5d": float(rng.normal(0.0, 0.01)),
                }
            )
    panel = pd.DataFrame(rows)
    cfg = SymbolicGAConfig(
        population=40,
        generations=6,
        top_k=10,
        label_column="forward_return_5d",
        validation_fraction=0.3,
        min_validation_rank_ic=0.10,  # demand non-trivial OOS IC
        seed=11,
    )
    result = synthesize_factors(panel, config=cfg)
    # Either nothing survives, or whatever survives has very small magnitude.
    assert (
        result.leaderboard.empty
        or result.leaderboard["validation_rank_ic"].abs().max() < 0.25
    )


def test_synthesize_factors_rejects_constant_only_expression():
    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    labels = panel["forward_return_5d"]
    dates = panel["trade_date"]
    cfg = SymbolicGAConfig(label_column="forward_return_5d")

    fitness, raw_ic, finite_ratio = _evaluate_fitness(
        E.TsStd(E.Rank(E.Constant(2.0)), 5),
        panel,
        labels,
        dates,
        cfg,
    )

    assert fitness == -1.0
    assert raw_ic == 0.0
    assert finite_ratio == 0.0


def test_synthesize_factors_normalizes_trade_date_before_label_merge():
    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    labels = panel[["symbol", "trade_date", "forward_return_5d"]].copy()
    panel = panel.drop(columns=["forward_return_5d"])
    panel["trade_date"] = panel["trade_date"].dt.strftime("%Y-%m-%d")
    cfg = SymbolicGAConfig(
        population=8,
        generations=1,
        top_k=2,
        label_column="forward_return_5d",
        fitness_sample_dates=30,
        fitness_sample_symbols=4,
        seed=99,
    )

    result = synthesize_factors(panel, labels=labels, config=cfg)

    assert result.history is not None


def test_labels_with_overlapping_ohlcv_columns_do_not_kill_population():
    """Regression: full training datasets carry their own OHLCV columns; the
    merge must not suffix the panel's market columns (which made every tree
    raise KeyError and every fitness -1.0)."""
    panel = _make_panel_with_momentum_signal(n_symbols=5, n_days=120)
    labels = panel[["symbol", "trade_date", "forward_return_5d"]].copy()
    # Simulate a training-dataset labels file that also contains OHLCV.
    labels["close"] = 1.0
    labels["volume"] = 2.0
    labels["open"] = 3.0
    panel_no_label = panel.drop(columns=["forward_return_5d"])
    cfg = SymbolicGAConfig(
        population=12,
        generations=1,
        max_depth=3,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=3,
    )
    result = synthesize_factors(panel_no_label, labels=labels, config=cfg)
    assert (result.history["best_fitness"] > -1.0).any()


def test_new_dsl_operators_evaluate_and_roundtrip():
    from quantagent.factors.factor_synthesis import parse_expression

    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    candidates = [
        E.TsCorr(E.Rank(E.Close), E.Rank(E.Volume), 10),
        E.TsCov(E.Returns(E.Close, 1), E.Returns(E.Volume, 1), 10),
        E.DecayLinear(E.Delta(E.Close, 3), 10),
        E.CsZscore(E.Div(E.Close, E.TsMean(E.Close, 20))),
    ]
    for tree in candidates:
        values = tree.evaluate(panel)
        assert values.notna().sum() > 0, repr(tree)
        rebuilt = parse_expression(repr(tree))
        pd.testing.assert_series_equal(
            rebuilt.evaluate(panel), values, check_names=False
        )


def test_decay_linear_weights_recent_bars_most():
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=10, freq="B")):
        rows.append({"symbol": "S", "trade_date": d, "open": 1.0, "high": 1.0,
                     "low": 1.0, "close": float(i), "volume": 1.0, "amount": 1.0})
    frame = pd.DataFrame(rows)
    out = E.DecayLinear(E.Close, 3).evaluate(frame)
    # window [1,2,3] with ascending weights 1/6,2/6,3/6 over closes 1,2,3 -> (1*1+2*2+3*3)/6
    expected = (1 * 1 + 2 * 2 + 3 * 3) / 6.0
    assert abs(out.iloc[3] - expected) < 1e-9


def test_ts_rank_matches_pandas_reference():
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("A", "B"):
        for i, d in enumerate(pd.date_range("2024-01-01", periods=40, freq="B")):
            rows.append({"symbol": sym, "trade_date": d, "open": 1.0, "high": 1.0,
                         "low": 1.0, "close": float(rng.normal()), "volume": 1.0, "amount": 1.0})
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    fast = E.TsRank(E.Close, 10).evaluate(frame)
    ref = (
        frame.assign(trade_date=pd.to_datetime(frame["trade_date"]))
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol")["close"]
        .rolling(10, min_periods=10)
        .apply(lambda b: pd.Series(b).rank(method="average").iloc[-1] / len(b), raw=True)
        .reset_index(level=0, drop=True)
    )
    aligned = fast.reindex(ref.index)
    mask = ref.notna()
    assert np.allclose(aligned[mask], ref[mask])
