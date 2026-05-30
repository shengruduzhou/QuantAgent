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
