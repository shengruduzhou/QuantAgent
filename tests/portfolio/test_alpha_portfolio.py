"""Unit tests for the turnover-controlled alpha portfolio constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.alpha_portfolio import (
    AlphaPortfolioConfig,
    build_alpha_portfolio,
)


def _synthetic_predictions(n_dates: int = 60, n_symbols: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    syms = [f"{600000 + i}.SH" for i in range(n_symbols)]
    rows = []
    for d in dates:
        scores = rng.normal(size=n_symbols)
        for s, sc in zip(syms, scores):
            rows.append((d, s, float(sc)))
    return pd.DataFrame(rows, columns=["trade_date", "symbol", "alpha_score"])


def test_long_only_decile_weights_sum_to_one_and_respect_book_fraction():
    # 400 names → decile = 40, equal weight 0.025 < 0.05 cap (cap non-binding)
    preds = _synthetic_predictions(n_symbols=400)
    cfg = AlphaPortfolioConfig(book_fraction=0.10, rebalance_interval=1)
    w = build_alpha_portfolio(preds, config=cfg)
    assert not w.empty
    # each rebalance row sums to ~1.0 (fully invested long book)
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-9)
    # exactly ~10 % of 400 names held long per date
    held = (w.iloc[0] > 0).sum()
    assert held == 40


def test_rebalance_interval_emits_fewer_rows():
    preds = _synthetic_predictions(n_dates=60)
    daily = build_alpha_portfolio(preds, config=AlphaPortfolioConfig(rebalance_interval=1))
    held20 = build_alpha_portfolio(preds, config=AlphaPortfolioConfig(rebalance_interval=20))
    assert len(daily) == 60
    assert len(held20) == 3  # 60 / 20
    # rebalance dates are a subset of daily dates
    assert set(held20.index).issubset(set(daily.index))


def test_max_name_weight_cap_is_respected_with_rank_weighting():
    # 400 names → decile = 40, feasible cap (40 * 0.04 = 1.6 ≥ 1) so the
    # book stays fully invested while no name exceeds the cap.
    preds = _synthetic_predictions(n_symbols=400)
    cfg = AlphaPortfolioConfig(book_fraction=0.10, weighting="rank",
                               max_name_weight=0.04, rebalance_interval=1)
    w = build_alpha_portfolio(preds, config=cfg)
    assert float(w.values.max()) <= 0.04 + 1e-9
    assert np.allclose(w.sum(axis=1).values, 1.0, atol=1e-9)


def test_long_short_is_market_neutral():
    preds = _synthetic_predictions(n_symbols=400)
    cfg = AlphaPortfolioConfig(book_fraction=0.10, long_short=True, rebalance_interval=1)
    w = build_alpha_portfolio(preds, config=cfg)
    # net exposure ~0, gross ~1
    net = w.sum(axis=1)
    gross = w.abs().sum(axis=1)
    assert np.allclose(net.values, 0.0, atol=1e-9)
    assert np.allclose(gross.values, 1.0, atol=1e-9)


def test_gross_scale_scales_the_book():
    preds = _synthetic_predictions(n_symbols=400)
    full = build_alpha_portfolio(preds, config=AlphaPortfolioConfig(rebalance_interval=1, gross_scale=1.0))
    half = build_alpha_portfolio(preds, config=AlphaPortfolioConfig(rebalance_interval=1, gross_scale=0.5))
    assert np.allclose(half.sum(axis=1).values, 0.5, atol=1e-9)
    assert np.allclose(full.sum(axis=1).values, 1.0, atol=1e-9)


def test_liquidity_floor_drops_illiquid_names():
    preds = _synthetic_predictions(n_dates=20, n_symbols=200)
    dates = sorted(pd.to_datetime(preds["trade_date"].unique()))
    syms = sorted(preds["symbol"].unique())
    # first 100 symbols are illiquid (amount below floor), rest liquid
    rows = []
    for d in dates:
        for j, s in enumerate(syms):
            rows.append({"trade_date": d, "symbol": s,
                         "avg_amount": 1e6 if j < 100 else 1e8})
    liq = pd.DataFrame(rows)
    cfg = AlphaPortfolioConfig(book_fraction=0.10, rebalance_interval=1,
                               min_avg_amount_yuan=5e7)
    w = build_alpha_portfolio(preds, config=cfg, liquidity=liq)
    held = set(w.columns[(w != 0).any()])
    # only liquid names (>= floor) may be held
    illiquid = set(syms[:100])
    assert held.isdisjoint(illiquid)


def test_per_date_regime_scale_zero_means_cash():
    preds = _synthetic_predictions(n_dates=40)
    dates = sorted(pd.to_datetime(preds["trade_date"].unique()))
    scale = pd.Series(1.0, index=dates)
    scale.iloc[:20] = 0.0  # first rebalance window is "crisis" → cash
    w = build_alpha_portfolio(
        preds, config=AlphaPortfolioConfig(rebalance_interval=20),
        gross_scale_by_date=scale,
    )
    # first rebalance row is all-cash (zero gross), second is invested
    assert float(w.iloc[0].abs().sum()) == 0.0
    assert float(w.iloc[1].abs().sum()) > 0.0
