"""Cycle 32 P1-3: performance-metric call sites delegate to ``quant_math.performance``.

Each ``_frozen_*`` is the private formula as it stood before consolidation.
Delegation is accepted only if every output stays within 1e-12 (relative for
Sharpe, absolute for drawdown) of the frozen copy on randomized return series,
including degenerate ones (constant, two points, NaN gaps, deep losses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.quant_math.performance import max_drawdown, sharpe_ratio

SEEDS = range(25)


def _close(new: float, old: float, tol: float = 1e-12) -> bool:
    if old is None or new is None:
        return old is None and new is None
    if np.isnan(old):
        return bool(np.isnan(new))
    return abs(new - old) <= tol * max(1.0, abs(old))


def _returns(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    kind = seed % 5
    n = int(rng.integers(2, 400))
    if kind == 0:
        values = rng.normal(0.0005, 0.02, n)
    elif kind == 1:
        values = rng.normal(-0.01, 0.05, n)  # deep losses
    elif kind == 2:
        values = np.full(n, 0.001)  # constant
    elif kind == 3:
        values = rng.normal(0.0, 1e-7, n)
    else:
        values = rng.normal(0.001, 0.01, n)
        values[rng.random(n) < 0.1] = np.nan
    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(values, index=index)


# ---- frozen references -----------------------------------------------------


def _frozen_strict_sharpe(daily_ret: pd.Series, obs_per_year: float) -> float:
    std = float(daily_ret.std(ddof=1))
    return float(daily_ret.mean() / std * (obs_per_year ** 0.5)) if std > 1e-12 else 0.0


def _frozen_strict_max_dd(base_nav: float, nav_clean: pd.Series) -> float:
    nav_curve = np.concatenate(([base_nav], nav_clean.to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(nav_curve)
    drawdowns = nav_curve / peaks - 1.0
    return float(abs(drawdowns.min())) if len(drawdowns) else 0.0


def _frozen_paper_sharpe(daily_returns: pd.Series, observations_per_year: float):
    daily_std = float(daily_returns.std(ddof=0)) if len(daily_returns) else 0.0
    return float(daily_returns.mean() / daily_std * (observations_per_year**0.5)) if daily_std > 0 else None


def _frozen_engine(nav: pd.Series, returns: pd.Series) -> tuple[float, float]:
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() * 252) / vol) if vol > 1e-12 else 0.0
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return sharpe, float(drawdown.min()) if not drawdown.empty else 0.0


def _frozen_mol_sharpe(returns: pd.Series, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(returns.mean() / std * (periods ** 0.5))


def _frozen_returns_max_dd(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(abs(dd.min()))


def _frozen_beta_sharpe(daily: pd.Series, rf: float = 0.0, ann: int = 244) -> float:
    d = daily.dropna()
    if len(d) < 2 or d.std() < 1e-12:
        return float("nan")
    return float((d.mean() - rf / ann) / d.std() * np.sqrt(ann))


def _frozen_beta_max_dd(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    return float(abs((nav / nav.cummax() - 1).min()))


# ---- equivalence -------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_strict_v8_metrics_unchanged(seed: int) -> None:
    from quantagent.backtest.strict_v8 import _compute_metrics

    returns = _returns(seed).dropna()
    nav = 1_000_000.0 * (1.0 + returns.clip(lower=-0.5)).cumprod()
    metrics = _compute_metrics(nav, pd.DataFrame(), initial_nav=1_000_000.0)
    nav_clean = nav.sort_index().astype(float)
    daily_ret = nav_clean.pct_change()
    daily_ret.iloc[0] = float(nav_clean.iloc[0] / 1_000_000.0 - 1.0)
    elapsed_years = max((max(1, (nav_clean.index[-1] - nav_clean.index[0]).days + 1)) / 365.25, 1.0 / 252)
    obs_per_year = len(daily_ret) / elapsed_years
    expected_sharpe = _frozen_strict_sharpe(daily_ret, obs_per_year) if len(daily_ret) >= 2 else 0.0
    assert _close(metrics.sharpe, expected_sharpe)
    assert _close(metrics.max_drawdown, _frozen_strict_max_dd(1_000_000.0, nav_clean))


@pytest.mark.parametrize("seed", SEEDS)
def test_paper_report_sharpe_and_drawdown_unchanged(seed: int) -> None:
    from quantagent.backtest import paper_report

    returns = _returns(seed).dropna()
    nav = 1_000_000.0 * (1.0 + returns.clip(lower=-0.5)).cumprod()
    pnl = paper_report._pnl_frame(nav, 1_000_000.0, None, None)
    assert np.array_equal(pnl["drawdown"].to_numpy(), (pnl["nav"] / pnl["nav"].cummax() - 1.0).to_numpy())
    summary = paper_report._summary(pnl, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 1_000_000.0)
    dates = pd.to_datetime(pnl["trade_date"])
    elapsed_years = max(max(1, (dates.iloc[-1] - dates.iloc[0]).days + 1) / 365.25, 1.0 / 252.0)
    daily = pd.to_numeric(pnl["daily_return"], errors="coerce").dropna()
    assert _close(summary["sharpe"], _frozen_paper_sharpe(daily, len(daily) / elapsed_years))


@pytest.mark.parametrize("seed", SEEDS)
def test_engine_performance_report_unchanged(seed: int) -> None:
    from quantagent.backtest.engine import _performance_report

    returns = _returns(seed).dropna()
    nav = 1_000_000.0 * (1.0 + returns.clip(lower=-0.5)).cumprod()
    report = _performance_report(nav, returns, pd.DataFrame(), pd.DataFrame())
    sharpe, max_dd = _frozen_engine(nav, returns)
    assert _close(report["sharpe"], sharpe)
    assert _close(report["max_drawdown"], max_dd)


@pytest.mark.parametrize("seed", SEEDS)
def test_multi_objective_and_sleeve_helpers_unchanged(seed: int) -> None:
    from quantagent.optimization.multi_objective_loss import _max_drawdown as mol_dd
    from quantagent.optimization.multi_objective_loss import _sharpe
    from quantagent.portfolio.walk_forward_sleeve_allocator import _max_drawdown as sleeve_dd

    returns = _returns(seed)
    assert _close(_sharpe(returns.dropna()), _frozen_mol_sharpe(returns.dropna()))
    assert _close(mol_dd(returns), _frozen_returns_max_dd(returns))
    assert _close(sleeve_dd(returns), _frozen_returns_max_dd(returns))
    assert mol_dd(returns.iloc[0:0]) == 0.0 and sleeve_dd(returns.iloc[0:0]) == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_beta_decomposition_unchanged(seed: int) -> None:
    from quantagent.backtest import beta_decomposition as bd

    returns = _returns(seed)
    for rf in (0.0, 0.02):
        assert _close(bd.sharpe(returns, rf=rf), _frozen_beta_sharpe(returns, rf=rf, ann=bd.ANN))
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    assert _close(bd.max_drawdown(nav), _frozen_beta_max_dd(nav))
    assert np.isnan(bd.max_drawdown(nav.iloc[0:0]))


def test_sharpe_ddof_is_explicit() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.003])
    assert sharpe_ratio(r, periods_per_year=252, ddof=0) == pytest.approx(
        sharpe_ratio(r, periods_per_year=252) * np.sqrt(4 / 3)
    )
    assert max_drawdown(pd.Series([1.0, 0.8, 0.9, 0.7])) == pytest.approx(-0.3)
