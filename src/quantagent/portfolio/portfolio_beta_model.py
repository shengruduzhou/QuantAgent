"""Portfolio beta estimation.

The hedge decision engine needs to know how much of the portfolio's
volatility is explained by broad market or sector exposure before it can
size a hedge. This module produces three views:

* ``estimate_symbol_betas`` — OLS regression of each symbol's daily
  returns against a benchmark return series.
* ``portfolio_beta`` — weighted aggregate of symbol betas (or computed
  directly from the portfolio return series when available).
* ``estimate_sector_betas`` — per-sector exposure to a sector benchmark.

The functions are intentionally dependency-free. When the price history
is too short or has too many NaNs the function returns a zero beta with
the appropriate diagnostic flag instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaEstimate:
    beta: float
    r_squared: float
    sample_count: int
    confidence: float


@dataclass(frozen=True)
class PortfolioBeta:
    portfolio_beta: float
    benchmark_symbol: str
    sample_count: int
    sector_betas: dict[str, BetaEstimate]
    diagnostics: dict[str, float]


def estimate_symbol_betas(
    symbol_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    min_overlap: int = 30,
) -> dict[str, BetaEstimate]:
    """OLS beta of every column in ``symbol_returns`` against ``benchmark_returns``."""

    if symbol_returns is None or symbol_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return {}
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
    if benchmark.empty:
        return {}
    panel = symbol_returns.copy().apply(pd.to_numeric, errors="coerce")
    aligned_benchmark = benchmark.reindex(panel.index).dropna()
    panel = panel.loc[aligned_benchmark.index]
    out: dict[str, BetaEstimate] = {}
    for symbol in panel.columns:
        series = panel[symbol].dropna()
        if len(series) < min_overlap:
            continue
        joined = pd.concat({"x": aligned_benchmark.loc[series.index], "y": series}, axis=1).dropna()
        if len(joined) < min_overlap:
            continue
        beta, r2 = _ols(joined["x"].to_numpy(), joined["y"].to_numpy())
        sample = int(len(joined))
        confidence = float(min(0.95, 0.30 + 0.65 * r2)) if r2 >= 0.0 else 0.20
        out[symbol] = BetaEstimate(beta=beta, r_squared=r2, sample_count=sample, confidence=confidence)
    return out


def portfolio_beta(
    target_weights: dict[str, float],
    symbol_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    sector_map: dict[str, str] | None = None,
    sector_returns: pd.DataFrame | None = None,
    benchmark_symbol: str = "000300.SH",
    min_overlap: int = 30,
) -> PortfolioBeta:
    """Compute the portfolio beta given target weights and a price history."""

    symbol_betas = estimate_symbol_betas(symbol_returns, benchmark_returns, min_overlap=min_overlap)
    if not target_weights or not symbol_betas:
        return PortfolioBeta(
            portfolio_beta=0.0,
            benchmark_symbol=benchmark_symbol,
            sample_count=0,
            sector_betas={},
            diagnostics={"fallback": 1.0},
        )
    weights_sum = sum(abs(value) for value in target_weights.values()) or 1.0
    weighted_beta = 0.0
    coverage = 0.0
    for symbol, weight in target_weights.items():
        estimate = symbol_betas.get(symbol)
        if estimate is None:
            continue
        weighted_beta += estimate.beta * weight / weights_sum
        coverage += abs(weight) / weights_sum
    sector_betas: dict[str, BetaEstimate] = {}
    if sector_returns is not None and not sector_returns.empty:
        sector_betas = estimate_symbol_betas(sector_returns, benchmark_returns, min_overlap=min_overlap)
    diagnostics = {
        "weight_coverage": float(coverage),
        "symbol_betas_resolved": float(len(symbol_betas)),
    }
    return PortfolioBeta(
        portfolio_beta=float(weighted_beta),
        benchmark_symbol=benchmark_symbol,
        sample_count=int(max((est.sample_count for est in symbol_betas.values()), default=0)),
        sector_betas=sector_betas,
        diagnostics=diagnostics,
    )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denom = float(np.dot(dx, dx))
    if denom <= 0.0:
        return 0.0, 0.0
    beta = float(np.dot(dx, dy) / denom)
    fitted = beta * dx + y_mean
    ss_tot = float(np.dot(dy, dy))
    ss_res = float(np.dot(y - fitted, y - fitted))
    r2 = 0.0 if ss_tot <= 0.0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return beta, r2


def returns_panel_from_close(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Turn a long-form ``(trade_date, symbol, close)`` panel into a wide return frame."""

    if price_panel is None or price_panel.empty:
        return pd.DataFrame()
    data = price_panel.copy()
    if {"trade_date", "symbol", "close"}.issubset(data.columns) is False:
        return pd.DataFrame()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"]).sort_values(["symbol", "trade_date"])
    data["ret"] = data.groupby("symbol")["close"].pct_change()
    pivot = data.pivot_table(index="trade_date", columns="symbol", values="ret", aggfunc="last")
    return pivot.sort_index()


def benchmark_returns_from_close(price_panel: pd.DataFrame, benchmark_symbol: str) -> pd.Series:
    if price_panel is None or price_panel.empty or benchmark_symbol not in set(price_panel.get("symbol", [])):
        return pd.Series(dtype=float)
    slice_ = price_panel[price_panel["symbol"] == benchmark_symbol].copy()
    slice_["trade_date"] = pd.to_datetime(slice_["trade_date"], errors="coerce")
    slice_ = slice_.dropna(subset=["trade_date"]).sort_values("trade_date")
    slice_["ret"] = slice_["close"].pct_change()
    return slice_.set_index("trade_date")["ret"].dropna()
